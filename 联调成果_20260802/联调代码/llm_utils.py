"""
LLM toolchain for module C (interfaces 33-35 / 旧编号 31-33).

Aligned with:
  - 全函数接口表 20260718_v2
  - Campus multi-group chat scenario (DORM/CLASS/MAJOR/CAMPUS)
  - Roles: ORDINARY / ACTIVE / RATIONAL / CONTROLLER
  - MessageType: original / forward / paraphrase / exaggerate / clarification
  - ActionType: SEND_MESSAGE / REPLY / FORWARD / SILENT / MUTE / ANNOUNCE

B module call path:
  build_prompt(belief, memory, env_info: Dict) -> chat -> parse_llm_response
  uses: opinion_updates, emotion_delta, action_type, target_id,
        source_message_id/source_group_id/destination_group_id
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Protocol


# ---------------------------------------------------------------------------
# Action / message vocab (string layer; enum resolved if types_def available)
# ---------------------------------------------------------------------------

ACTION_NAMES = ("SEND_MESSAGE", "REPLY", "FORWARD", "SILENT", "MUTE", "ANNOUNCE")
ACTION_CODE = {
    "SEND_MESSAGE": 0,
    "REPLY": 1,
    "FORWARD": 2,
    "SILENT": 3,
    "MUTE": 4,
    "ANNOUNCE": 5,
}

# legacy aliases -> v2/v3 ActionType
_LEGACY_ACTION = {
    "POST": "SEND_MESSAGE",
    "COMMENT": "REPLY",
    "REPOST": "FORWARD",
    "LIKE": "SILENT",
    "SEND_MESSAGE": "SEND_MESSAGE",
    "REPLY": "REPLY",
    "FORWARD": "FORWARD",
    "SILENT": "SILENT",
    "MUTE": "MUTE",
    "ANNOUNCE": "ANNOUNCE",
}

MESSAGE_TYPES = (
    "original",
    "forward",
    "paraphrase",
    "exaggerate",
    "clarification",
)

ROLE_HINTS = {
    "ORDINARY": (
        "你是普通群员：多数时候 SILENT 或简短 REPLY；"
        "若 ENV.recent_messages 非空且话题热/负面偏高，可偶尔 FORWARD 到你的另一个群；"
        "很少 SEND_MESSAGE 发全新内容；不要夸大，不要扮演管理者。"
    ),
    "ACTIVE": (
        "你是活跃讨论者：当 ENV.recent_messages 非空时，优先考虑 FORWARD "
        "（跨群扩散）或 paraphrase 转发，而不是只发新消息；"
        "可以带一点情绪；message_type 可用 forward / paraphrase / exaggerate。"
    ),
    "RATIONAL": (
        "你是理性讨论者：优先质疑来源、要求证据；"
        "可用 FORWARD+paraphrase 把冷静转述带到其他群，或 REPLY 提醒勿传谣；"
        "避免 exaggerate。"
    ),
    "CONTROLLER": (
        "你是群管理者：仅当 topic_negative/热度明显偏高时才 ANNOUNCE 或 MUTE；"
        "常态不要每一步都公告；无干预必要时可 SILENT 或简短 REPLY；"
        "ANNOUNCE 的 message_type 必须是 clarification；"
        "MUTE 必须给出 target_id；宿舍群(DORM)通常不进行官方干预。"
    ),
}

GROUP_HINTS = {
    "DORM": "当前在宿舍群：熟人小圈、管理很弱、曝光低；不适合强硬官方公告。",
    "CLASS": "当前在班级群：弱管控，班委可提醒，传播范围中等。",
    "MAJOR": "当前在专业群：有一定管控，辅导员/专业负责人权威较高。",
    "CAMPUS": "当前在校园群：管理强、曝光高，澄清更正式，影响面更大。",
}


class LLMClient(Protocol):
    def chat(self, prompt: str, *, user_id: Optional[str] = None) -> str:
        ...


class MockLLMClient:
    """Offline client: returns a safe silent JSON for B's belief update path."""

    def chat(self, prompt: str, *, user_id: Optional[str] = None) -> str:
        _ = prompt
        _ = user_id
        return json.dumps(
            {
                "action_type": "SILENT",
                "message_type": "original",
                "content": "",
                "topic_id": "",
                "opinion_value": 0.0,
                "target_id": None,
                "source_message_id": None,
                "source_group_id": "",
                "destination_group_id": "",
                "opinion_updates": {},
                "emotion_delta": {"valence": 0.0, "arousal": 0.0},
            },
            ensure_ascii=False,
        )


class OpenAICompatibleClient:
    """
    DeepSeek / OpenAI-compatible client for campus sim.

    llm_config knobs for speed/realism A/B:
      max_tokens, json_mode, thinking, max_concurrency,
      user_id_prefix, max_prompt_chars, max_memory_items, max_recent_messages
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        import threading

        # 官网：deepseek-chat / deepseek-reasoner 将于 2026-07-24 弃用；
        # 默认用 deepseek-v4-flash（对应原 chat 的非思考模式）。
        self.model = str(
            config.get("model")
            or os.getenv("DEEPSEEK_MODEL")
            or "deepseek-v4-flash"
        )
        self.timeout = float(config.get("timeout", 60))
        self.max_retries = int(config.get("max_retries", 2))
        self.temperature = float(config.get("temperature", 0.3))
        self.base_url = str(config.get("base_url") or "https://api.deepseek.com")
        # V4 默认常开 thinking；仿真要稳定 JSON，默认关掉（可用 thinking=True 打开）
        self.thinking = bool(config.get("thinking", False))
        self.max_tokens = int(config.get("max_tokens", 512))
        self.json_mode = bool(config.get("json_mode", True))
        self.user_id_prefix = str(config.get("user_id_prefix", "agent"))
        self.max_prompt_chars = int(config.get("max_prompt_chars", 8000))
        self.max_memory_items = int(config.get("max_memory_items", 5))
        self.max_recent_messages = int(config.get("max_recent_messages", 5))
        # 客户端并发闸（账号 flash 上限 2500；同账号多 Key 不加倍）
        self.max_concurrency = max(1, int(config.get("max_concurrency", 16)))
        self._sem = threading.Semaphore(self.max_concurrency)
        self.last_usage: Dict[str, Any] = {}
        self.call_stats: Dict[str, Any] = {
            "n_calls": 0,
            "n_empty": 0,
            "n_retry": 0,
            "n_429": 0,
            "seconds_total": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
        }
        self.api_key = str(
            config.get("api_key") or os.getenv("DEEPSEEK_API_KEY", "")
        ).strip()
        if not self.api_key:
            raise ValueError(
                "Missing API key. Put it in llm_config['api_key'] "
                "or env DEEPSEEK_API_KEY."
            )

        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from exc

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @staticmethod
    def sanitize_user_id(raw: str) -> str:
        """DeepSeek: user_id must match [a-zA-Z0-9\\-_]+ , len <= 512."""
        cleaned = re.sub(r"[^a-zA-Z0-9\-_]", "-", str(raw or "anon"))
        return (cleaned or "anon")[:512]

    def chat(self, prompt: str, *, user_id: Optional[str] = None) -> str:
        if len(prompt) > self.max_prompt_chars:
            prompt = prompt[: self.max_prompt_chars] + "\n...(truncated)"

        uid = self.sanitize_user_id(
            user_id or f"{self.user_id_prefix}-default"
        )
        system_content = (
            "你是校园群聊舆情仿真中的智能体决策模块。"
            "只输出一个合法 JSON 对象，不要 Markdown，不要解释。"
            "Output must be a single json object matching the schema in the user message."
        )
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            acquired = self._sem.acquire(timeout=max(self.timeout, 1.0))
            if not acquired:
                last_error = TimeoutError("LLM concurrency semaphore timeout")
                continue
            try:
                extra_body: Dict[str, Any] = {
                    "thinking": {
                        "type": "enabled" if self.thinking else "disabled"
                    },
                    "user_id": uid,
                }
                create_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": self.max_tokens,
                    "extra_body": extra_body,
                }
                # 思考模式下 temperature 等不生效；非思考模式才传
                if not self.thinking:
                    create_kwargs["temperature"] = self.temperature
                if self.json_mode:
                    create_kwargs["response_format"] = {"type": "json_object"}

                resp = self._client.chat.completions.create(**create_kwargs)
                content = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)
                self._record_usage(usage, time.perf_counter() - t0, empty=not content)
                if not content:
                    self.call_stats["n_empty"] += 1
                    last_error = RuntimeError("empty LLM content")
                    if attempt < self.max_retries:
                        self.call_stats["n_retry"] += 1
                        time.sleep(0.7 * (attempt + 1))
                        continue
                    return ""
                return content
            except Exception as exc:
                last_error = exc
                err_s = str(exc)
                if "429" in err_s:
                    self.call_stats["n_429"] += 1
                if attempt >= self.max_retries:
                    break
                self.call_stats["n_retry"] += 1
                delay = (2.0 if "429" in err_s else 0.7) * (attempt + 1)
                time.sleep(delay)
            finally:
                self._sem.release()
        raise RuntimeError(f"LLM request failed after retries: {last_error}")

    def _record_usage(self, usage: Any, seconds: float, empty: bool) -> None:
        self.call_stats["n_calls"] += 1
        self.call_stats["seconds_total"] += float(seconds)
        if usage is None:
            self.last_usage = {"seconds": round(seconds, 4), "empty": empty}
            return
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        hit = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        miss = int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0)
        self.call_stats["prompt_tokens"] += prompt_tokens
        self.call_stats["completion_tokens"] += completion_tokens
        self.call_stats["cache_hit_tokens"] += hit
        self.call_stats["cache_miss_tokens"] += miss
        self.last_usage = {
            "seconds": round(seconds, 4),
            "empty": empty,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_cache_hit_tokens": hit,
            "prompt_cache_miss_tokens": miss,
        }


def setup_llm_client(llm_config: Dict[str, Any]) -> LLMClient:
    """
    Factory for LLM client.
    provider: mock | deepseek | openai | openai_compatible
    """
    provider = str(llm_config.get("provider", "mock")).lower()
    if provider == "mock":
        return MockLLMClient()
    if provider in {"deepseek", "openai", "openai_compatible"}:
        # 真客户端初始化失败时默认抛错，避免 silent 假成功。
        # 若联调需要降级：llm_config["fallback_mock"]=True
        try:
            return OpenAICompatibleClient(llm_config)
        except Exception:
            if bool(llm_config.get("fallback_mock", False)):
                return MockLLMClient()
            raise
    return MockLLMClient()


class LLMUtils:
    """Facade for B: model.llm_utils.build_prompt / get_client / parse_llm_response."""

    def __init__(self, llm_config: Dict[str, Any]) -> None:
        self._client = setup_llm_client(llm_config)

    def get_client(self) -> LLMClient:
        return self._client

    @staticmethod
    def build_prompt(
        belief: Any,
        memory: List[Any],
        env_info: Any,
        budget: Optional[Dict[str, Any]] = None,
    ) -> str:
        return build_prompt(belief, memory, env_info, budget=budget)

    @staticmethod
    def parse_llm_response(raw: str) -> Dict[str, Any]:
        return parse_llm_response(raw)


def build_prompt(
    belief: Any,
    memory: List[Any],
    env_info: Any,
    budget: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build campus-group-chat BDI prompt.

    env_info (from B) typically includes:
      group_type, beta, role, nickname,
      recent_messages, mentions, topic_heat, topic_negative

    budget (teacher window control):
      max_memory_items, max_recent_messages, max_prompt_chars
    """
    budget = budget or {}
    max_mem = int(budget.get("max_memory_items", 5))
    max_recent = int(budget.get("max_recent_messages", 5))
    max_chars = int(budget.get("max_prompt_chars", 8000))

    belief_dict = _to_plain_dict(belief)
    # belief 过大时只保留 identity / emotion / 少量 opinions
    if isinstance(belief_dict, dict) and "opinions" in belief_dict:
        ops = belief_dict.get("opinions") or {}
        if isinstance(ops, dict) and len(ops) > 8:
            # 保留最近插入顺序的尾部若干 topic
            keys = list(ops.keys())[-8:]
            belief_dict["opinions"] = {k: ops[k] for k in keys}

    mem_list = [_to_plain_dict(m) for m in (memory or [])[-max_mem:]]
    for m in mem_list:
        if isinstance(m, dict) and "content" in m:
            m["content"] = str(m.get("content", ""))[:120]

    env_dict = _to_plain_dict(env_info) if not isinstance(env_info, dict) else dict(env_info)
    recent = env_dict.get("recent_messages")
    if isinstance(recent, list):
        env_dict["recent_messages"] = recent[:max_recent]
        for item in env_dict["recent_messages"]:
            if isinstance(item, dict) and "content" in item:
                item["content"] = str(item.get("content", ""))[:160]

    role = str(
        env_dict.get("role")
        or _dig(belief_dict, "identity", "agent_type")
        or "ORDINARY"
    ).upper()
    group = str(
        env_dict.get("group_type")
        or _dig(belief_dict, "identity", "group_type")
        or "CLASS"
    ).upper()

    # normalize enum-like values "AgentType.ACTIVE" / "1"
    role = role.replace("AGENTTYPE.", "").split(".")[-1]
    group = group.replace("GROUPTYPE.", "").split(".")[-1]
    if role.isdigit():
        role = {0: "ORDINARY", 1: "ACTIVE", 2: "RATIONAL", 3: "CONTROLLER"}.get(
            int(role), "ORDINARY"
        )
    if group.isdigit():
        group = {0: "DORM", 1: "CLASS", 2: "MAJOR", 3: "CAMPUS"}.get(int(group), "CLASS")

    role_hint = ROLE_HINTS.get(role, ROLE_HINTS["ORDINARY"])
    group_hint = GROUP_HINTS.get(group, GROUP_HINTS["CLASS"])

    schema = {
        "action_type": "SEND_MESSAGE|REPLY|FORWARD|SILENT|MUTE|ANNOUNCE",
        "message_type": "original|forward|paraphrase|exaggerate|clarification",
        "content": "string, campus WeChat-style short message; empty if SILENT/MUTE",
        "topic_id": "string, e.g. T001",
        "opinion_value": "float in [-1,1]",
        "target_id": "int|null; FORWARD/REPLY 时为源消息作者；MUTE 时为被禁言 agent",
        "source_message_id": "string|null; 必须从 ENV.recent_messages 选择",
        "source_group_id": "string; 源消息所在群",
        "destination_group_id": "string; 实际发入的群，必须属于 ENV.group_ids",
        "opinion_updates": {"T001": "float in [-1,1]"},
        "emotion_delta": {"valence": "float", "arousal": "float"},
    }

    n_recent_avail = (
        len(env_dict["recent_messages"])
        if isinstance(env_dict.get("recent_messages"), list)
        else 0
    )

    rules = [
        "1) 只输出一个 JSON 对象，不要代码块，不要额外文字。",
        "2) 这是校园多层群聊舆情仿真：消息会在多群之间转发扩散。",
        "3) 若 action_type=SILENT，content 置空，message_type 可填 original。",
        "4) Controller：仅在负面/热度偏高时 ANNOUNCE（message_type=clarification）或 MUTE（填 target_id）；"
        "不要无必要连发公告。非 CONTROLLER 不得 MUTE/ANNOUNCE。",
        "5) 【转发优先】当 ENV.recent_messages 非空时："
        "ACTIVE 应优先 FORWARD；ORDINARY/RATIONAL 也要有一定概率 FORWARD；"
        "不要总是 SEND_MESSAGE 或 SILENT。",
        "6) FORWARD 字段填写："
        "source_message_id / source_group_id 必须来自 ENV.recent_messages 某条真实消息；"
        "destination_group_id 必须在 ENV.group_ids 中，且与 source_group_id 不同（跨群）；"
        "target_id 填该消息的 source_id；message_type 用 forward 或 paraphrase。",
        "7) 若你属于多个群（ENV.group_ids 长度>1），FORWARD 的目标优先选尚未出现该话题的更大/其他群。",
        "8) REPLY：destination_group_id=被回复消息的 group_id；target_id=作者 source_id。",
        "9) SEND_MESSAGE/ANNOUNCE 的 destination_group_id 必须在 ENV.group_ids 中；禁止编造 ID。",
        "10) opinion_updates / emotion_delta 幅度宜小（如 |delta|<=0.3）。",
        f"11) 当前 ENV.recent_messages 条数={n_recent_avail}；"
        + ("有可转发消息，请认真考虑 FORWARD。" if n_recent_avail > 0 else "暂无可转发消息。"),
        f"12) 角色约束：{role_hint}",
        f"13) 主群约束：{group_hint}",
    ]

    # 顺序：规则+schema 固定在前，ENV 次之，belief/memory 在后——超预算时先截断后部，避免切掉 schema
    head = (
        "【任务】根据信念、记忆与当前群聊环境，更新观点/情绪，并给出下一步行动。"
        "重点：有可转发消息时尽量产生跨群 FORWARD，以形成真实扩散。\n\n"
        + "【规则】\n"
        + "\n".join(rules)
        + "\n\n"
        + f"【OUTPUT JSON SCHEMA】\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        + f"【ENV】\n{json.dumps(env_dict, ensure_ascii=False)}\n\n"
    )
    tail = (
        f"【MEMORY last{max_mem}】\n{json.dumps(mem_list, ensure_ascii=False)}\n\n"
        + f"【BELIEF】\n{json.dumps(belief_dict, ensure_ascii=False)}\n"
    )
    text = head + tail
    if len(text) > max_chars:
        # 保头（规则+ENV+schema），截尾（memory/belief）
        keep = max_chars - len(head) - 32
        if keep > 200:
            text = head + tail[:keep] + "\n...(truncated belief/memory)"
        else:
            text = head[: max_chars - 20] + "\n...(truncated)"
    return text


def parse_llm_response(raw: str) -> Dict[str, Any]:
    """
    Parse/validate LLM JSON.
    Always returns a normalized dict; invalid -> silent fallback.
    """
    text = (raw or "").strip()
    if not text:
        return _silent_response()

    parsed = _try_parse_json(text)
    if isinstance(parsed, dict):
        return _normalize_response(parsed)

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        parsed = _try_parse_json(fence.group(1))
        if isinstance(parsed, dict):
            return _normalize_response(parsed)

    brace = re.search(r"\{[\s\S]*\}", text, flags=re.S)
    if brace:
        parsed = _try_parse_json(brace.group(0))
        if isinstance(parsed, dict):
            return _normalize_response(parsed)

    return _silent_response()


def _try_parse_json(candidate: str) -> Optional[Any]:
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _normalize_response(data: Dict[str, Any]) -> Dict[str, Any]:
    out = _silent_response()

    # ---- action_type ----
    raw_action = data.get("action_type", "SILENT")
    if isinstance(raw_action, (int, float)):
        action_name = {
            0: "SEND_MESSAGE",
            1: "REPLY",
            2: "FORWARD",
            3: "SILENT",
            4: "MUTE",
            5: "ANNOUNCE",
        }.get(int(raw_action), "SILENT")
    else:
        key = str(raw_action).upper().strip()
        action_name = _LEGACY_ACTION.get(key, "SILENT")
        if action_name not in ACTION_NAMES:
            action_name = "SILENT"

    out["action_type"] = _to_action_enum_or_name(action_name)
    out["action_type_name"] = action_name
    out["action_type_code"] = ACTION_CODE[action_name]

    # ---- message_type ----
    mt = str(data.get("message_type", out["message_type"])).lower().strip()
    if mt not in MESSAGE_TYPES:
        # heuristics
        if action_name == "FORWARD":
            mt = "forward"
        elif action_name == "SILENT":
            mt = "original"
        else:
            mt = "original"
    # Controller-style clarification override if content suggests 澄清 and role not forced
    if mt == "clarification" or "clarif" in mt:
        mt = "clarification"
    out["message_type"] = mt

    # ---- content / topic ----
    out["content"] = str(data.get("content", out["content"]))
    topic_id = data.get("topic_id", data.get("event_id", out["topic_id"]))
    out["topic_id"] = str(topic_id or "")
    # keep event_id alias for older callers
    out["event_id"] = out["topic_id"]

    try:
        out["opinion_value"] = _clip(
            float(data.get("opinion_value", out["opinion_value"])), -1.0, 1.0
        )
    except Exception:
        pass

    target_id = data.get("target_id", None)
    if target_id is None or target_id == "" or str(target_id).lower() == "null":
        out["target_id"] = None
    else:
        try:
            out["target_id"] = int(target_id)
        except Exception:
            out["target_id"] = None

    source_message_id = data.get("source_message_id", None)
    if source_message_id is None or str(source_message_id).strip().lower() in {"", "null"}:
        out["source_message_id"] = None
    else:
        out["source_message_id"] = str(source_message_id).strip()

    out["source_group_id"] = str(data.get("source_group_id", "") or "").strip()
    out["destination_group_id"] = str(
        data.get("destination_group_id", data.get("group_id", "")) or ""
    ).strip()

    # ---- opinion_updates (topic_id -> float) ----
    opinion_updates = data.get("opinion_updates", {})
    if isinstance(opinion_updates, dict):
        clean: Dict[str, float] = {}
        for k, v in opinion_updates.items():
            try:
                clean[str(k)] = _clip(float(v), -1.0, 1.0)
            except Exception:
                continue
        out["opinion_updates"] = clean

    emotion_delta = data.get("emotion_delta", {})
    if isinstance(emotion_delta, dict):
        out["emotion_delta"] = {
            "valence": _clip(_safe_float(emotion_delta.get("valence", 0.0), 0.0), -1.0, 1.0),
            "arousal": _clip(_safe_float(emotion_delta.get("arousal", 0.0), 0.0), -1.0, 1.0),
        }

    # silent / mute consistency
    if action_name in ("SILENT", "MUTE"):
        if action_name == "SILENT":
            out["content"] = ""
    if action_name == "ANNOUNCE" and out["message_type"] != "clarification":
        out["message_type"] = "clarification"

    return out


def _silent_response() -> Dict[str, Any]:
    action_name = "SILENT"
    return {
        "action_type": _to_action_enum_or_name(action_name),
        "action_type_name": action_name,
        "action_type_code": ACTION_CODE[action_name],
        "message_type": "original",
        "content": "",
        "topic_id": "",
        "event_id": "",
        "opinion_value": 0.0,
        "target_id": None,
        "source_message_id": None,
        "source_group_id": "",
        "destination_group_id": "",
        "opinion_updates": {},
        "emotion_delta": {"valence": 0.0, "arousal": 0.0},
    }


def _to_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_dict(v) for v in value]
    if hasattr(value, "name") and hasattr(value, "value") and not isinstance(value, type):
        # Enum-like
        try:
            return value.name  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            str(k): _to_plain_dict(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return value


def _dig(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_action_enum_or_name(action_name: str) -> Any:
    """Prefer ActionType enum from types_def if importable."""
    for module_name in ("types_def", "sim_types"):
        try:
            module = __import__(module_name, fromlist=["ActionType"])
            action_enum = getattr(module, "ActionType", None)
            if action_enum is not None and hasattr(action_enum, action_name):
                return action_enum[action_name]
        except Exception:
            continue
    return action_name
