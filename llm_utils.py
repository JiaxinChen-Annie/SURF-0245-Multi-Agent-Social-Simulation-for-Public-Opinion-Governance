"""
LLM toolchain for module C (interfaces 33-35 / 旧编号 31-33).

Aligned with:
  - 全函数接口表 20260718_v2
  - Campus multi-group chat scenario (DORM/CLASS/MAJOR/CAMPUS)
  - Roles: ORDINARY / ACTIVE / RATIONAL / CONTROLLER
  - MessageType: original / forward / paraphrase / exaggerate / clarification
  - ActionType: SEND_MESSAGE / REPLY / FORWARD / SILENT

B module call path:
  build_prompt(belief, memory, env_info: Dict) -> chat -> parse_llm_response
  uses: opinion_updates, emotion_delta
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

ACTION_NAMES = ("SEND_MESSAGE", "REPLY", "FORWARD", "SILENT")
ACTION_CODE = {"SEND_MESSAGE": 0, "REPLY": 1, "FORWARD": 2, "SILENT": 3}

# legacy aliases -> v2 ActionType
_LEGACY_ACTION = {
    "POST": "SEND_MESSAGE",
    "COMMENT": "REPLY",
    "REPOST": "FORWARD",
    "LIKE": "SILENT",
    "SEND_MESSAGE": "SEND_MESSAGE",
    "REPLY": "REPLY",
    "FORWARD": "FORWARD",
    "SILENT": "SILENT",
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
        "你是普通群员：多数时候保持沉默或简短附和；"
        "很少主动发新信息；不要夸大，不要扮演管理者。"
    ),
    "ACTIVE": (
        "你是活跃讨论者：更倾向于转发、转述或分享；"
        "可以带一点情绪，但输出仍必须是合法 JSON；"
        "若选择扩散，message_type 可用 forward / paraphrase / exaggerate。"
    ),
    "RATIONAL": (
        "你是理性讨论者：优先质疑来源、要求证据、提醒勿传未经核实信息；"
        "避免夸张；必要时用 paraphrase 做冷静转述，或建议澄清。"
    ),
    "CONTROLLER": (
        "你是群管理者：在热度/负面偏高时应降温；"
        "干预时 message_type 必须是 clarification，语气正式、克制；"
        "宿舍群(DORM)通常不进行官方干预。"
    ),
}

GROUP_HINTS = {
    "DORM": "当前在宿舍群：熟人小圈、管理很弱、曝光低；不适合强硬官方公告。",
    "CLASS": "当前在班级群：弱管控，班委可提醒，传播范围中等。",
    "MAJOR": "当前在专业群：有一定管控，辅导员/专业负责人权威较高。",
    "CAMPUS": "当前在校园群：管理强、曝光高，澄清更正式，影响面更大。",
}


class LLMClient(Protocol):
    def chat(self, prompt: str) -> str:
        ...


class MockLLMClient:
    """Offline client: returns a safe silent JSON for B's belief update path."""

    def chat(self, prompt: str) -> str:
        _ = prompt
        return json.dumps(
            {
                "action_type": "SILENT",
                "message_type": "original",
                "content": "",
                "topic_id": "",
                "opinion_value": 0.0,
                "target_id": None,
                "opinion_updates": {},
                "emotion_delta": {"valence": 0.0, "arousal": 0.0},
            },
            ensure_ascii=False,
        )


class OpenAICompatibleClient:
    def __init__(self, config: Dict[str, Any]) -> None:
        # 官网：deepseek-chat / deepseek-reasoner 将于 2026-07-24 弃用；
        # 默认用 deepseek-v4-flash（对应原 chat 的非思考模式）。
        self.model = str(
            config.get("model")
            or os.getenv("DEEPSEEK_MODEL")
            or "deepseek-v4-flash"
        )
        self.timeout = float(config.get("timeout", 30))
        self.max_retries = int(config.get("max_retries", 2))
        self.temperature = float(config.get("temperature", 0.3))
        self.base_url = str(config.get("base_url") or "https://api.deepseek.com")
        # V4 默认常开 thinking；仿真要稳定 JSON，默认关掉（可用 thinking=True 打开）
        self.thinking = bool(config.get("thinking", False))
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

    def chat(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                create_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是校园群聊舆情仿真中的智能体决策模块。"
                                "只输出一个合法 JSON 对象，不要 Markdown，不要解释。"
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                    # OpenAI SDK：DeepSeek 扩展字段走 extra_body
                    "extra_body": {
                        "thinking": {
                            "type": "enabled" if self.thinking else "disabled"
                        }
                    },
                }
                resp = self._client.chat.completions.create(**create_kwargs)
                content = resp.choices[0].message.content
                return content or ""
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.7 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after retries: {last_error}")


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
    def build_prompt(belief: Any, memory: List[Any], env_info: Any) -> str:
        return build_prompt(belief, memory, env_info)

    @staticmethod
    def parse_llm_response(raw: str) -> Dict[str, Any]:
        return parse_llm_response(raw)


def build_prompt(belief: Any, memory: List[Any], env_info: Any) -> str:
    """
    Build campus-group-chat BDI prompt.

    env_info (from B) typically includes:
      group_type, beta, role, nickname,
      recent_messages, mentions, topic_heat, topic_negative
    """
    belief_dict = _to_plain_dict(belief)
    mem_list = [_to_plain_dict(m) for m in (memory or [])[-5:]]
    env_dict = _to_plain_dict(env_info) if not isinstance(env_info, dict) else dict(env_info)

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
        "action_type": "SEND_MESSAGE|REPLY|FORWARD|SILENT",
        "message_type": "original|forward|paraphrase|exaggerate|clarification",
        "content": "string, campus WeChat-style short message; empty if SILENT",
        "topic_id": "string, e.g. T001",
        "opinion_value": "float in [-1,1]",
        "target_id": "int|null",
        "opinion_updates": {"T001": "float in [-1,1]"},
        "emotion_delta": {"valence": "float", "arousal": "float"},
    }

    rules = [
        "1) 只输出一个 JSON 对象，不要代码块，不要额外文字。",
        "2) 这是校园多层群聊舆情仿真：消息可能被转发/转述/夸大/澄清。",
        "3) 若 action_type=SILENT，content 置空，message_type 可填 original。",
        "4) Controller 干预时：action_type 建议 SEND_MESSAGE，message_type 必须 clarification。",
        "5) ACTIVE 扩散时可用 forward/paraphrase/exaggerate；RATIONAL 避免 exaggerate。",
        "6) opinion_updates 用于更新对 topic 的立场；emotion_delta 为相对变化量，幅度建议较小（如 |delta|<=0.3）。",
        "7) 结合 topic_heat / topic_negative：热度高且负面高时，更应降温或求证，而非继续夸大。",
        f"8) 角色约束：{role_hint}",
        f"9) 群约束：{group_hint}",
    ]

    return (
        "【任务】根据信念、记忆与当前群聊环境，更新你的观点/情绪倾向，并给出下一步行动意向。\n\n"
        + "【规则】\n"
        + "\n".join(rules)
        + "\n\n"
        + f"【BELIEF】\n{json.dumps(belief_dict, ensure_ascii=False)}\n\n"
        + f"【MEMORY last5】\n{json.dumps(mem_list, ensure_ascii=False)}\n\n"
        + f"【ENV】\n{json.dumps(env_dict, ensure_ascii=False)}\n\n"
        + f"【OUTPUT JSON SCHEMA】\n{json.dumps(schema, ensure_ascii=False)}\n"
    )


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
        action_name = {0: "SEND_MESSAGE", 1: "REPLY", 2: "FORWARD", 3: "SILENT"}.get(
            int(raw_action), "SILENT"
        )
        # also accept old 0..4 coding
        if int(raw_action) == 4:
            action_name = "SILENT"
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

    # silent consistency
    if action_name == "SILENT":
        out["content"] = ""

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
