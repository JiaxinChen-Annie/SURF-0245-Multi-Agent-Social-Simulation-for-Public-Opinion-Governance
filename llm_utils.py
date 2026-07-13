"""
LLM toolchain for module C (functions 31-33).

Includes:
- setup_llm_client
- build_prompt
- parse_llm_response
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Protocol


class LLMClient(Protocol):
    def chat(self, prompt: str) -> str:
        ...


class MockLLMClient:
    def chat(self, prompt: str) -> str:
        # Safe default for simulation continuity.
        _ = prompt
        return json.dumps(
            {
                "action_type": "SILENT",
                "content": "",
                "event_id": "",
                "opinion_value": 0.0,
                "target_id": None,
                "opinion_updates": {},
                "emotion_delta": {"valence": 0.0, "arousal": 0.0},
            },
            ensure_ascii=True,
        )


class OpenAICompatibleClient:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.model = str(config.get("model", "deepseek-v4-flash"))
        self.timeout = float(config.get("timeout", 30))
        self.max_retries = int(config.get("max_retries", 2))
        self.temperature = float(config.get("temperature", 0.3))

        # DeepSeek-friendly defaults
        self.base_url = str(config.get("base_url") or "https://api.deepseek.com")
        self.api_key = str(config.get("api_key") or os.getenv("DEEPSEEK_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError(
                "Missing API key. Put it in llm_config['api_key'] or env DEEPSEEK_API_KEY."
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
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
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
    Supported mode:
    - provider = "mock"
    - provider in {"deepseek", "openai", "openai_compatible"} with OpenAI SDK
    """
    provider = str(llm_config.get("provider", "mock")).lower()
    if provider == "mock":
        return MockLLMClient()
    if provider in {"deepseek", "openai", "openai_compatible"}:
        try:
            return OpenAICompatibleClient(llm_config)
        except Exception:
            # graceful downgrade keeps simulation running even when API is unavailable
            return MockLLMClient()
    return MockLLMClient()


class LLMUtils:
    """
    Facade for B module integration style:
    - build_prompt(...)
    - parse_llm_response(...)
    - get_client()
    """

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
    Build a stable reasoning prompt from belief + memory + environment.
    Works with dataclass objects or plain dict-like structures.
    """
    belief_dict = _to_plain_dict(belief)
    mem_list = [_to_plain_dict(m) for m in memory[-5:]]
    env_dict = _to_plain_dict(env_info)

    schema = {
        "action_type": "POST|COMMENT|REPOST|LIKE|SILENT",
        "content": "string",
        "event_id": "string",
        "opinion_value": "float in [-1,1]",
        "target_id": "int|null",
        "opinion_updates": {"event_id": "float in [-1,1]"},
        "emotion_delta": {"valence": "float", "arousal": "float"},
    }

    return (
        "You are a social simulation agent.\n"
        "Given your belief, memory, and current environment, decide the next action.\n"
        "Output ONLY one JSON object.\n\n"
        f"BELIEF:\n{json.dumps(belief_dict, ensure_ascii=False)}\n\n"
        f"MEMORY (last 5):\n{json.dumps(mem_list, ensure_ascii=False)}\n\n"
        f"ENV:\n{json.dumps(env_dict, ensure_ascii=False)}\n\n"
        f"REQUIRED JSON SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}\n"
    )


def parse_llm_response(raw: str) -> Dict[str, Any]:
    """
    Parse and validate LLM output.
    Returns normalized dict and falls back to silent action when invalid.
    """
    text = (raw or "").strip()
    if not text:
        return _silent_response()

    # 1) whole text as json
    parsed = _try_parse_json(text)
    if isinstance(parsed, dict):
        return _normalize_response(parsed)

    # 2) fenced json block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        parsed = _try_parse_json(fence.group(1))
        if isinstance(parsed, dict):
            return _normalize_response(parsed)

    # 3) first json object in noisy text
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

    raw_action = data.get("action_type", "SILENT")
    action_name = "SILENT"
    if isinstance(raw_action, (int, float)):
        action_map = {0: "POST", 1: "COMMENT", 2: "REPOST", 3: "LIKE", 4: "SILENT"}
        action_name = action_map.get(int(raw_action), "SILENT")
    else:
        action = str(raw_action).upper().strip()
        if action in {"POST", "COMMENT", "REPOST", "LIKE", "SILENT"}:
            action_name = action

    out["action_type"] = _to_action_enum_or_name(action_name)
    out["action_type_name"] = action_name
    out["action_type_code"] = {"POST": 0, "COMMENT": 1, "REPOST": 2, "LIKE": 3, "SILENT": 4}[action_name]

    out["content"] = str(data.get("content", out["content"]))
    out["event_id"] = str(data.get("event_id", out["event_id"]))

    try:
        out["opinion_value"] = _clip(float(data.get("opinion_value", out["opinion_value"])), -1.0, 1.0)
    except Exception:
        pass

    target_id = data.get("target_id", None)
    if target_id is None:
        out["target_id"] = None
    else:
        try:
            out["target_id"] = int(target_id)
        except Exception:
            out["target_id"] = None

    # Extra keys for the B version that updates beliefs directly from LLM output
    opinion_updates = data.get("opinion_updates", {})
    if isinstance(opinion_updates, dict):
        clean_updates: Dict[str, float] = {}
        for k, v in opinion_updates.items():
            try:
                clean_updates[str(k)] = _clip(float(v), -1.0, 1.0)
            except Exception:
                continue
        out["opinion_updates"] = clean_updates

    emotion_delta = data.get("emotion_delta", {})
    if isinstance(emotion_delta, dict):
        valence = _safe_float(emotion_delta.get("valence", 0.0), 0.0)
        arousal = _safe_float(emotion_delta.get("arousal", 0.0), 0.0)
        out["emotion_delta"] = {"valence": valence, "arousal": arousal}

    return out


def _silent_response() -> Dict[str, Any]:
    action_name = "SILENT"
    return {
        "action_type": _to_action_enum_or_name(action_name),
        "action_type_name": action_name,
        "action_type_code": 4,
        "content": "",
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
    if hasattr(value, "__dict__"):
        return {str(k): _to_plain_dict(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_action_enum_or_name(action_name: str) -> Any:
    """
    Prefer returning ActionType enum value when available in shared type modules.
    Falls back to action name string for maximal compatibility.
    """
    for module_name in ("types_def", "sim_types"):
        try:
            module = __import__(module_name, fromlist=["ActionType"])
            action_enum = getattr(module, "ActionType", None)
            if action_enum is not None:
                return action_enum[action_name]
        except Exception:
            continue
    return action_name
