"""
DeepSeek live smoke test for module C.

Usage (Anaconda Prompt / cmd):
  set DEEPSEEK_API_KEY=your_real_key
  python deepseek_smoke.py
"""

from __future__ import annotations

import os

from llm_utils import build_prompt, parse_llm_response, setup_llm_client


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is empty. Set it in your shell first.")

    client = setup_llm_client(
        {
            "provider": "deepseek",
            # 官网推荐：deepseek-v4-flash（默认）或 deepseek-v4-pro
            # deepseek-chat 将于 2026/07/24 弃用，勿再作为默认
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "base_url": "https://api.deepseek.com",
            "timeout": 30,
            "max_retries": 2,
            "temperature": 0.2,
            "thinking": False,  # 仿真要纯 JSON，关闭思考模式
        }
    )

    belief = {
        "identity": {
            "agent_type": "CONTROLLER",
            "group_type": "CAMPUS",
            "nickname": "辅导员助手",
        },
        "emotion": {"valence": -0.1, "arousal": 0.6},
        "opinions": {"T001": {"topic_id": "T001", "opinion_value": 0.0, "confidence": 0.6}},
    }
    memory = []
    env = {
        "group_type": "CAMPUS",
        "beta": 0.30,
        "role": "CONTROLLER",
        "nickname": "辅导员助手",
        "recent_messages": ["[exaggerate] 同学A: 事情非常严重，全校都完了"],
        "mentions": [],
        "topic_heat": {"T001": 0.85},
        "topic_negative": {"T001": 0.80},
    }

    prompt = build_prompt(belief, memory, env)
    raw = client.chat(prompt)
    parsed = parse_llm_response(raw)

    print("=== PROMPT (first 400 chars) ===")
    print(prompt[:400].replace("\n", " | "))
    print("=== RAW ===")
    print(raw[:500].replace("\n", " "))
    print("=== PARSED ===")
    print(parsed)

    # soft checks (LLM may vary; parse must not crash)
    assert isinstance(parsed, dict)
    assert "opinion_updates" in parsed
    assert "emotion_delta" in parsed
    assert parsed.get("action_type_name") in {
        "SEND_MESSAGE",
        "REPLY",
        "FORWARD",
        "SILENT",
    }
    print("DeepSeek live smoke OK.")


if __name__ == "__main__":
    main()
