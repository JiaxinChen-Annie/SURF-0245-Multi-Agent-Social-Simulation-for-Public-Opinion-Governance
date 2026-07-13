from __future__ import annotations

import os

from llm_utils import parse_llm_response, setup_llm_client


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is empty. Set it in your shell first.")

    client = setup_llm_client(
        {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
            "timeout": 30,
            "max_retries": 2,
            "temperature": 0.2,
        }
    )

    prompt = (
        "Return ONLY a JSON object with fields: "
        "action_type, content, event_id, opinion_value, target_id, opinion_updates, emotion_delta. "
        "Use action_type SILENT."
    )
    raw = client.chat(prompt)
    parsed = parse_llm_response(raw)
    print("Raw:", raw[:200].replace("\n", " "))
    print("Parsed:", parsed)


if __name__ == "__main__":
    main()
