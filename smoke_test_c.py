"""
Module C local smoke tests (no API key required).

Covers:
  - HawkesEngine sampling
  - campus-group build_prompt / parse_llm_response
"""

from __future__ import annotations

import random

from hawkes_engine import HawkesEngine
from llm_utils import build_prompt, parse_llm_response, setup_llm_client


def test_hawkes() -> None:
    rng = random.Random(42)
    hawkes = HawkesEngine(mu=0.1, alpha=0.6, beta=1.2, rng=rng)
    current_t = 0.0
    for _ in range(5):
        t_next = hawkes.sample_next_time(current_t)
        hawkes.add_event(t_next)
        current_t = t_next
    print("Hawkes history:", [round(x, 4) for x in hawkes.history])
    print("Intensity at now:", round(hawkes.intensity(current_t), 4))
    assert len(hawkes.history) == 5
    assert hawkes.intensity(current_t) >= hawkes.mu


def test_llm_toolchain() -> None:
    belief = {
        "identity": {
            "agent_type": "ACTIVE",
            "group_type": "CLASS",
            "nickname": "小张",
            "stance_prior": -0.2,
        },
        "emotion": {"valence": -0.3, "arousal": 0.7},
        "opinions": {"T001": {"topic_id": "T001", "opinion_value": -0.4, "confidence": 0.5}},
    }
    memory = [
        {
            "tick": 3,
            "info": {
                "topic_id": "T001",
                "message_type": "paraphrase",
                "content": "听说食堂出事了",
                "source_nickname": "阿强",
            },
        }
    ]
    env = {
        "group_type": "CLASS",
        "beta": 0.12,
        "role": "ACTIVE",
        "nickname": "小张",
        "recent_messages": ["[paraphrase] 阿强: 听说食堂出事了"],
        "mentions": [],
        "topic_heat": {"T001": 0.82},
        "topic_negative": {"T001": 0.71},
    }

    prompt = build_prompt(belief, memory, env)
    print("Prompt length:", len(prompt))
    assert "ACTIVE" in prompt or "活跃" in prompt
    assert "CLASS" in prompt or "班级" in prompt
    assert "topic_heat" in prompt
    assert "OUTPUT JSON SCHEMA" in prompt

    mock_client = setup_llm_client({"provider": "mock"})
    raw = mock_client.chat(prompt)
    parsed = parse_llm_response(raw)
    print("Parsed from mock:", parsed)
    assert parsed["action_type_name"] == "SILENT"
    assert "opinion_updates" in parsed
    assert "emotion_delta" in parsed

    # noisy + new action/message types
    noisy = """结果如下：
```json
{"action_type":"SEND_MESSAGE","message_type":"clarification","content":"请勿传播未经核实信息","topic_id":"T001","opinion_value":0.1,"opinion_updates":{"T001":0.0},"emotion_delta":{"valence":0.05,"arousal":-0.1}}
```"""
    parsed_noisy = parse_llm_response(noisy)
    print("Parsed from noisy:", parsed_noisy)
    assert parsed_noisy["action_type_name"] == "SEND_MESSAGE"
    assert parsed_noisy["message_type"] == "clarification"
    assert parsed_noisy["topic_id"] == "T001"
    assert abs(parsed_noisy["emotion_delta"]["arousal"] - (-0.1)) < 1e-6

    # legacy alias still works
    legacy = parse_llm_response('{"action_type":"POST","event_id":"T002","content":"x"}')
    assert legacy["action_type_name"] == "SEND_MESSAGE"
    assert legacy["topic_id"] == "T002"


if __name__ == "__main__":
    test_hawkes()
    test_llm_toolchain()
    print("C module smoke test passed.")
