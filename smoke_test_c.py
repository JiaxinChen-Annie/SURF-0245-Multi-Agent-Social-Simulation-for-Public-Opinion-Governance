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


def test_llm_toolchain() -> None:
    belief = {"identity": {"agent_type": "PUBLIC"}, "emotion": {"valence": -0.2, "arousal": 0.7}}
    memory = [{"tick": 3, "info": {"event_id": "E001", "content": "rumor spreading"}}]
    env = {"tick": 4, "trending_topics": [["E001", 0.9]]}

    prompt = build_prompt(belief, memory, env)
    print("Prompt length:", len(prompt))

    mock_client = setup_llm_client({"provider": "mock"})
    raw = mock_client.chat(prompt)
    parsed = parse_llm_response(raw)
    print("Parsed from mock:", parsed)

    # noisy response parse check
    noisy = """Here is result:\n```json\n{"action_type":"POST","content":"test","event_id":"E001","opinion_value":0.8}\n```"""
    parsed_noisy = parse_llm_response(noisy)
    print("Parsed from noisy:", parsed_noisy)


if __name__ == "__main__":
    test_hawkes()
    test_llm_toolchain()
    print("C module smoke test passed.")
