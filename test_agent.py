# test_social_agent_unit.py
# 支持单独测试每个函数：python test_social_agent_unit.py 1  (测第1个)
# 或全部测试：python test_social_agent_unit.py all

import sys
import json
import random

# ===================== 模拟依赖（假环境 + 假 LLM） =====================

class DummyLLMClient:
    def chat(self, prompt):
        return '{"opinion_updates": {"E001": 0.75}, "emotion_delta": {"valence": 0.2}}'

class DummyLLMUtils:
    def build_prompt(self, belief, memory, env_info):
        return "Mock Prompt"
    def get_client(self):
        return DummyLLMClient()
    def parse_llm_response(self, raw):
        return json.loads(raw)

class DummyModel:
    def __init__(self):
        self.random = random.Random(42)
        self.schedule = type('Obj', (), {'time': 5})()
        self.llm_utils = DummyLLMUtils()
        self.trending = [("热点A", 0.8), ("热点B", 0.6)]
    def submit_action(self, record):
        print(f"    [环境] 收到提交行动: {record.action_type.name}")
    def get_neighbor_actions(self, agent_id):
        from social_agent import SocialInfo, ActionType
        return [
            SocialInfo(source_id=2, action_type=ActionType.POST, 
                       content="邻居发言", event_id="E001", opinion_value=0.5, timestamp=4)
        ]
    def get_mentions(self, agent_id):
        return []

from social_agent import (
    SocialAgent, AgentType, ActionType, 
    BeliefSystem, OpinionBelief, Personality, EmotionState,
    Perception, MemoryRecord, Desire, Intention, ActionRecord
)

# ===================== 创建测试用的 Agent 实例 =====================

def create_test_agent():
    model = DummyModel()
    agent = SocialAgent(
        unique_id=1,
        model=model,
        agent_type=AgentType.PUBLIC,
        init_config={"role_desc": "普通网民", "stance_prior": 0.1}
    )
    agent.beliefs.opinions["E001"] = OpinionBelief(event_id="E001", opinion_value=0.0)
    agent.memory = []
    return agent

# ===================== 10 个测试函数（每个对应一个类方法） =====================

def test_1_init_psychology():
    """测试 _init_psychology"""
    print("\n[测试1] _init_psychology")
    agent = create_test_agent()
    agent._init_psychology()
    p = agent.beliefs.psychology
    print(f"  人格: openness={p.personality.openness:.2f}, risk_aversion={p.risk_aversion:.2f}")
    assert 0 <= p.personality.openness <= 1
    assert 0 <= p.risk_aversion <= 1
    print("  ✅ 通过")

def test_2_perceive():
    """测试 _perceive"""
    print("\n[测试2] _perceive")
    agent = create_test_agent()
    perception = agent._perceive()
    print(f"  邻居动态数: {len(perception.neighbor_actions)}")
    print(f"  话题榜: {perception.trending_topics}")
    print(f"  当前tick: {perception.tick}")
    assert isinstance(perception, Perception)
    assert perception.tick == 5
    print("  ✅ 通过")

def test_3_retrieve_memory():
    """测试 _retrieve_memory"""
    print("\n[测试3] _retrieve_memory")
    agent = create_test_agent()
    from social_agent import SocialInfo
    info1 = SocialInfo(source_id=2, action_type=ActionType.POST, 
                       content="旧内容", event_id="E001", opinion_value=0.3, timestamp=2)
    info2 = SocialInfo(source_id=3, action_type=ActionType.COMMENT, 
                       content="新内容", event_id="E001", opinion_value=0.4, timestamp=4)
    agent.memory = [
        MemoryRecord(tick=2, info=info1, relevance=0.5),
        MemoryRecord(tick=4, info=info2, relevance=0.8),
    ]
    perception = Perception(tick=5, neighbor_actions=[], trending_topics=[])
    retrieved = agent._retrieve_memory(perception)
    print(f"  检索到 {len(retrieved)} 条记忆")
    assert len(retrieved) == 2
    print("  ✅ 通过")

def test_4_update_beliefs():
    """测试 _update_beliefs (LLM驱动)"""
    print("\n[测试4] _update_beliefs")
    agent = create_test_agent()
    agent.beliefs.opinions["E001"].opinion_value = 0.0
    agent.beliefs.emotion.valence = 0.0
    print(f"  更新前: 观点=0.0")
    perception = Perception(tick=5, neighbor_actions=[], trending_topics=[("事件", 0.9)])
    agent._update_beliefs(perception, [])
    new_opinion = agent.beliefs.opinions["E001"].opinion_value
    print(f"  更新后: 观点={new_opinion}")
    assert new_opinion == 0.75, f"观点更新失败，得到{new_opinion}"
    print("  ✅ 通过")

def test_5_infer_desires():
    """测试 _infer_desires"""
    print("\n[测试5] _infer_desires")
    agent = create_test_agent()
    agent.beliefs.emotion.valence = -0.5
    agent.beliefs.opinions["E001"].confidence = 0.3
    desires = agent._infer_desires()
    print(f"  生成欲望数: {len(desires)}")
    for d in desires:
        print(f"    - {d.goal_type} (优先级: {d.priority})")
    assert len(desires) > 0
    assert any(d.goal_type == "vent_emotion" for d in desires)
    print("  ✅ 通过")

def test_6_plan_intentions():
    """测试 _plan_intentions"""
    print("\n[测试6] _plan_intentions")
    agent = create_test_agent()
    desires = [
        Desire(goal_type="vent_emotion", priority=0.8, event_id="E001"),
        Desire(goal_type="get_info", priority=0.5, event_id="E001"),
    ]
    intention = agent._plan_intentions(desires)
    print(f"  计划行动: {intention.action_type.name}, 事件: {intention.event_id}")
    assert intention.action_type == ActionType.POST
    print("  ✅ 通过")

def test_7_execute_action():
    """测试 _execute_action"""
    print("\n[测试7] _execute_action")
    agent = create_test_agent()
    intention = Intention(
        action_type=ActionType.COMMENT,
        event_id="E001",
        content_plan="我同意这个观点",
        target_id=2
    )
    record = agent._execute_action(intention)
    print(f"  生成记录: agent={record.agent_id}, action={record.action_type.name}")
    print(f"  记忆长度: {len(agent.memory)}")
    assert record.action_type == ActionType.COMMENT
    assert record.content == "我同意这个观点"
    assert len(agent.memory) == 1
    print("  ✅ 通过")

def test_8_update_emotion():
    """测试 _update_emotion"""
    print("\n[测试8] _update_emotion")
    agent = create_test_agent()
    agent.beliefs.emotion.valence = 0.5
    agent.beliefs.emotion.arousal = 0.8
    
    agent._update_emotion({})
    print(f"  无反馈衰减后: valence={agent.beliefs.emotion.valence:.3f}")
    assert abs(agent.beliefs.emotion.valence - 0.475) < 0.001
    
    agent.beliefs.emotion.valence = 0.0
    agent._update_emotion({"valence_delta": 0.3, "arousal_delta": -0.1})
    print(f"  有反馈更新后: valence={agent.beliefs.emotion.valence:.3f}")
    assert abs(agent.beliefs.emotion.valence - 0.3) < 0.001
    print("  ✅ 通过")

def test_9_full_step():
    """测试 step 完整流程（集成）"""
    print("\n[测试9] step 完整流程")
    agent = create_test_agent()
    agent.beliefs.opinions["E001"] = OpinionBelief(event_id="E001", opinion_value=0.0)
    agent.step()
    print(f"  step 执行完成，最终观点: {agent.beliefs.opinions['E001'].opinion_value}")
    print("  ✅ 通过 (无崩溃)")

def test_10_edge_cases():
    """测试边缘情况：LLM失败、畸形JSON"""
    print("\n[测试10] 边缘情况测试")
    class BadLLMClient:
        def chat(self, prompt):
            return "这不是JSON"
    class BadLLMUtils:
        def build_prompt(self, belief, memory, env_info): return "x"
        def get_client(self): return BadLLMClient()
        def parse_llm_response(self, raw): 
            raise json.JSONDecodeError("err", raw, 0)
    
    model = DummyModel()
    model.llm_utils = BadLLMUtils()
    agent = SocialAgent(unique_id=2, model=model, agent_type=AgentType.PUBLIC, 
                        init_config={"role_desc": "测试"})
    agent.beliefs.opinions["E001"] = OpinionBelief(event_id="E001", opinion_value=0.5)
    old_opinion = agent.beliefs.opinions["E001"].opinion_value
    
    agent._update_beliefs(Perception(tick=0), [])
    new_opinion = agent.beliefs.opinions["E001"].opinion_value
    print(f"  LLM失败后，观点保持为: {new_opinion} (原值: {old_opinion})")
    assert new_opinion == old_opinion
    print("  ✅ 通过")

# ===================== 测试运行器（支持单独运行） =====================

# 映射表：编号 -> (测试函数, 描述)
TEST_MAP = {
    "1": (test_1_init_psychology, "_init_psychology"),
    "2": (test_2_perceive, "_perceive"),
    "3": (test_3_retrieve_memory, "_retrieve_memory"),
    "4": (test_4_update_beliefs, "_update_beliefs"),
    "5": (test_5_infer_desires, "_infer_desires"),
    "6": (test_6_plan_intentions, "_plan_intentions"),
    "7": (test_7_execute_action, "_execute_action"),
    "8": (test_8_update_emotion, "_update_emotion"),
    "9": (test_9_full_step, "step (集成)"),
    "10": (test_10_edge_cases, "边缘情况"),
}

def run_single(test_key):
    if test_key not in TEST_MAP:
        print(f"❌ 无效编号: {test_key}")
        print("   可选编号: 1~10, all")
        return False
    func, name = TEST_MAP[test_key]
    print("=" * 50)
    print(f"单独运行: 测试 {test_key} - {name}")
    print("=" * 50)
    try:
        func()
        return True
    except AssertionError as e:
        print(f"  ❌ 断言失败: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False

def run_all():
    print("=" * 50)
    print("运行全部 10 个测试")
    print("=" * 50)
    passed = 0
    for key, (func, name) in TEST_MAP.items():
        print(f"\n--- [{key}] {name} ---")
        try:
            func()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ 断言失败: {e}")
        except Exception as e:
            print(f"  ❌ 异常: {e}")
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{len(TEST_MAP)} 个通过")
    if passed == len(TEST_MAP):
        print("🎉 全部通过！")
    else:
        print("⚠️ 部分未通过")

if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
        if choice == "all":
            run_all()
        else:
            run_single(choice)
    else:
        # 无参数时，显示菜单
        print("=" * 50)
        print("B 模块单元测试 - 交互式菜单")
        print("=" * 50)
        print("可选测试编号:")
        for key, (_, name) in TEST_MAP.items():
            print(f"  {key}: {name}")
        print("  all: 运行全部")
        print("  q: 退出")
        choice = input("\n请输入编号: ").strip()
        if choice == "q":
            print("退出")
        elif choice == "all":
            run_all()
        else:
            run_single(choice)