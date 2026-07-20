# test_agent.py
# B 模块（SocialAgent）微信场景完整测试程序
# 适配最终场景设定文档 v3.0
#
# 测试覆盖：
#   - 4类角色（ORDINARY / ACTIVE / RATIONAL / CONTROLLER）
#   - 4类群组（DORM / CLASS / MAJOR / CAMPUS）
#   - 5类消息类型（original / forward / paraphrase / exaggerate / clarification）
#   - 热度指数衰减模型：H(t+1) = H(t)·e^(-α)·e^(-βₖ·𝟙[t≥t_k^int])
#   - Controller 干预机制（宿舍群不触发）
#   - 消息转发扭曲（distortion）

import json
import random
import math
from typing import Dict, List, Optional, Any

# 导入 B 模块
from social_agent import (
    SocialAgent,
    AgentType,
    GroupType,
    ActionType,
    MessageType,
    SocialInfo,
    Perception,
    ActionRecord,
    OpinionBelief,
    GROUP_BETA,
    # 测试时需要用到的常量
)


# ======================== 模拟 A 模块 ========================

class DummyLLMClient:
    """模拟 C 模块：假 LLM 客户端"""
    def chat(self, prompt: str) -> str:
        # 模拟 LLM 返回（包含观点更新和情绪变化）
        return json.dumps({
            "opinion_updates": {
                "tuition_fee": 0.75,
                "campus_food": -0.30,
            },
            "emotion_delta": {
                "valence": 0.20,
                "arousal": -0.10
            }
        })


class DummyLLMUtils:
    """模拟 C 模块：假 LLM 工具"""
    def build_prompt(self, belief, memory, env_info):
        print(f"   [联调] build_prompt 被调用")
        if isinstance(env_info, dict):
            print(f"      - 群类型: {env_info.get('group_type', 'unknown')}")
            print(f"      - βₖ: {env_info.get('beta', 0.0):.3f}")
            print(f"      - 角色: {env_info.get('role', 'unknown')}")
            print(f"      - 话题热度: {env_info.get('topic_heat', {})}")
        return "模拟微信场景 Prompt"

    def get_client(self):
        return DummyLLMClient()

    def parse_llm_response(self, raw: str):
        return json.loads(raw)


class DummyModel:
    """模拟 A 模块：提供群聊上下文、热度计算、接收行动"""

    # 自然衰减率 α（与 B 模块保持一致）
    ALPHA: float = 0.15
    # 干预阈值 θ
    THETA: float = 0.7

    def __init__(self):
        self.random = random.Random(42)
        self.schedule = type('Obj', (), {'time': 0})()
        self.llm_utils = DummyLLMUtils()

        # Agent → 群ID 映射
        self.agent_groups = {
            1: "group_class_01",
            2: "group_dorm_02",
            3: "group_major_03",
            4: "group_campus_04",
        }

        # 群 → 群类型 映射
        self.group_types = {
            "group_class_01": GroupType.CLASS,
            "group_dorm_02": GroupType.DORM,
            "group_major_03": GroupType.MAJOR,
            "group_campus_04": GroupType.CAMPUS,
        }

        # 群消息缓存
        self.group_messages: Dict[str, List[SocialInfo]] = {
            "group_class_01": [],
            "group_dorm_02": [],
            "group_major_03": [],
            "group_campus_04": [],
        }

        # 话题热度 H(t) 缓存（按群）
        self.topic_heat: Dict[str, Dict[str, float]] = {
            "group_class_01": {},
            "group_dorm_02": {},
            "group_major_03": {},
            "group_campus_04": {},
        }

        # 话题负面程度缓存（按群）
        self.topic_negative: Dict[str, Dict[str, float]] = {
            "group_class_01": {},
            "group_dorm_02": {},
            "group_major_03": {},
            "group_campus_04": {},
        }

        # 干预触发时间 t_k^int（按群）
        self.intervention_ticks: Dict[str, Optional[int]] = {
            "group_class_01": None,
            "group_dorm_02": None,
            "group_major_03": None,
            "group_campus_04": None,
        }

        # 行动历史
        self.action_history: List[ActionRecord] = []
        self.last_action: Optional[ActionRecord] = None

        # 模拟被@消息
        self.last_mentions: List[SocialInfo] = []

        # 仿真步数
        self.step_count = 0

    # ---------- 供 B 调用的接口 ----------

    def get_agent_group(self, agent_id: int) -> str:
        return self.agent_groups.get(agent_id, "group_class_01")

    def get_group_type(self, group_id: str) -> GroupType:
        return self.group_types.get(group_id, GroupType.CLASS)

    def get_group_messages(self, group_id: str, limit: int = 20) -> List[SocialInfo]:
        messages = self.group_messages.get(group_id, [])
        return messages[-limit:] if messages else []

    def get_topic_heat(self, group_id: str) -> Dict[str, float]:
        return self.topic_heat.get(group_id, {})

    def get_topic_negative(self, group_id: str) -> Dict[str, float]:
        return self.topic_negative.get(group_id, {})

    def get_intervention_tick(self, group_id: str) -> Optional[int]:
        return self.intervention_ticks.get(group_id)

    def submit_action(self, record: ActionRecord) -> None:
        """接收 B 提交的行动"""
        self.last_action = record
        self.action_history.append(record)

        # ---- 处理行动：更新群消息缓存 ----
        group_id = self.get_agent_group(record.agent_id)

        # 构造 SocialInfo
        info = SocialInfo(
            source_id=record.agent_id,
            source_nickname=f"用户{record.agent_id}",
            content=record.content,
            message_type=record.message_type,
            timestamp=record.tick,
            is_mention=False,
            topic_id=record.topic_id,
            distortion_level=record.distortion_level,
            original_content=record.content,
            negative_score=record.negative_score,
            heat=record.heat,
        )

        # 存入群消息缓存
        if group_id not in self.group_messages:
            self.group_messages[group_id] = []
        self.group_messages[group_id].append(info)

        # ---- 更新话题热度 H(t) ----
        topic_id = record.topic_id
        if topic_id:
            # 获取当前热度
            current_heat = self.topic_heat.get(group_id, {}).get(topic_id, 0.0)

            # 如果是新话题，初始化 H₀
            if current_heat == 0.0:
                initial_heat = record.heat if record.heat > 0 else 0.5  # 默认初始热度
                current_heat = initial_heat

            # ---- 应用热度衰减公式 ----
            # H(t+1) = H(t) · e^(-α) · e^(-βₖ · 𝟙[t ≥ t_k^int])
            beta = GROUP_BETA.get(self.group_types.get(group_id, GroupType.CLASS), 0.0)
            intervention_tick = self.intervention_ticks.get(group_id)

            # 自然衰减
            natural_decay = math.exp(-self.ALPHA)

            # 干预衰减（宿舍群不触发）
            group_type = self.group_types.get(group_id, GroupType.CLASS)
            if group_type == GroupType.DORM:
                intervention_decay = 1.0
            elif intervention_tick is not None and self.step_count >= intervention_tick:
                intervention_decay = math.exp(-beta)
            else:
                intervention_decay = 1.0

            new_heat = current_heat * natural_decay * intervention_decay
            new_heat = max(0.0, min(1.0, new_heat))

            # 更新热度缓存
            if group_id not in self.topic_heat:
                self.topic_heat[group_id] = {}
            self.topic_heat[group_id][topic_id] = new_heat

            # 更新负面程度（模拟：随热度下降而下降）
            current_negative = self.topic_negative.get(group_id, {}).get(topic_id, record.negative_score)
            new_negative = current_negative * (0.9 + 0.1 * new_heat)  # 随热度下降
            if group_id not in self.topic_negative:
                self.topic_negative[group_id] = {}
            self.topic_negative[group_id][topic_id] = max(0.0, min(1.0, new_negative))

        print(f"\n   ✅ [环境] 收到行动:")
        print(f"      - Agent: {record.agent_id}")
        print(f"      - 行动: {record.action_type.name}")
        print(f"      - 消息类型: {record.message_type}")
        print(f"      - 扭曲: {record.distortion_level:.3f}")
        print(f"      - 负面: {record.negative_score:.3f}")
        if record.topic_id:
            print(f"      - 当前热度 H(t): {self.topic_heat.get(group_id, {}).get(record.topic_id, 0.0):.3f}")

    # ---------- 辅助方法：手动设置初始消息 ----------

    def add_initial_message(self, group_id: str, topic_id: str, initial_heat: float = 0.8):
        """模拟事件发起，设置初始热度"""
        if group_id not in self.topic_heat:
            self.topic_heat[group_id] = {}
        self.topic_heat[group_id][topic_id] = initial_heat

        if group_id not in self.topic_negative:
            self.topic_negative[group_id] = {}
        self.topic_negative[group_id][topic_id] = initial_heat * 0.8

    def add_mention(self, agent_id: int, content: str = "@你 怎么看？"):
        """模拟 @ 消息"""
        info = SocialInfo(
            source_id=999,
            source_nickname="系统",
            content=content,
            message_type=MessageType.ORIGINAL,
            timestamp=self.step_count,
            is_mention=True,
            topic_id="tuition_fee",
            distortion_level=0.0,
            original_content=content,
            negative_score=0.3,
            heat=0.0,
        )
        self.last_mentions = [info]

    def advance_step(self):
        """推进时间步"""
        self.step_count += 1
        self.schedule.time = self.step_count
        self.last_mentions = []


# ======================== 创建测试 Agent ========================

def create_test_agent(
    agent_type: AgentType = AgentType.ORDINARY,
    group_type: GroupType = GroupType.CLASS,
    agent_id: int = 1,
    initial_opinions: Dict[str, float] = None,
) -> tuple[SocialAgent, DummyModel]:
    """创建测试用智能体"""
    if initial_opinions is None:
        initial_opinions = {"tuition_fee": 0.20, "campus_food": -0.10}

    model = DummyModel()
    model.agent_groups[agent_id] = f"group_{group_type.name.lower()}"

    agent = SocialAgent(
        unique_id=agent_id,
        model=model,
        agent_type=agent_type,
        group_type=group_type,
        init_config={
            "nickname": f"测试_{agent_type.name}",
            "role_desc": f"{agent_type.name} 角色测试",
            "stance_prior": 0.0,
            "initial_opinions": initial_opinions,
            "initial_confidence": 0.60,
            "max_memory": 20,
            "initial_heat": 0.5,
        }
    )
    agent.memory = []

    # 如果有初始话题，设置初始热度
    for topic_id in initial_opinions.keys():
        model.add_initial_message(f"group_{group_type.name.lower()}", topic_id, 0.8)

    return agent, model


# ======================== 测试场景 ========================

def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"🧪 {title}")
    print("=" * 70)


def test_1_agent_creation():
    """测试1：智能体初始化"""
    print_section("测试1：智能体初始化（4类角色 × 4类群组）")

    for agent_type in [AgentType.ORDINARY, AgentType.ACTIVE, AgentType.RATIONAL, AgentType.CONTROLLER]:
        for group_type in [GroupType.DORM, GroupType.CLASS, GroupType.MAJOR, GroupType.CAMPUS]:
            agent, _ = create_test_agent(agent_type, group_type)
            beta = GROUP_BETA.get(group_type, 0.0)
            print(f"  {agent_type.name:10} + {group_type.name:8} | βₖ={beta:.3f} | {agent.beliefs.identity.nickname}")
            assert agent.agent_type == agent_type
            assert agent.group_type == group_type

    print("  ✅ 全部初始化成功")


def test_2_perceive():
    """测试2：感知功能"""
    print_section("测试2：感知 _perceive()")

    agent, model = create_test_agent(AgentType.ORDINARY, GroupType.CLASS)
    model.add_initial_message("group_class_01", "tuition_fee", 0.8)
    model.add_mention(agent.unique_id)

    perception = agent._perceive()

    print(f"  群ID: {perception.group_id}")
    print(f"  群类型: {perception.group_type.name}")
    print(f"  βₖ: {perception.beta:.3f}")
    print(f"  话题热度: {perception.topic_heat}")
    print(f"  @消息数: {len(perception.mentions)}")

    assert perception.group_type == GroupType.CLASS
    assert "tuition_fee" in perception.topic_heat
    print("  ✅ 通过")


def test_3_memory():
    """测试3：记忆功能"""
    print_section("测试3：记忆 _retrieve_memory()")

    agent, _ = create_test_agent(AgentType.ORDINARY, GroupType.CLASS)

    # 手动添加记忆
    for i in range(5):
        info = SocialInfo(
            source_id=i,
            source_nickname=f"用户{i}",
            content=f"记忆内容{i}",
            message_type=MessageType.ORIGINAL,
            timestamp=i,
            is_mention=False,
            topic_id="test",
            distortion_level=0.0,
            original_content=f"内容{i}",
            negative_score=0.1,
            heat=0.0,
        )
        agent.memory.append(type('MemoryRecord', (), {'tick': i, 'info': info, 'relevance': 0.5})())

    perception = Perception(tick=10)
    memories = agent._retrieve_memory(perception)

    print(f"  记忆条数: {len(memories)}")
    print(f"  预期: 最多5条")
    assert len(memories) <= 5
    print("  ✅ 通过")


def test_4_update_beliefs():
    """测试4：信念更新（LLM调用）"""
    print_section("测试4：信念更新 _update_beliefs()")

    agent, model = create_test_agent(AgentType.ORDINARY, GroupType.CLASS)
    model.add_initial_message("group_class_01", "tuition_fee", 0.8)

    perception = agent._perceive()
    print(f"  更新前 - 学费观点: {agent.beliefs.opinions['tuition_fee'].opinion_value:.3f}")
    print(f"  更新前 - 情绪效价: {agent.beliefs.emotion.valence:.3f}")

    agent._update_beliefs(perception, [])

    print(f"  更新后 - 学费观点: {agent.beliefs.opinions['tuition_fee'].opinion_value:.3f}")
    print(f"  更新后 - 情绪效价: {agent.beliefs.emotion.valence:.3f}")

    assert abs(agent.beliefs.opinions['tuition_fee'].opinion_value - 0.75) < 0.01
    print("  ✅ 通过")


def test_5_desires():
    """测试5：欲望推断（含Controller干预）"""
    print_section("测试5：欲望推断 _infer_desires()")

    # 5a: 普通群员 + @消息
    print("\n  5a: 普通群员 + @消息 → 回复欲望")
    agent, model = create_test_agent(AgentType.ORDINARY, GroupType.CLASS)
    model.add_mention(agent.unique_id)
    perception = agent._perceive()
    agent._last_perception = perception
    desires = agent._infer_desires()
    print(f"    欲望列表: {[d.goal_type for d in desires]}")
    assert any(d.goal_type == "reply" for d in desires)

    # 5b: Controller + 热度超阈值 → 干预欲望
    print("\n  5b: Controller + 热度超阈值 → 干预欲望")
    agent, model = create_test_agent(AgentType.CONTROLLER, GroupType.CLASS)
    model.add_initial_message("group_class_01", "tuition_fee", 0.85)
    perception = agent._perceive()
    agent._last_perception = perception
    desires = agent._infer_desires()
    print(f"    欲望列表: {[d.goal_type for d in desires]}")
    assert any(d.goal_type == "intervene" for d in desires)

    # 5c: Controller + 宿舍群 → 不触发干预
    print("\n  5c: Controller + 宿舍群（不触发干预）")
    agent, model = create_test_agent(AgentType.CONTROLLER, GroupType.DORM)
    model.add_initial_message("group_dorm_02", "tuition_fee", 0.85)
    perception = agent._perceive()
    agent._last_perception = perception
    desires = agent._infer_desires()
    print(f"    欲望列表: {[d.goal_type for d in desires]}")
    # 宿舍群即使有Controller也不触发干预
    assert not any(d.goal_type == "intervene" for d in desires)
    print("  ✅ 全部通过")


def test_6_plan_intentions():
    """测试6：意图规划"""
    print_section("测试6：意图规划 _plan_intentions()")

    agent, _ = create_test_agent(AgentType.ACTIVE, GroupType.CLASS)

    # 测试各种欲望 → 意图映射
    test_cases = [
        ("reply", ActionType.REPLY),
        ("discuss", ActionType.SEND_MESSAGE),
        ("share", ActionType.FORWARD),
        ("clarify", ActionType.SEND_MESSAGE),
        ("intervene", ActionType.SILENT),
        ("silent", ActionType.SILENT),
    ]

    for goal_type, expected_action in test_cases:
        desires = [type('Desire', (), {'goal_type': goal_type, 'topic_id': 'test', 'priority': 1.0, 'target_id': None})()]
        intention = agent._plan_intentions(desires)
        print(f"  {goal_type:10} → {intention.action_type.name}")
        assert intention.action_type == expected_action

    print("  ✅ 全部通过")


def test_7_execute_action():
    """测试7：行动执行（含distortion和消息类型）"""
    print_section("测试7：行动执行 _execute_action()")

    # 7a: 不同角色转发 → 不同的distortion
    print("\n  7a: 不同角色转发 → distortion 差异")
    for agent_type in [AgentType.ORDINARY, AgentType.ACTIVE, AgentType.RATIONAL, AgentType.CONTROLLER]:
        agent, model = create_test_agent(agent_type, GroupType.CLASS)
        # 设置当前热度（用于计算负面）
        model.topic_heat["group_class_01"] = {"tuition_fee": 0.8}
        agent._last_perception = Perception(topic_heat={"tuition_fee": 0.8})

        intention = type('Intention', (), {
            'action_type': ActionType.FORWARD,
            'content_plan': '转发测试内容',
            'topic_id': 'tuition_fee',
            'target_id': None
        })()
        record = agent._execute_action(intention)

        print(f"    {agent_type.name:10} | distortion={record.distortion_level:.3f} | type={record.message_type}")
        # ORDINARY 的 distortion 应该较高
        if agent_type == AgentType.ORDINARY:
            assert record.distortion_level >= 0.1

    # 7b: Controller 干预记录 t_k^int
    print("\n  7b: Controller 干预 → 记录 t_k^int")
    agent, model = create_test_agent(AgentType.CONTROLLER, GroupType.CLASS)
    agent._last_perception = Perception(topic_heat={"tuition_fee": 0.8})
    intention = type('Intention', (), {
        'action_type': ActionType.SILENT,
        'content_plan': '触发干预',
        'topic_id': 'tuition_fee',
        'target_id': None
    })()
    agent._execute_action(intention)
    print(f"    intervention_tick: {agent.intervention_tick}")
    assert agent.intervention_tick is not None

    print("  ✅ 全部通过")


def test_8_update_emotion():
    """测试8：情绪更新"""
    print_section("测试8：情绪更新 _update_emotion()")

    agent, _ = create_test_agent(AgentType.ORDINARY, GroupType.CLASS)
    agent.beliefs.emotion.valence = 0.5

    # 无反馈 → 向基线衰减
    agent._update_emotion({})
    print(f"  无反馈衰减后: valence={agent.beliefs.emotion.valence:.3f}")
    assert agent.beliefs.emotion.valence < 0.5

    # 有反馈 → 应用增量
    agent.beliefs.emotion.valence = 0.0
    agent._update_emotion({"valence_delta": 0.3})
    print(f"  有反馈更新后: valence={agent.beliefs.emotion.valence:.3f}")
    assert abs(agent.beliefs.emotion.valence - 0.3) < 0.01

    print("  ✅ 通过")


def test_9_heat_decay():
    """测试9：热度衰减公式 calc_heat_decay()"""
    print_section("测试9：热度衰减 H(t+1) = H(t)·e^(-α)·e^(-βₖ·𝟙[t≥t_k^int])")

    # 9a: 无干预（自然衰减）
    print("\n  9a: 无干预（纯自然衰减）")
    agent, _ = create_test_agent(AgentType.ORDINARY, GroupType.CLASS)
    current_heat = 1.0
    for step in range(5):
        new_heat = agent.calc_heat_decay(current_heat, step, None)
        print(f"    t={step}: H={new_heat:.4f}")
        current_heat = new_heat
    assert current_heat < 1.0

    # 9b: 有干预（叠加衰减）
    print("\n  9b: 有干预（t_k^int=2）")
    agent, _ = create_test_agent(AgentType.CONTROLLER, GroupType.CLASS)
    current_heat = 1.0
    for step in range(5):
        intervention_tick = 2
        new_heat = agent.calc_heat_decay(current_heat, step, intervention_tick)
        print(f"    t={step}: H={new_heat:.4f} {'← 已干预' if step >= intervention_tick else ''}")
        current_heat = new_heat
    # 应该有明显下降
    assert current_heat < 0.5

    # 9c: 宿舍群不触发干预
    print("\n  9c: 宿舍群（不触发干预）")
    agent, _ = create_test_agent(AgentType.ORDINARY, GroupType.DORM)
    current_heat = 1.0
    for step in range(5):
        new_heat = agent.calc_heat_decay(current_heat, step, 2)
        print(f"    t={step}: H={new_heat:.4f}（宿舍群无干预效果）")
        current_heat = new_heat
    # 宿舍群即使传入干预时间，也不生效
    assert current_heat > 0.4

    print("  ✅ 全部通过")


def test_10_full_step():
    """测试10：完整 step 流程（集成）"""
    print_section("测试10：完整 step() 流程")

    agent, model = create_test_agent(AgentType.ORDINARY, GroupType.CLASS)

    # 设置环境：有 @消息 + 初始热度
    model.add_initial_message("group_class_01", "tuition_fee", 0.8)
    model.add_mention(agent.unique_id, "@你 学费涨这么多合理吗？")
    model.advance_step()

    print(f"  初始 - 学费观点: {agent.beliefs.opinions['tuition_fee'].opinion_value:.3f}")
    print(f"  初始 - 情绪效价: {agent.beliefs.emotion.valence:.3f}")

    # 执行 step
    agent.step()

    action = model.last_action
    print(f"\n  输出行动: {action.action_type.name if action else 'SILENT'}")
    if action:
        print(f"    - 消息类型: {action.message_type}")
        print(f"    - 扭曲程度: {action.distortion_level:.3f}")
        print(f"    - 负面程度: {action.negative_score:.3f}")

    print(f"\n  更新后 - 学费观点: {agent.beliefs.opinions['tuition_fee'].opinion_value:.3f}")
    print(f"  更新后 - 情绪效价: {agent.beliefs.emotion.valence:.3f}")
    print(f"  记忆条数: {len(agent.memory)}")

    # 验证是否有行动产生
    if action:
        print("  ✅ 通过（有行动产生）")
    else:
        print("  ⚠️ 通过（无行动，可能是SILENT）")


def test_11_beta_constraint():
    """测试11：βₖ 约束检查（β₁ < β₂ < β₃ < β₄）"""
    print_section("测试11：βₖ 分级约束检查")

    beta_values = [
        ("DORM", GROUP_BETA[GroupType.DORM]),
        ("CLASS", GROUP_BETA[GroupType.CLASS]),
        ("MAJOR", GROUP_BETA[GroupType.MAJOR]),
        ("CAMPUS", GROUP_BETA[GroupType.CAMPUS]),
    ]

    print("  βₖ 值:")
    for name, val in beta_values:
        print(f"    {name}: {val:.3f}")

    # 验证严格递增
    betas = [v for _, v in beta_values]
    assert betas[0] < betas[1] < betas[2] < betas[3]
    print("  ✅ 约束满足: β₁ < β₂ < β₃ < β₄")


def test_12_scenario_demo():
    """测试12：完整场景演示（多步仿真）"""
    print_section("测试12：完整场景演示（10步仿真）")

    agent, model = create_test_agent(AgentType.RATIONAL, GroupType.CLASS)
    agent_id = agent.unique_id

    # 初始化环境
    model.add_initial_message("group_class_01", "tuition_fee", 0.9)
    print(f"\n  初始热度: {model.topic_heat['group_class_01']['tuition_fee']:.3f}")

    # 模拟 10 步
    for step in range(10):
        model.advance_step()
        if step == 3:
            model.add_mention(agent_id, f"@你 第{step}步触发讨论")
        agent.step()

        heat = model.topic_heat.get("group_class_01", {}).get("tuition_fee", 0.0)
        print(f"    t={step+1:2d} | 热度={heat:.3f} | 行动={model.last_action.action_type.name if model.last_action and model.last_action.action_type != ActionType.SILENT else 'SILENT'}")

        # 重置 last_action 避免重复打印
        model.last_action = None

    print("\n  ✅ 场景仿真完成")


# ======================== 主菜单 ========================

def main():
    print("\n" + "🧪" * 35)
    print("   B 模块 SocialAgent 微信场景完整测试")
    print("   适配场景设定 v3.0 | 覆盖全部 10 个函数")
    print("🧪" * 35)

    tests = [
        ("1", "智能体初始化", test_1_agent_creation),
        ("2", "感知 _perceive()", test_2_perceive),
        ("3", "记忆 _retrieve_memory()", test_3_memory),
        ("4", "信念更新 _update_beliefs()", test_4_update_beliefs),
        ("5", "欲望推断 _infer_desires()", test_5_desires),
        ("6", "意图规划 _plan_intentions()", test_6_plan_intentions),
        ("7", "行动执行 _execute_action()", test_7_execute_action),
        ("8", "情绪更新 _update_emotion()", test_8_update_emotion),
        ("9", "热度衰减 calc_heat_decay()", test_9_heat_decay),
        ("10", "完整流程 step()", test_10_full_step),
        ("11", "βₖ 约束检查", test_11_beta_constraint),
        ("12", "场景演示（10步）", test_12_scenario_demo),
    ]

    print("\n可选测试:")
    for num, name, _ in tests:
        print(f"  {num:2}: {name}")
    print("  all: 全部运行")
    print("  q: 退出")

    choice = input("\n请输入编号: ").strip()

    if choice == "q":
        print("退出")
        return

    if choice == "all":
        for num, name, func in tests:
            try:
                func()
            except Exception as e:
                print(f"\n  ❌ 测试 {num} 失败: {e}")
        print("\n" + "=" * 70)
        print("✅ 全部测试完成！")
        return

    # 运行单个测试
    for num, name, func in tests:
        if choice == num:
            try:
                func()
            except Exception as e:
                print(f"\n  ❌ 测试失败: {e}")
                import traceback
                traceback.print_exc()
            return

    print(f"❌ 无效编号: {choice}")
    print("请输入 1-12, all, 或 q")


if __name__ == "__main__":
    main()