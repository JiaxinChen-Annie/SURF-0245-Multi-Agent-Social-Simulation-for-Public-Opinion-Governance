"""
test_contract.py — 接口契约验证脚本（v2，W4）
----------------------------------------------
更新内容：
  - AgentType: ORDINARY/ACTIVE/RATIONAL/CONTROLLER
  - ActionType: SEND_MESSAGE/REPLY/FORWARD/SILENT（移除 LIKE）
  - GroupType: DORM/CLASS/MAJOR/CAMPUS
  - MessageType 字符串常量
  - ActionRecord 新增内容字段与消息/群路由字段检查
  - submit_action 新增 topic_heat/topic_negative/cross_group_forward 检查
  - SocialAgent.__init__ 新增 group_type 参数验证
  - calc_heat_decay 接口验证（#11）
  - Perception 新增字段检查
  - 多群成员、真实源消息转发、目标群投递、LLM 行动执行回归测试
  - 微观 cross_group_forward / 宏观 cross_group_spread 计数分离
  - 各群 β 与干预触发 tick 的热度公式回归测试

用法：python3 test_contract.py
"""

from __future__ import annotations
import sys
import traceback

_results: list = []

def _check(label: str, fn) -> bool:
    try:
        fn()
        _results.append(("✅", label, ""))
        print(f"  ✅  {label}")
        return True
    except Exception as exc:
        msg = traceback.format_exc().strip().splitlines()[-1]
        _results.append(("❌", label, msg))
        print(f"  ❌  {label}")
        print(f"       └─ {msg}")
        return False


# ═══════════════════════════════════════════════════════════════════ #
# [types]  共享类型层（v2）
# ═══════════════════════════════════════════════════════════════════ #

def _test_types_import():
    from types_def import (
        SimConfig, AgentType, GroupType, ActionType, MessageType,
        ActionRecord, BeliefSystem, Perception, MemoryRecord,
        EmotionState, Desire, Intention, SocialInfo, OpinionBelief,
        InterventionType, ALPHA, THETA, GROUP_BETA,
    )
    cfg = SimConfig()
    assert cfg.n_agents > 0, "SimConfig 默认值异常"

    # v2 AgentType 枚举值
    assert AgentType.ORDINARY   == 0, "ORDINARY 应为 0"
    assert AgentType.ACTIVE     == 1, "ACTIVE 应为 1"
    assert AgentType.RATIONAL   == 2, "RATIONAL 应为 2"
    assert AgentType.CONTROLLER == 3, "CONTROLLER 应为 3"

    # 旧枚举值不应存在
    assert not hasattr(AgentType, "PUBLIC"),         "PUBLIC 已移除"
    assert not hasattr(AgentType, "OPINION_LEADER"), "OPINION_LEADER 已移除"

    # v2 ActionType 枚举值
    assert ActionType.SEND_MESSAGE == 0
    assert ActionType.REPLY        == 1
    assert ActionType.FORWARD      == 2
    assert ActionType.SILENT       == 3
    assert not hasattr(ActionType, "LIKE"), "LIKE 已移除"
    assert not hasattr(ActionType, "POST"), "POST 已移除"

    # GroupType
    assert GroupType.DORM   == 0
    assert GroupType.CLASS  == 1
    assert GroupType.MAJOR  == 2
    assert GroupType.CAMPUS == 3

    # MessageType 字符串常量
    assert MessageType.ORIGINAL      == "original"
    assert MessageType.CLARIFICATION == "clarification"

    # 常量
    assert abs(ALPHA - 0.15) < 1e-9, f"ALPHA 应为 0.15，实际 {ALPHA}"
    assert abs(THETA - 0.7)  < 1e-9, f"THETA 应为 0.7，实际 {THETA}"
    assert GROUP_BETA[GroupType.DORM]   < GROUP_BETA[GroupType.CLASS]
    assert GROUP_BETA[GroupType.CLASS]  < GROUP_BETA[GroupType.MAJOR]
    assert GROUP_BETA[GroupType.MAJOR]  < GROUP_BETA[GroupType.CAMPUS]

    # topic_id 替代 event_id
    rec = ActionRecord(topic_id="T001")
    assert hasattr(rec, "topic_id"),         "ActionRecord 缺少 topic_id"
    assert not hasattr(rec, "event_id"),     "ActionRecord 不应有 event_id"
    assert hasattr(rec, "distortion_level"), "ActionRecord 缺少 distortion_level"
    assert hasattr(rec, "message_type"),     "ActionRecord 缺少 message_type"
    assert hasattr(rec, "negative_score"),   "ActionRecord 缺少 negative_score"
    assert hasattr(rec, "heat"),             "ActionRecord 缺少 heat"
    assert hasattr(rec, "message_id")
    assert hasattr(rec, "group_id")
    assert hasattr(rec, "source_message_id")
    assert hasattr(rec, "source_group_id")

    # Perception v2 字段
    p = Perception()
    assert hasattr(p, "group_id"),        "Perception 缺少 group_id"
    assert hasattr(p, "group_type"),      "Perception 缺少 group_type"
    assert hasattr(p, "beta"),            "Perception 缺少 beta"
    assert hasattr(p, "recent_messages"), "Perception 缺少 recent_messages（原 neighbor_actions）"
    assert not hasattr(p, "neighbor_actions"), "neighbor_actions 已重命名"
    assert hasattr(p, "topic_heat"),      "Perception 缺少 topic_heat"
    assert hasattr(p, "topic_negative"),  "Perception 缺少 topic_negative"
    assert hasattr(p, "group_ids"),       "Perception 缺少多群成员列表 group_ids"
    assert hasattr(p, "topic_heat_by_group")

    # SocialInfo v2 字段
    si = SocialInfo()
    assert hasattr(si, "source_nickname"),  "SocialInfo 缺少 source_nickname"
    assert hasattr(si, "message_type"),     "SocialInfo 缺少 message_type"
    assert hasattr(si, "is_mention"),       "SocialInfo 缺少 is_mention"
    assert hasattr(si, "topic_id"),         "SocialInfo 缺少 topic_id"
    assert hasattr(si, "distortion_level"), "SocialInfo 缺少 distortion_level"
    assert hasattr(si, "negative_score"),   "SocialInfo 缺少 negative_score"
    assert hasattr(si, "heat"),             "SocialInfo 缺少 heat"
    assert hasattr(si, "message_id")
    assert hasattr(si, "group_id")
    assert hasattr(si, "source_message_id")
    assert hasattr(si, "source_group_id")

    # Desire / Intention v2 字段
    d = Desire()
    assert hasattr(d, "topic_id"),  "Desire 缺少 topic_id"
    assert hasattr(d, "target_id"), "Desire 缺少 target_id"
    assert hasattr(d, "destination_group_id")
    assert hasattr(d, "source_message_id")
    assert not hasattr(d, "event_id"), "Desire 不应有 event_id"
    it = Intention()
    assert hasattr(it, "topic_id"), "Intention 缺少 topic_id"
    assert hasattr(it, "destination_group_id")
    assert hasattr(it, "source_message_id")
    assert hasattr(it, "message_type")
    assert not hasattr(it, "event_id"), "Intention 不应有 event_id"


# ═══════════════════════════════════════════════════════════════════ #
# [C]  HawkesEngine 接口（不变）
# ═══════════════════════════════════════════════════════════════════ #

def _test_hawkes_init():
    from hawkes_engine import HawkesEngine
    h = HawkesEngine(mu=0.1, alpha=0.5, beta=1.0)
    assert h.history == []
    try:
        HawkesEngine(mu=-1, alpha=0, beta=1)
        raise AssertionError("mu=-1 应触发 ValueError")
    except ValueError:
        pass
    try:
        HawkesEngine(mu=0.1, alpha=0, beta=-1)
        raise AssertionError("beta=-1 应触发 ValueError")
    except ValueError:
        pass


def _test_hawkes_intensity():
    from hawkes_engine import HawkesEngine
    h = HawkesEngine(mu=0.2, alpha=0.8, beta=1.0)
    lam0 = h.intensity(0.0)
    assert isinstance(lam0, float)
    assert abs(lam0 - 0.2) < 1e-9
    h.add_event(1.0)
    assert h.intensity(1.5) > 0.2


def _test_hawkes_sample():
    from hawkes_engine import HawkesEngine
    h = HawkesEngine(mu=0.5, alpha=0.3, beta=1.0)
    t_next = h.sample_next_time(0.0)
    assert isinstance(t_next, float) and t_next > 0.0


def _test_hawkes_add_event():
    from hawkes_engine import HawkesEngine
    h = HawkesEngine(mu=0.1, alpha=0.5, beta=1.0)
    h.add_event(3.0); h.add_event(1.0); h.add_event(2.0)
    assert h.history == sorted(h.history), "history 应保持升序"


# ═══════════════════════════════════════════════════════════════════ #
# [B]  SocialAgent 接口（v2）
# ═══════════════════════════════════════════════════════════════════ #

def _make_single_model():
    from types_def import SimConfig
    from opinion_model import OpinionModel
    return OpinionModel(SimConfig(
        n_agents=1,
        hawkes_params={"mu": 0.5, "alpha": 0.3, "beta": 1.0},
    ))


def _test_agent_init_group_type():
    """SocialAgent.__init__ 须接受 group_type 参数（v2 新增）。"""
    from types_def import SimConfig, AgentType, GroupType
    from social_agent import SocialAgent
    model = _make_single_model()
    agent = SocialAgent(
        unique_id=999,
        model=model,
        agent_type=AgentType.CONTROLLER,
        group_type=GroupType.CAMPUS,
        init_config={"stance_prior": 0.5, "topic_id": "T001", "initial_heat": 0.5},
    )
    assert agent.beliefs.identity.group_type == GroupType.CAMPUS, "group_type 未正确写入"
    assert agent.beliefs.identity.agent_type == AgentType.CONTROLLER


def _test_agent_attributes_v2():
    """SocialAgent 暴露 beliefs / pending_action；beliefs.identity 含 group_type/nickname。"""
    model = _make_single_model()
    agent = model.schedule.agents[0]
    assert hasattr(agent, "beliefs")
    assert hasattr(agent.beliefs, "opinions")
    assert hasattr(agent.beliefs, "emotion")
    assert hasattr(agent.beliefs, "psychology")
    assert hasattr(agent, "pending_action")
    assert hasattr(agent.beliefs.identity, "group_type"), "IdentityBelief 缺少 group_type"
    assert hasattr(agent.beliefs.identity, "group_ids"), "IdentityBelief 缺少 group_ids"
    assert hasattr(agent.beliefs.identity, "primary_group_id")
    assert hasattr(agent.beliefs.identity, "nickname"),   "IdentityBelief 缺少 nickname"
    assert hasattr(agent, "beta"),                        "SocialAgent 缺少 beta 属性"


def _test_agent_calc_heat_decay():
    """calc_heat_decay 接口（#11）：纯计算，无副作用。"""
    model = _make_single_model()
    agent = model.schedule.agents[0]
    assert hasattr(agent, "calc_heat_decay"), "SocialAgent 缺少 calc_heat_decay（#11）"

    # 无干预时，DORM 群自然衰减
    import math
    from types_def import GroupType
    # 创建一个 DORM agent
    from types_def import AgentType
    from social_agent import SocialAgent
    dorm_agent = SocialAgent(
        unique_id=888,
        model=model,
        agent_type=AgentType.ORDINARY,
        group_type=GroupType.DORM,
        init_config={"stance_prior": 0.0, "topic_id": "T001", "initial_heat": 0.5},
    )
    h0   = 1.0
    h1   = dorm_agent.calc_heat_decay(h0, elapsed_steps=5, intervention_tick=None)
    expected = h0 * math.exp(-0.15)   # ALPHA=0.15，DORM 无干预衰减
    assert abs(h1 - expected) < 1e-6, f"DORM 自然衰减结果 {h1:.4f}，期望 {expected:.4f}"

    # 非 DORM 触发干预后，衰减更快
    campus_agent = SocialAgent(
        unique_id=887,
        model=model,
        agent_type=AgentType.CONTROLLER,
        group_type=GroupType.CAMPUS,
        init_config={"stance_prior": 0.0, "topic_id": "T001", "initial_heat": 0.5},
    )
    h2 = campus_agent.calc_heat_decay(h0, elapsed_steps=10, intervention_tick=5)
    assert h2 < h1, f"CAMPUS 干预后衰减 {h2:.4f} 应 < DORM 自然衰减 {h1:.4f}"
    assert h2 >= 0.0, "热度不应为负"


def _test_agent_step_no_crash():
    model = _make_single_model()
    agent = model.schedule.agents[0]
    for _ in range(5):
        agent.step()


def _test_agent_opinion_range():
    from types_def import SimConfig
    from opinion_model import OpinionModel
    model = OpinionModel(SimConfig(
        n_agents=5,
        network_params={"m": 1},
        hawkes_params={"mu": 0.8, "alpha": 0.5, "beta": 1.0},
    ))
    for _ in range(20):
        model.step()
    for agent in model.schedule.agents:
        for op in agent.beliefs.opinions.values():
            assert -1.0 <= op.opinion_value <= 1.0, \
                f"Agent-{agent.unique_id} opinion_value={op.opinion_value} 超出 [-1,1]"
        e = agent.beliefs.emotion
        assert -1.0 <= e.valence <= 1.0
        assert  0.0 <= e.arousal <= 1.0


# ═══════════════════════════════════════════════════════════════════ #
# [A]  OpinionModel 核心流程（v2）
# ═══════════════════════════════════════════════════════════════════ #

def _test_model_datacollector_format():
    """DataCollector 含接口表§5 全部指标。"""
    from types_def import SimConfig
    from opinion_model import OpinionModel
    N_STEPS = 5
    model = OpinionModel(SimConfig(
        n_agents=3,
        network_params={"m": 1},
        n_steps=N_STEPS,
        hawkes_params={"mu": 0.5, "alpha": 0.3, "beta": 1.0},
    ))
    for _ in range(N_STEPS):
        model.step()
    df = model.datacollector.get_model_vars_dataframe()

    assert len(df) == N_STEPS, f"DataFrame 行数 {len(df)} ≠ 步数 {N_STEPS}"
    required = {
        "avg_opinion", "polarization", "emotional_contagion",
        "message_count", "negative_emotion", "distortion_level",
        "cross_group_forward", "cross_group_spread",
        "intervention_tick", "recovery_time",
    }
    missing = required - set(df.columns)
    assert not missing, f"缺少必需列：{missing}"
    assert df.isnull().sum().sum() == 0, "DataCollector 存在 NaN"
    assert (df["avg_opinion"].abs() <= 1.0).all()
    assert (df["polarization"] >= 0).all()
    assert (df["negative_emotion"] >= 0).all()
    assert (df["distortion_level"] >= 0).all()


def _routing_fixture():
    from types_def import SimConfig, GroupType
    from opinion_model import OpinionModel

    model = OpinionModel(SimConfig(
        n_agents=20,
        network_params={
            "m": 1,
            "group_membership_mode": "hierarchical",
            "forward_destination_strategy": "next_larger",
            "cross_group_spread_probability": 0.0,
        },
        hawkes_params={"mu": 0.5, "alpha": 0.3, "beta": 1.0},
        random_seed=42,
        scenario_params={
            # 路由类测试只验证「消息投递到哪个群」，因此关掉事件源与转发次数
            # 门槛这两个正交机制；它们各自有独立测试。
            "inject_initial_event": False,
            "cross_group_forward_threshold": 0,
            "secondary_event_probability": 0.0,
        },
    ))
    dorm_agents = [
        agent for agent in model.schedule.agents
        if agent.beliefs.identity.group_type == GroupType.DORM
    ]
    class_agents = [
        agent for agent in model.schedule.agents
        if agent.beliefs.identity.group_type == GroupType.CLASS
    ]
    assert len(dorm_agents) >= 2 and class_agents
    return model, dorm_agents[0], dorm_agents[1], class_agents[0]


def _test_model_submit_action_v2():
    """submit_action 按真实源消息与 source_group→destination_group 统计转发。"""
    from types_def import ActionRecord, ActionType, MessageType, GroupType

    model, origin, bridge, _ = _routing_fixture()
    original = ActionRecord(
        agent_id=origin.unique_id,
        action_type=ActionType.SEND_MESSAGE,
        content="宿舍群原始消息",
        topic_id="T001",
        distortion_level=0.0,
        message_type=MessageType.ORIGINAL,
        negative_score=0.2,
        heat=0.5,
        tick=0,
        group_id="GROUP_DORM",
    )
    assert model.submit_action(original)
    assert original.message_id

    before_forward = model.cross_group_forward
    before_class_heat = model.topic_heat["T001"][GroupType.CLASS]
    forwarded = ActionRecord(
        agent_id=bridge.unique_id,
        action_type=ActionType.FORWARD,
        content="转发到班级群",
        target_id=origin.unique_id,
        topic_id="T001",
        distortion_level=0.25,
        message_type=MessageType.PARAPHRASE,
        negative_score=0.2,
        heat=0.2,
        tick=0,
        group_id="GROUP_CLASS",
        source_message_id=original.message_id,
    )
    assert model.submit_action(forwarded)
    assert forwarded.source_group_id == "GROUP_DORM"
    assert forwarded.group_id == "GROUP_CLASS"
    assert forwarded.target_id == origin.unique_id
    assert model.cross_group_forward == before_forward + 1
    assert model.topic_heat["T001"][GroupType.CLASS] > before_class_heat


def _test_true_cross_group_delivery_and_rule_forward():
    """小群原文不可见；桥接成员真实转发后，大群成员才看到新记录。"""
    from types_def import AgentType, ActionRecord, ActionType, MessageType

    model, origin, bridge, class_viewer = _routing_fixture()
    original = ActionRecord(
        agent_id=origin.unique_id,
        action_type=ActionType.SEND_MESSAGE,
        content="宿舍群里的原始消息",
        topic_id="T001",
        message_type=MessageType.ORIGINAL,
        heat=0.2,
        tick=0,
        group_id="GROUP_DORM",
    )
    assert model.submit_action(original)

    assert original.message_id not in {
        record.message_id
        for record in model.get_group_messages(class_viewer.unique_id)
    }, "非 DORM 成员不应随机看到宿舍群原文"

    bridge.beliefs.identity.agent_type = AgentType.ACTIVE
    bridge.beliefs.emotion.arousal = 0.5
    bridge.beliefs.emotion.valence = 0.0
    bridge.beliefs.psychology.personality.extraversion = 1.0
    bridge.beliefs.psychology.risk_aversion = 0.0
    perception = bridge._perceive()
    assert any(si.message_id == original.message_id for si in perception.recent_messages)

    original_random = model.random.random
    model.random.random = lambda: 0.0
    try:
        desires = bridge._infer_desires()
        share = next((d for d in desires if d.goal_type == "share"), None)
        assert share is not None
        assert share.source_message_id == original.message_id
        assert share.source_group_id == "GROUP_DORM"
        assert share.destination_group_id == "GROUP_CLASS"

        intention = bridge._plan_intentions(desires)
        assert intention.action_type == ActionType.FORWARD
        assert intention.destination_group_id == "GROUP_CLASS"
        before = model.cross_group_forward
        feedback = bridge._execute_action(intention)
        assert feedback.get("submitted") == 1.0
        assert model.cross_group_forward == before + 1
    finally:
        model.random.random = original_random

    visible_after = model.get_group_messages(class_viewer.unique_id)
    routed = [
        record for record in visible_after
        if record.action_type == ActionType.FORWARD
        and record.source_message_id == original.message_id
    ]
    assert routed, "目标班级群成员应看到真正投递到 GROUP_CLASS 的转发记录"
    assert all(record.group_id == "GROUP_CLASS" for record in routed)


def _test_llm_action_drives_execution():
    """LLM 的 action_type/source/target/destination 字段须直接形成真实行动。"""
    import json
    from types_def import ActionRecord, ActionType, MessageType

    model, origin, bridge, _ = _routing_fixture()
    original = ActionRecord(
        agent_id=origin.unique_id,
        action_type=ActionType.SEND_MESSAGE,
        content="供 LLM 选择的宿舍群消息",
        topic_id="T001",
        message_type=MessageType.ORIGINAL,
        heat=0.2,
        tick=0,
        group_id="GROUP_DORM",
    )
    assert model.submit_action(original)

    class StubLLM:
        def chat(self, prompt: str) -> str:
            assert original.message_id in prompt
            return json.dumps({
                "action_type": "FORWARD",
                "message_type": "forward",
                "content": "LLM 决定转发到校园群",
                "topic_id": "T001",
                "target_id": origin.unique_id,
                "source_message_id": original.message_id,
                "source_group_id": "GROUP_DORM",
                "destination_group_id": "GROUP_CAMPUS",
                "opinion_updates": {},
                "emotion_delta": {"valence": 0.0, "arousal": 0.0},
            }, ensure_ascii=False)

    model._llm_client = StubLLM()
    model.llm_action_enabled = True
    before = model.cross_group_forward
    bridge.step()
    record = bridge.pending_action
    assert record is not None, "LLM 合法行动不应被规则链覆盖"
    assert record.action_type == ActionType.FORWARD
    assert record.target_id == origin.unique_id
    assert record.source_message_id == original.message_id
    assert record.source_group_id == "GROUP_DORM"
    assert record.group_id == "GROUP_CAMPUS"
    assert record.message_type == MessageType.FORWARD
    assert model.cross_group_forward == before + 1


def _test_model_update_environment():
    """_update_environment 调用后热度应衰减，且不为负。"""
    from types_def import SimConfig, GroupType
    from opinion_model import OpinionModel
    model = OpinionModel(SimConfig(
        n_agents=5,
        network_params={"m": 1},
        hawkes_params={"mu": 0.5, "alpha": 0.3, "beta": 1.0},
    ))
    # 手动注入热度
    model.topic_heat["T001"][GroupType.CAMPUS] = 1.0
    model._update_environment()
    new_heat = model.topic_heat["T001"][GroupType.CAMPUS]
    assert new_heat < 1.0, "热度衰减后应 < 初始值"
    assert new_heat >= 0.0, "热度不应为负"


def _test_group_specific_heat_decay():
    """每个群必须使用自己的 β；干预后 CAMPUS 衰减最快。"""
    from types_def import SimConfig, GroupType
    from opinion_model import OpinionModel

    model = OpinionModel(SimConfig(
        n_agents=20,
        network_params={"m": 1, "cross_group_spread_probability": 0.0},
        hawkes_params={"mu": 0.5, "alpha": 0.3, "beta": 1.0},
        random_seed=42,
    ))
    model.cross_group_spread_probability = 0.0
    for group_type in GroupType:
        model.topic_heat["T001"][group_type] = 0.6  # 低于 THETA，避免新触发逻辑干扰
    model.intervention_tick[GroupType.DORM] = None
    model.intervention_tick[GroupType.CLASS] = 0
    model.intervention_tick[GroupType.MAJOR] = 0
    model.intervention_tick[GroupType.CAMPUS] = 0
    model.schedule.time = 1

    model._update_environment()
    h = model.topic_heat["T001"]
    assert h[GroupType.DORM] > h[GroupType.CLASS] > h[GroupType.MAJOR] > h[GroupType.CAMPUS], \
        f"群别 β 未生效：{ {g.name: h[g] for g in GroupType} }"


def _test_intervention_applies_same_tick():
    """H(t) 达阈值时，干预项应立即参与 H(t+1) 计算。"""
    import math
    from types_def import SimConfig, AgentType, GroupType, ALPHA, GROUP_BETA
    from opinion_model import OpinionModel

    model = OpinionModel(SimConfig(
        n_agents=20,
        network_params={"m": 1, "cross_group_spread_probability": 0.0},
        hawkes_params={"mu": 0.5, "alpha": 0.3, "beta": 1.0},
        random_seed=42,
        scenario_params={
            "intervention_trigger": "heat",     # 本测试专门验证热度触发路径
            "intervention_min_messages": 0,     # 纯热度触发不需要消息证据
            "inject_initial_event": False,
            "secondary_event_probability": 0.0,
        },
    ))
    model.cross_group_spread_probability = 0.0

    controller = list(model.schedule.agents)[0]
    controller.beliefs.identity.agent_type = AgentType.CONTROLLER
    controller.beliefs.identity.group_type = GroupType.CAMPUS
    controller.beliefs.identity.group_ids = ["GROUP_CAMPUS"]
    controller.beta = GROUP_BETA[GroupType.CAMPUS]

    model.topic_heat["T001"] = {g: 0.0 for g in GroupType}
    model.topic_heat["T001"][GroupType.CAMPUS] = 1.0
    model.intervention_tick[GroupType.CAMPUS] = None
    model.schedule.time = 0
    model._update_environment()

    expected = math.exp(-ALPHA) * math.exp(-GROUP_BETA[GroupType.CAMPUS])
    actual = model.topic_heat["T001"][GroupType.CAMPUS]
    assert model.intervention_tick[GroupType.CAMPUS] == 0
    assert abs(actual - expected) < 1e-9, \
        f"干预应在触发 tick 立即生效，actual={actual}, expected={expected}"


def _test_macro_spread_separate_from_forward():
    """宏观热度扩散只记 cross_group_spread，不得污染 Agent 转发数。"""
    from types_def import SimConfig, GroupType
    from opinion_model import OpinionModel

    model = OpinionModel(SimConfig(
        n_agents=4,
        network_params={"m": 1, "cross_group_spread_probability": 1.0},
        hawkes_params={"mu": 0.5, "alpha": 0.3, "beta": 1.0},
        scenario_params={
            "macro_spread_requires_threshold": False,   # 门槛有独立测试
            "inject_initial_event": False,
            "secondary_event_probability": 0.0,
        },
    ))
    model.topic_heat["T001"] = {g: 0.0 for g in GroupType}
    model.topic_heat["T001"][GroupType.DORM] = 1.0
    model._update_environment()

    assert model.cross_group_spread > 0, "概率为 1 时应发生宏观跨群热度扩散"
    assert model.cross_group_forward == 0, \
        "没有 Agent FORWARD 时，cross_group_forward 必须保持 0"


def _test_model_single_node():
    from types_def import SimConfig
    from opinion_model import OpinionModel
    model = OpinionModel(SimConfig(
        n_agents=1,
        hawkes_params={"mu": 0.5, "alpha": 0.3, "beta": 1.0},
    ))
    for _ in range(5):
        model.step()
    df = model.datacollector.get_model_vars_dataframe()
    assert len(df) == 5


def _test_model_place_agents_group_type():
    """_place_agents 须建立主群兼容字段与分层多群成员关系。"""
    from types_def import SimConfig, GroupType
    from opinion_model import OpinionModel
    model = OpinionModel(SimConfig(n_agents=10, network_params={"m": 1}))
    for agent in model.schedule.agents:
        assert hasattr(agent.beliefs.identity, "group_type"), \
            f"Agent-{agent.unique_id} 缺少 group_type"
        assert isinstance(agent.beliefs.identity.group_type, GroupType), \
            f"Agent-{agent.unique_id} group_type 类型错误"
        identity = agent.beliefs.identity
        assert identity.primary_group_id in identity.group_ids
        expected = {
            f"GROUP_{group_type.name}"
            for group_type in GroupType
            if int(group_type) >= int(identity.group_type)
        }
        assert set(identity.group_ids) == expected, \
            f"Agent-{agent.unique_id} 分层成员关系错误: {identity.group_ids}"




# ═══════════════════════════════════════════════════════════════════ #
# [A · v3] 场景一致性测试：逐条对应《场景设定》与流程图的 8 个问题
# ═══════════════════════════════════════════════════════════════════ #

def _v3_model(n=30, **scenario):
    from types_def import SimConfig
    from opinion_model import OpinionModel
    base = {"inject_initial_event": False, "secondary_event_probability": 0.0}
    base.update(scenario)
    return OpinionModel(SimConfig(
        n_agents=n,
        network_params={"m": 2, "cross_group_spread_probability": 0.0},
        hawkes_params={"mu": 0.5, "alpha": 0.3, "beta": 1.0},
        random_seed=42,
        scenario_params=base,
    ))


def _test_v3_negative_triggers_without_heat():
    """问题①：负面程度超阈必须触发干预，即使热度远低于 θ。"""
    from types_def import ActionRecord, ActionType, AgentType, GroupType, MessageType

    model = _v3_model(intervention_trigger="negative", intervention_min_messages=1)
    campus = [a for a in model.schedule.agents
              if "GROUP_CAMPUS" in a.beliefs.identity.group_ids]
    assert campus, "夹具里应存在校园群成员"
    campus[0].beliefs.identity.agent_type = AgentType.CONTROLLER

    for i in range(4):
        assert model.submit_action(ActionRecord(
            agent_id=campus[i % len(campus)].unique_id,
            action_type=ActionType.SEND_MESSAGE,
            content=f"高负面消息{i}",
            tick=0, topic_id="T001", message_type=MessageType.ORIGINAL,
            negative_score=0.95, heat=0.0, group_id="GROUP_CAMPUS",
        ))
    # 群负面用了 EMA 平滑（negative_smoothing=0.5），需要走满几个 tick 才收敛
    for _ in range(3):
        model._refresh_group_negative()
    model._evaluate_intervention_triggers()

    assert model.topic_heat["T001"][GroupType.CAMPUS] < model.heat_threshold, \
        "本测试要求热度低于阈值，否则测不出「负面高但不热」"
    assert model.topic_negative["T001"][GroupType.CAMPUS] >= model.negative_threshold
    assert model.intervention_tick[GroupType.CAMPUS] is not None, \
        "负面超阈却没有触发干预 —— 问题①未修复"
    assert model.intervention_tick[GroupType.DORM] is None, \
        "DORM 群 t_int = +∞，永不触发"


def _test_v3_event_source_injects():
    """问题②：事件源是独立模块，随机投到小群或大群，并留下 EventRecord。"""
    from types_def import SMALL_GROUPS, LARGE_GROUPS
    from event_source import EventSource

    model = _v3_model(inject_initial_event=True)
    assert isinstance(model.event_source, EventSource)
    events = model.event_source.events
    assert len(events) == 1, "初始事件应且仅应投放一次"
    e = events[0]
    assert e.is_initial and e.message_id
    assert e.group_type in tuple(SMALL_GROUPS) + tuple(LARGE_GROUPS)
    assert e.scope in {"small", "large"}
    assert any(r.message_id == e.message_id for r in model.info_stream_cache), \
        "事件消息必须真正进入信息流"
    assert model.topic_heat["T001"][e.group_type] >= e.heat - 1e-9, \
        "H₀ 应按事件初始热度写入目标群"


def _test_v3_forward_count_gate():
    """问题③：转发次数未达阈时，跨群通道必须关闭；达阈后才打开。"""
    from types_def import ActionRecord, ActionType, MessageType, ForwardDirection

    model = _v3_model(
        cross_group_forward_threshold=2,
        cross_group_forward_threshold_by_direction={"upward": 2, "lateral": 0, "downward": 3},
        cross_group_heat_bypass=False,
    )
    dorm = [a for a in model.schedule.agents
            if a.beliefs.identity.group_type.name == "DORM"]
    assert len(dorm) >= 2, "夹具里应有至少两个宿舍群成员"
    origin, bridge = dorm[0], dorm[1]

    original = ActionRecord(
        agent_id=origin.unique_id, action_type=ActionType.SEND_MESSAGE,
        content="宿舍群原文", tick=0, topic_id="T001",
        message_type=MessageType.ORIGINAL, heat=0.1, group_id="GROUP_DORM",
    )
    assert model.submit_action(original)
    assert original.forward_count == 0

    # fc=0 时 upward 通道关闭
    assert not model.is_cross_group_channel_open(original, ForwardDirection.UPWARD)
    candidates = model.get_forward_destination_candidates(
        bridge.unique_id, "GROUP_DORM", forward_count=0)
    assert "GROUP_CAMPUS" not in candidates, "未达门槛却开放了向上跨群通道"
    assert "GROUP_DORM" in candidates, "群内通道应始终可用（攒转发次数的唯一途径）"

    blocked_before = model.blocked_by_forward_gate
    bad = ActionRecord(
        agent_id=bridge.unique_id, action_type=ActionType.FORWARD,
        content="强行跨群", tick=0, topic_id="T001",
        group_id="GROUP_CAMPUS", source_message_id=original.message_id,
    )
    assert not model.submit_action(bad), "未达门槛的跨群转发必须被拒绝"
    assert model.blocked_by_forward_gate == blocked_before + 1

    # 两次群内接力后，fc 达阈，向上通道打开
    prev = original
    for i in range(2):
        relay = ActionRecord(
            agent_id=(bridge if i == 0 else origin).unique_id,
            action_type=ActionType.FORWARD, content=f"群内接力{i}", tick=0,
            topic_id="T001", group_id="GROUP_DORM", source_message_id=prev.message_id,
        )
        assert model.submit_action(relay)
        assert relay.forward_direction == ForwardDirection.INTRA
        assert relay.root_message_id == original.message_id
        prev = relay
    assert prev.forward_count == 2
    assert model.is_cross_group_channel_open(prev, ForwardDirection.UPWARD)

    ok = ActionRecord(
        agent_id=bridge.unique_id, action_type=ActionType.FORWARD,
        content="达阈后跨群", tick=0, topic_id="T001",
        group_id="GROUP_CAMPUS", source_message_id=prev.message_id,
    )
    before = model.cross_group_forward
    assert model.submit_action(ok), "达到门槛后跨群转发应被接受"
    assert ok.forward_direction == ForwardDirection.UPWARD
    assert model.cross_group_forward == before + 1
    assert model.upward_forward >= 1


def _test_v3_lateral_small_group_forward():
    """问题④：DORM↔CLASS 小群互转必须可用，且方向判定正确。"""
    from types_def import ActionRecord, ActionType, MessageType, GroupType, ForwardDirection

    model = _v3_model()
    fd = model.forward_direction
    assert fd(GroupType.DORM, GroupType.CLASS) == ForwardDirection.LATERAL
    assert fd(GroupType.CLASS, GroupType.DORM) == ForwardDirection.LATERAL
    assert fd(GroupType.DORM, GroupType.CAMPUS) == ForwardDirection.UPWARD
    assert fd(GroupType.MAJOR, GroupType.CAMPUS) == ForwardDirection.UPWARD, \
        "专业群→校园群是「小→大」，不是横向"
    assert fd(GroupType.CAMPUS, GroupType.CLASS) == ForwardDirection.DOWNWARD
    assert fd(GroupType.DORM, GroupType.DORM) == ForwardDirection.INTRA

    dorm = [a for a in model.schedule.agents
            if a.beliefs.identity.group_type == GroupType.DORM]
    origin, bridge = dorm[0], dorm[1]
    original = ActionRecord(
        agent_id=origin.unique_id, action_type=ActionType.SEND_MESSAGE,
        content="宿舍群原文", tick=0, topic_id="T001",
        message_type=MessageType.ORIGINAL, heat=0.1, group_id="GROUP_DORM",
    )
    assert model.submit_action(original)

    lateral = ActionRecord(
        agent_id=bridge.unique_id, action_type=ActionType.FORWARD,
        content="转到班级群", tick=0, topic_id="T001",
        group_id="GROUP_CLASS", source_message_id=original.message_id,
    )
    before = model.lateral_forward
    assert model.submit_action(lateral), "小群互转门槛为 0，应直接放行"
    assert lateral.forward_direction == ForwardDirection.LATERAL
    assert model.lateral_forward == before + 1

    # 向下通道默认开启但权重最低，且门槛最高
    assert model.forward_threshold_for(ForwardDirection.DOWNWARD) > \
        model.forward_threshold_for(ForwardDirection.UPWARD) > \
        model.forward_threshold_for(ForwardDirection.LATERAL)
    w = model.forward_direction_weights
    assert w[ForwardDirection.UPWARD] > w[ForwardDirection.LATERAL] > w[ForwardDirection.DOWNWARD], \
        "方向权重须满足 小→大 > 横向 > 向下"


def _test_v3_mute_is_enforced():
    """问题⑤：禁言必须真正阻断发言，且 DORM 群不可禁言。"""
    from types_def import ActionRecord, ActionType, AgentType, GroupType, MessageType

    model = _v3_model(enable_mute=True, mute_duration_ticks=3)
    campus = [a for a in model.schedule.agents
              if "GROUP_CAMPUS" in a.beliefs.identity.group_ids]
    controller, victim = campus[0], campus[1]
    controller.beliefs.identity.agent_type = AgentType.CONTROLLER
    victim.beliefs.identity.agent_type = AgentType.ORDINARY

    mute = ActionRecord(
        agent_id=controller.unique_id, action_type=ActionType.MUTE,
        content="禁言公告", target_id=victim.unique_id, tick=0, topic_id="T001",
        message_type=MessageType.CLARIFICATION, group_id="GROUP_CAMPUS",
    )
    assert model.submit_action(mute), "Controller 的禁言动作应被接受"
    assert model.mute_count == 1
    assert model.is_muted(victim.unique_id, "GROUP_CAMPUS")
    assert "GROUP_CAMPUS" in model.get_muted_groups(victim.unique_id)
    assert model.governance_tick[GroupType.CAMPUS] is not None, \
        "禁言应标记 Controller 已实际干预（恢复机制的起点）"

    blocked = ActionRecord(
        agent_id=victim.unique_id, action_type=ActionType.SEND_MESSAGE,
        content="我还想说话", tick=0, topic_id="T001", group_id="GROUP_CAMPUS",
    )
    assert not model.submit_action(blocked), "被禁言者不应还能发言 —— 问题⑤未修复"
    assert model.blocked_by_mute >= 1

    # 非 Controller 不能禁言
    bad = ActionRecord(
        agent_id=victim.unique_id, action_type=ActionType.MUTE,
        content="越权禁言", target_id=controller.unique_id, tick=0,
        topic_id="T001", group_id="GROUP_CAMPUS",
    )
    assert not model.submit_action(bad)

    # DORM 群管理强度最低，禁言不成立
    dorm_ctrl = [a for a in model.schedule.agents
                 if a.beliefs.identity.group_type == GroupType.DORM]
    dorm_ctrl[0].beliefs.identity.agent_type = AgentType.CONTROLLER
    dorm_mute = ActionRecord(
        agent_id=dorm_ctrl[0].unique_id, action_type=ActionType.MUTE,
        content="宿舍群禁言", target_id=dorm_ctrl[1].unique_id, tick=0,
        topic_id="T001", group_id="GROUP_DORM",
    )
    assert not model.submit_action(dorm_mute), "DORM 群不应允许禁言"

    # 到期自动解禁
    model.schedule.time = 99
    model._update_environment()
    assert not model.is_muted(victim.unique_id, "GROUP_CAMPUS")


def _test_v3_exposure_effects():
    """问题⑥：曝光必须同时影响热度增益与注意力配额。"""
    from types_def import ActionRecord, ActionType, MessageType, GroupType, GROUP_EXPOSURE

    assert (GROUP_EXPOSURE[GroupType.DORM] < GROUP_EXPOSURE[GroupType.CLASS]
            < GROUP_EXPOSURE[GroupType.MAJOR] < GROUP_EXPOSURE[GroupType.CAMPUS])

    model = _v3_model(exposure_heat_weight=1.0)
    dorm_agent = next(a for a in model.schedule.agents
                      if a.beliefs.identity.group_type == GroupType.DORM)

    def _post(group_id):
        rec = ActionRecord(
            agent_id=dorm_agent.unique_id, action_type=ActionType.SEND_MESSAGE,
            content="同一条消息", tick=0, topic_id="T001",
            message_type=MessageType.ORIGINAL, negative_score=0.5, heat=1.0,
            group_id=group_id,
        )
        assert model.submit_action(rec)

    _post("GROUP_DORM")
    _post("GROUP_CAMPUS")
    h_dorm = model.topic_heat["T001"][GroupType.DORM]
    h_campus = model.topic_heat["T001"][GroupType.CAMPUS]
    assert h_campus > h_dorm, \
        f"同一条消息在高曝光群应掀起更高热度，实际 CAMPUS={h_campus:.4f} DORM={h_dorm:.4f}"

    # 注意力配额：低曝光群也必须保底有名额，否则小群消息永远被大群刷掉
    for i in range(60):
        _post("GROUP_CAMPUS")
    visible = model.get_group_messages(dorm_agent.unique_id, limit=20)
    groups = {r.group_id for r in visible}
    assert "GROUP_DORM" in groups, "低曝光群的消息不应被高曝光群完全挤掉"
    assert len(visible) <= 20


def _test_v3_control_level_formula():
    """问题⑦：「基础降温 × control_level」必须与 §4.2 指数公式逐位等价。"""
    import math
    from types_def import (ALPHA, GROUP_BETA, GROUP_CONTROL_LEVEL, GroupType,
                           heat_decay)

    for g in GroupType:
        cl = GROUP_CONTROL_LEVEL[g]
        assert cl > 1.0, "control_level 必须 > 1"
        assert abs(GROUP_BETA[g] - ALPHA * (cl - 1.0)) < 1e-12, \
            "β_k 必须等于 α·(control_level_k − 1)"
        lhs = heat_decay(1.0, g, intervened=True)                 # 文字版
        rhs = math.exp(-ALPHA) * math.exp(-GROUP_BETA[g])          # §4.2 公式版
        assert abs(lhs - rhs) < 1e-12, f"{g.name} 两种表述不等价"
    assert (GROUP_CONTROL_LEVEL[GroupType.DORM] < GROUP_CONTROL_LEVEL[GroupType.CLASS]
            < GROUP_CONTROL_LEVEL[GroupType.MAJOR] < GROUP_CONTROL_LEVEL[GroupType.CAMPUS])
    # 管理越强降得越快
    assert (heat_decay(1.0, GroupType.CAMPUS, True)
            < heat_decay(1.0, GroupType.MAJOR, True)
            < heat_decay(1.0, GroupType.CLASS, True)
            < heat_decay(1.0, GroupType.DORM, True))


def _test_v3_trust_and_bias_modeled():
    """问题⑧：trust / confirmation_bias 必须单独建模，且按角色分层。"""
    from types_def import PsychologyBelief, AgentType

    psy = PsychologyBelief()
    assert hasattr(psy, "trust") and hasattr(psy, "confirmation_bias")

    model = _v3_model(n=120)
    by_role = {}
    for agent in model.schedule.agents:
        role = agent.beliefs.identity.agent_type
        p = agent.beliefs.psychology
        assert 0.0 <= p.trust <= 1.0 and 0.0 <= p.confirmation_bias <= 1.0
        by_role.setdefault(role, []).append((p.trust, p.confirmation_bias))

    def _mean(role, idx):
        vals = [v[idx] for v in by_role.get(role, [])]
        return sum(vals) / len(vals) if vals else 0.5

    assert _mean(AgentType.RATIONAL, 0) < _mean(AgentType.ORDINARY, 0), \
        "理性讨论者的信任度应低于普通群员"
    assert _mean(AgentType.RATIONAL, 1) < _mean(AgentType.ACTIVE, 1), \
        "理性讨论者的确认偏误应低于活跃讨论者"


def _test_v3_full_cycle_smoke():
    """端到端：事件源 → 扩散 → 负面超阈 → Controller 干预 → 恢复。"""
    from types_def import SimConfig
    from opinion_model import OpinionModel

    triggered = governed = recovered = 0
    for seed in (11, 42, 77, 2026):
        model = OpinionModel(SimConfig(n_agents=80, network_params={"m": 3},
                                       random_seed=seed))
        for _ in range(50):
            model.step()
        summary = model.get_final_summary()
        assert summary["event_source"]["n_events"] >= 1
        triggered += summary["earliest_intervention_tick"] is not None
        governed += any(t is not None for t in model.governance_tick.values())
        recovered += summary["recovery_time"] is not None
        df = model.datacollector.get_model_vars_dataframe()
        assert df.isnull().sum().sum() == 0
    assert triggered >= 2, f"4 个种子里只有 {triggered} 个触发干预，触发条件可能过严"
    assert governed >= 2, "Controller 应在多数种子下实际动手"


# ═══════════════════════════════════════════════════════════════════ #
# 执行所有测试
# ═══════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  接口契约验证  (test_contract.py · v3 · 场景对齐版)")
    print("═" * 60)

    print("\n── [types]  types_def.py 共享类型（v2）")
    _check("v2 枚举值 / 新字段 / 常量全量验证", _test_types_import)

    print("\n── [C]  HawkesEngine 接口")
    _check("__init__ 含参数校验",              _test_hawkes_init)
    _check("intensity(t) 返回 float，无事件=mu", _test_hawkes_intensity)
    _check("sample_next_time > current_t",     _test_hawkes_sample)
    _check("add_event 保持 history 升序",      _test_hawkes_add_event)

    print("\n── [B]  SocialAgent 接口（v2）")
    _check("__init__ 接受 group_type 参数",    _test_agent_init_group_type)
    _check("beliefs/pending_action/beta/group_type/nickname 属性存在", _test_agent_attributes_v2)
    _check("calc_heat_decay(#11) 接口与公式",  _test_agent_calc_heat_decay)
    _check("step() 连续5次不崩溃",             _test_agent_step_no_crash)
    _check("20步后 opinion/emotion 值域 [-1,1]", _test_agent_opinion_range)

    print("\n── [A]  OpinionModel 核心流程（v2）")
    _check("DataCollector 含§5全部指标列",     _test_model_datacollector_format)
    _check("submit_action 按真实消息路由统计跨群转发", _test_model_submit_action_v2)
    _check("小群原文隔离，规则转发后目标群可见", _test_true_cross_group_delivery_and_rule_forward)
    _check("LLM action/target/群路由直接驱动执行", _test_llm_action_drives_execution)
    _check("_update_environment 热度衰减≥0",   _test_model_update_environment)
    _check("各群按自身 β 衰减（CAMPUS 最快）", _test_group_specific_heat_decay)
    _check("干预衰减在触发 tick 立即生效", _test_intervention_applies_same_tick)
    _check("宏观扩散与 Agent 转发指标分离", _test_macro_spread_separate_from_forward)
    _check("_place_agents 建立分层多群成员关系", _test_model_place_agents_group_type)
    _check("n=1 单节点 5步不崩溃",             _test_model_single_node)

    print("\n── [A · v3]  场景一致性（对应《场景设定》与流程图的 8 个问题）")
    _check("① 负面超阈触发干预（热度不高也触发）", _test_v3_negative_triggers_without_heat)
    _check("② 事件源独立模块随机投小群/大群",      _test_v3_event_source_injects)
    _check("③ 转发次数门槛控制跨群通道开闭",       _test_v3_forward_count_gate)
    _check("④ DORM↔CLASS 小群互转与方向判定",     _test_v3_lateral_small_group_forward)
    _check("⑤ 禁言真实阻断发言、DORM 不可禁言",   _test_v3_mute_is_enforced)
    _check("⑥ 曝光影响热度增益与注意力配额",       _test_v3_exposure_effects)
    _check("⑦ 基础降温×control_level 与§4.2等价", _test_v3_control_level_formula)
    _check("⑧ trust/confirmation_bias 分角色建模", _test_v3_trust_and_bias_modeled)
    _check("端到端：投放→扩散→超阈→干预→恢复",   _test_v3_full_cycle_smoke)

    n_pass = sum(1 for r in _results if r[0] == "✅")
    n_fail = sum(1 for r in _results if r[0] == "❌")
    total  = len(_results)

    print("\n" + "─" * 60)
    print(f"  通过 {n_pass}/{total}    失败 {n_fail}/{total}")
    if n_fail == 0:
        print("  ✅  全部通过，接口兼容，可交付")
    else:
        fails = [r[1] for r in _results if r[0] == "❌"]
        print(f"  ❌  失败项：{fails}")
        print("  请修复后再交付给下游成员")
    print("─" * 60 + "\n")

    sys.exit(0 if n_fail == 0 else 1)
