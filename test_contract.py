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
    ))
    model.cross_group_spread_probability = 0.0

    controller = list(model.schedule.agents)[0]
    controller.beliefs.identity.agent_type = AgentType.CONTROLLER
    controller.beliefs.identity.group_type = GroupType.CAMPUS
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
# 执行所有测试
# ═══════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  接口契约验证  (test_contract.py · v2 · W4)")
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
