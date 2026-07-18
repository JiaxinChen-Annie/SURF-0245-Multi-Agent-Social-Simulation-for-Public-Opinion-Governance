"""
test_contract.py — 接口契约验证脚本（v2，W4）
----------------------------------------------
更新内容：
  - AgentType: ORDINARY/ACTIVE/RATIONAL/CONTROLLER
  - ActionType: SEND_MESSAGE/REPLY/FORWARD/SILENT（移除 LIKE）
  - GroupType: DORM/CLASS/MAJOR/CAMPUS
  - MessageType 字符串常量
  - ActionRecord 新增字段检查（topic_id/distortion_level/message_type/negative_score/heat）
  - submit_action 新增 topic_heat/topic_negative/cross_group_forward 检查
  - SocialAgent.__init__ 新增 group_type 参数验证
  - calc_heat_decay 接口验证（#11）
  - Perception 新增字段检查

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

    # Perception v2 字段
    p = Perception()
    assert hasattr(p, "group_id"),        "Perception 缺少 group_id"
    assert hasattr(p, "group_type"),      "Perception 缺少 group_type"
    assert hasattr(p, "beta"),            "Perception 缺少 beta"
    assert hasattr(p, "recent_messages"), "Perception 缺少 recent_messages（原 neighbor_actions）"
    assert not hasattr(p, "neighbor_actions"), "neighbor_actions 已重命名"
    assert hasattr(p, "topic_heat"),      "Perception 缺少 topic_heat"
    assert hasattr(p, "topic_negative"),  "Perception 缺少 topic_negative"

    # SocialInfo v2 字段
    si = SocialInfo()
    assert hasattr(si, "source_nickname"),  "SocialInfo 缺少 source_nickname"
    assert hasattr(si, "message_type"),     "SocialInfo 缺少 message_type"
    assert hasattr(si, "is_mention"),       "SocialInfo 缺少 is_mention"
    assert hasattr(si, "topic_id"),         "SocialInfo 缺少 topic_id"
    assert hasattr(si, "distortion_level"), "SocialInfo 缺少 distortion_level"
    assert hasattr(si, "negative_score"),   "SocialInfo 缺少 negative_score"
    assert hasattr(si, "heat"),             "SocialInfo 缺少 heat"

    # Desire / Intention v2 字段
    d = Desire()
    assert hasattr(d, "topic_id"),  "Desire 缺少 topic_id"
    assert hasattr(d, "target_id"), "Desire 缺少 target_id"
    assert not hasattr(d, "event_id"), "Desire 不应有 event_id"
    it = Intention()
    assert hasattr(it, "topic_id"), "Intention 缺少 topic_id"
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
        "cross_group_forward", "intervention_tick", "recovery_time",
    }
    missing = required - set(df.columns)
    assert not missing, f"缺少必需列：{missing}"
    assert df.isnull().sum().sum() == 0, "DataCollector 存在 NaN"
    assert (df["avg_opinion"].abs() <= 1.0).all()
    assert (df["polarization"] >= 0).all()
    assert (df["negative_emotion"] >= 0).all()
    assert (df["distortion_level"] >= 0).all()


def _test_model_submit_action_v2():
    """submit_action 更新 topic_heat / topic_negative / cross_group_forward。"""
    from types_def import SimConfig, ActionRecord, ActionType, MessageType
    from opinion_model import OpinionModel
    model = OpinionModel(SimConfig(
        n_agents=2,
        hawkes_params={"mu": 0.5, "alpha": 0.3, "beta": 1.0},
    ))
    before_cache = len(model.info_stream_cache)
    before_fwd   = model.cross_group_forward

    r = ActionRecord(
        agent_id=0,
        action_type=ActionType.SEND_MESSAGE,
        content="测试消息",
        topic_id="T001",          # v2：topic_id 而非 event_id
        distortion_level=0.1,
        message_type=MessageType.ORIGINAL,
        negative_score=0.2,
        heat=0.5,
        tick=0,
    )
    model.submit_action(r)

    assert len(model.info_stream_cache) == before_cache + 1, "缓存长度应 +1"
    assert "T001" in model.topic_heat, "topic_heat 应含 T001"
    assert "T001" in model.topic_negative, "topic_negative 应含 T001"

    # heat 已更新（大于初始 0.0）
    from types_def import GroupType
    total_heat = sum(model.topic_heat["T001"].values())
    assert total_heat > 0.0, "submit_action 后 topic_heat 应增加"


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
    """_place_agents 须给每个 agent 分配 group_type（v2 要求）。"""
    from types_def import SimConfig, GroupType
    from opinion_model import OpinionModel
    model = OpinionModel(SimConfig(n_agents=10, network_params={"m": 1}))
    for agent in model.schedule.agents:
        assert hasattr(agent.beliefs.identity, "group_type"), \
            f"Agent-{agent.unique_id} 缺少 group_type"
        assert isinstance(agent.beliefs.identity.group_type, GroupType), \
            f"Agent-{agent.unique_id} group_type 类型错误"


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
    _check("submit_action 写缓存+热度+负面度", _test_model_submit_action_v2)
    _check("_update_environment 热度衰减≥0",   _test_model_update_environment)
    _check("_place_agents 给每个 agent 分配 group_type", _test_model_place_agents_group_type)
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
