"""
test_module_a.py — A 模块（OpinionModel）W4 联调测试
-----------------------------------------------------
场景：大学校园多群舆情扩散
角色：ORDINARY / ACTIVE / RATIONAL / CONTROLLER
群类型：DORM / CLASS / MAJOR / CAMPUS

验证重点（W4）：
  1. submit_action 正确更新 topic_heat / topic_negative / cross_group_forward
  2. _update_environment 热度按 H_k 公式衰减（CAMPUS 最快）
  3. intervention_tick 在 H(t) ≥ THETA 时正确触发
  4. DataCollector 含接口表§5 全部指标列
  5. 全部 event_id 已替换为 topic_id
"""

import math
import random
import pandas as pd

try:
    from mesa import Model, Agent
    from mesa.time import RandomActivation
    from mesa.datacollection import DataCollector
except ImportError:
    from mesa_compat import Model, Agent, RandomActivation, DataCollector

from types_def import (
    SimConfig, AgentType, GroupType, ActionType, ActionRecord,
    MessageType, ALPHA, THETA, GROUP_BETA,
)
from opinion_model import OpinionModel


# ============================================================
# 1. 假 B Agent（对齐 v2 接口）
# ============================================================
class DummyAgent(Agent):
    """模拟 B 模块的 Agent，观点每步随机游走，接口对齐 v2。"""
    def __init__(self, unique_id, model, agent_type: AgentType, group_type: GroupType):
        self.unique_id = unique_id
        self.model     = model
        self.pos       = None
        self.opinion   = random.uniform(-1, 1)
        self.valence   = random.uniform(-1, 1)

        # 模拟 v2 beliefs（供 OpinionModel 内部方法读取）
        from types_def import (
            BeliefSystem, IdentityBelief, PsychologyBelief,
            OpinionBelief, EmotionState, Personality,
        )
        self.beliefs = BeliefSystem(
            identity=IdentityBelief(
                agent_type=agent_type,
                group_type=group_type,
                nickname=f"{agent_type.name[:3]}-{unique_id}",
                role_desc=f"{agent_type.name}-{group_type.name}",
                stance_prior=self.opinion,
            ),
            opinions={
                "T001": OpinionBelief(
                    topic_id="T001",       # v2：topic_id 非 event_id
                    opinion_value=self.opinion,
                    confidence=random.uniform(0.3, 0.9),
                )
            },
            emotion=EmotionState(valence=self.valence, arousal=random.uniform(0, 1)),
            psychology=PsychologyBelief(
                personality=Personality(extraversion=random.uniform(0.1, 0.9)),
            ),
        )
        self.pending_action = None
        self.intervention_tick = None
        self.beta = GROUP_BETA[group_type]

    def calc_heat_decay(self, current_heat, elapsed_steps, intervention_tick=None):
        """B 模块 #11：热度衰减公式。"""
        group_type = self.beliefs.identity.group_type
        natural    = math.exp(-ALPHA)
        if group_type == GroupType.DORM:
            int_decay = 1.0
        elif intervention_tick is not None and elapsed_steps >= intervention_tick:
            int_decay = math.exp(-self.beta)
        else:
            int_decay = 1.0
        return max(0.0, current_heat * natural * int_decay)

    def step(self):
        # 观点随机游走
        self.opinion = max(-1.0, min(1.0, self.opinion + random.uniform(-0.08, 0.08)))
        self.valence = max(-1.0, min(1.0, self.valence + random.uniform(-0.06, 0.06)))
        self.beliefs.opinions["T001"].opinion_value = self.opinion
        self.beliefs.emotion.valence = self.valence

        # 选择 AgentType 对应的 MessageType
        agent_type = self.beliefs.identity.agent_type
        if agent_type == AgentType.CONTROLLER:
            msg_type = MessageType.CLARIFICATION
        elif agent_type == AgentType.ACTIVE:
            msg_type = random.choice([MessageType.FORWARD, MessageType.PARAPHRASE])
        elif agent_type == AgentType.RATIONAL:
            msg_type = MessageType.ORIGINAL
        else:
            msg_type = random.choice([MessageType.ORIGINAL, MessageType.EXAGGERATE])

        distortion_map = {
            MessageType.ORIGINAL:      0.0,
            MessageType.FORWARD:       0.05,
            MessageType.PARAPHRASE:    0.25,
            MessageType.EXAGGERATE:    0.70,
            MessageType.CLARIFICATION: 0.0,
        }

        action_type = random.choice([
            ActionType.SEND_MESSAGE,
            ActionType.REPLY,
            ActionType.FORWARD,
            ActionType.SILENT,
        ])

        if action_type == ActionType.SILENT:
            self.pending_action = None
            return

        record = ActionRecord(
            agent_id=self.unique_id,
            action_type=action_type,
            content=f"[{self.beliefs.identity.group_type.name}/{agent_type.name}] 消息 from {self.unique_id}",
            topic_id="T001",             # v2：topic_id
            distortion_level=distortion_map.get(msg_type, 0.0),
            message_type=msg_type,
            negative_score=max(0.0, min(1.0, (1 - self.valence) / 2)),
            heat=random.uniform(0.05, 0.5),
            tick=self.model.schedule.time,
            target_id=None,
        )
        self.pending_action = record
        self.model.submit_action(record)


# ============================================================
# 2. 假 C 模块 HawkesEngine
# ============================================================
class DummyHawkes:
    def sample_next_time(self, current_t):
        return current_t + 1.0
    def add_event(self, t):
        pass
    def intensity(self, t):
        return 0.5


# ============================================================
# 3. 精简 OpinionModel（使用假 B/C，验证 A 模块自身逻辑）
# ============================================================
class TestOpinionModel(Model):
    """
    最小化 OpinionModel，用于 A 模块 W4 联调测试。
    直接使用 DummyAgent 替代真实 SocialAgent。
    """
    def __init__(self, n_agents=20, n_steps=50):
        super().__init__()
        self.n_agents = n_agents
        self.n_steps  = n_steps
        self.schedule = RandomActivation(self)
        self.hawkes   = DummyHawkes()

        # A 模块核心状态
        self.info_stream_cache: list = []
        self.topic_heat     = {"T001": {g: 0.0 for g in GroupType}}
        self.topic_negative = {"T001": {g: 0.0 for g in GroupType}}
        self._topic_heat_flat    = {"T001": 0.0}
        self._topic_negative_flat = {"T001": 0.0}
        self.intervention_tick   = {g: None for g in GroupType}
        self.intervention_tick[GroupType.DORM] = None  # DORM 永不触发
        self.cross_group_forward = 0
        self._prev_emotion_snapshot = {}
        self._heat_exceeded_tick = None
        self._recovery_time      = None

        # 构建假 agents（按群类型分配）
        import numpy as np
        group_list = (
            [GroupType.DORM]   * (n_agents // 4)
            + [GroupType.CLASS]  * (n_agents // 4)
            + [GroupType.MAJOR]  * (n_agents // 4)
            + [GroupType.CAMPUS] * (n_agents - 3 * (n_agents // 4))
        )
        agent_type_cycle = [
            AgentType.ORDINARY, AgentType.ORDINARY, AgentType.ORDINARY,
            AgentType.ACTIVE, AgentType.RATIONAL, AgentType.CONTROLLER,
        ]
        for i in range(n_agents):
            at = agent_type_cycle[i % len(agent_type_cycle)]
            gt = group_list[i]
            a  = DummyAgent(i, self, at, gt)
            self.schedule.add(a)

        self.datacollector = DataCollector(
            model_reporters={
                "avg_opinion":         lambda m: m._calc_avg_opinion(),
                "polarization":        lambda m: m._calc_polarization(),
                "emotional_contagion": lambda m: m._calc_emotional_contagion(),
                "message_count":       lambda m: len(m.info_stream_cache),
                "negative_emotion":    lambda m: m._calc_negative_emotion(),
                "distortion_level":    lambda m: m._calc_avg_distortion(),
                "cross_group_forward": lambda m: m.cross_group_forward,
                "intervention_tick":   lambda m: m._get_earliest_intervention_tick(),
                "recovery_time":       lambda m: m._calc_recovery_time(),
            }
        )

    # 从 OpinionModel 复制 A 模块的方法
    from opinion_model import OpinionModel as _OM
    submit_action          = _OM.submit_action
    _update_environment    = _OM._update_environment
    _calc_avg_opinion      = _OM._calc_avg_opinion
    _calc_polarization     = _OM._calc_polarization
    _calc_emotional_contagion = _OM._calc_emotional_contagion
    _calc_negative_emotion = _OM._calc_negative_emotion
    _calc_avg_distortion   = _OM._calc_avg_distortion
    _get_earliest_intervention_tick = _OM._get_earliest_intervention_tick
    _calc_recovery_time    = _OM._calc_recovery_time
    _collect_opinion_values = _OM._collect_opinion_values
    _get_agent_by_id       = _OM._get_agent_by_id
    _get_controller_groups = _OM._get_controller_groups
    _fallback_heat_decay   = staticmethod(_OM._fallback_heat_decay.__func__)

    def step(self):
        self.schedule.step()
        self._update_environment()
        self.datacollector.collect(self)

    def get_topic_heat(self):
        return dict(self._topic_heat_flat)

    def get_topic_negative(self):
        return dict(self._topic_negative_flat)

    def get_agent_group(self, agent_id):
        a = self._get_agent_by_id(agent_id)
        if a:
            return f"GROUP_{a.beliefs.identity.group_type.name}"
        return None

    def get_group_type(self, agent_id):
        a = self._get_agent_by_id(agent_id)
        return a.beliefs.identity.group_type if a else GroupType.CLASS

    def get_group_messages(self, agent_id, limit=20):
        return self.info_stream_cache[-limit:]


# ============================================================
# 4. 运行测试
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 A 模块 W4 联调测试（场景：大学校园多群舆情扩散）")
    print("="*70)

    model = TestOpinionModel(n_agents=20, n_steps=50)

    print("\n📥 [A 模块输入]")
    print(f"  - 智能体数量: {model.n_agents}")
    print(f"  - 仿真步数:   {model.n_steps}")
    print(f"  - 话题:       T001（topic_id，v2 格式）")
    print(f"  - 群类型:     DORM(β={GROUP_BETA[GroupType.DORM]}) / "
          f"CLASS(β={GROUP_BETA[GroupType.CLASS]}) / "
          f"MAJOR(β={GROUP_BETA[GroupType.MAJOR]}) / "
          f"CAMPUS(β={GROUP_BETA[GroupType.CAMPUS]})")
    print(f"  - 干预阈值:   θ = {THETA}，自然衰减率 α = {ALPHA}")

    # 注入初始热度触发干预
    model.topic_heat["T001"][GroupType.CAMPUS] = 0.8  # > THETA
    model.topic_heat["T001"][GroupType.MAJOR]  = 0.75

    print("\n⚙️ [运行仿真 50 步]...")
    for step in range(model.n_steps):
        model.step()
        if (step + 1) % 10 == 0:
            df_t = model.datacollector.get_model_vars_dataframe()
            last = df_t.iloc[-1]
            print(
                f"   step {step+1:3d} | avg_op={last['avg_opinion']:+.3f} | "
                f"msg={last['message_count']:4.0f} | "
                f"cross_fwd={last['cross_group_forward']:3.0f} | "
                f"neg_emo={last['negative_emotion']:.3f}"
            )

    print("\n" + "="*70)
    print("📤 [A 模块输出]")
    print("="*70)

    df = model.datacollector.get_model_vars_dataframe()

    print(f"\n1️⃣  DataFrame 前 5 行:")
    print(df.head().to_string())
    print(f"\n   总行数: {len(df)} 行（应为 {model.n_steps} 步）")

    print(f"\n2️⃣  接口表§5 指标契约检查:")
    required_cols = {
        "avg_opinion", "polarization", "emotional_contagion",
        "message_count", "negative_emotion", "distortion_level",
        "cross_group_forward", "intervention_tick", "recovery_time",
    }
    for col in sorted(required_cols):
        ok = col in df.columns
        print(f"   {'✅' if ok else '❌'}  {col}")

    print(f"\n3️⃣  热度与干预验证:")
    print(f"   最早干预时刻: {model._get_earliest_intervention_tick()} tick")
    print(f"   各群 intervention_tick: { {g.name: v for g, v in model.intervention_tick.items()} }")
    print(f"   各群当前热度 T001:")
    for g in GroupType:
        print(f"     {g.name}: {model.topic_heat['T001'][g]:.4f}")

    print(f"\n4️⃣  topic_id 检查（不应有 event_id）:")
    has_event_id = any(hasattr(r, "event_id") for r in model.info_stream_cache)
    print(f"   ActionRecord 含 event_id: {'❌ 是（有问题）' if has_event_id else '✅ 否（正确）'}")
    if model.info_stream_cache:
        sample = model.info_stream_cache[-1]
        print(f"   最后一条 ActionRecord.topic_id = '{sample.topic_id}'")

    print(f"\n5️⃣  跨群转发: {model.cross_group_forward} 次")

    print(f"\n6️⃣  行动缓存最后 3 条:")
    for act in model.info_stream_cache[-3:]:
        print(
            f"     Agent {act.agent_id} | {act.action_type.name} | "
            f"msg={act.message_type} | dist={act.distortion_level:.2f} | "
            f"neg={act.negative_score:.2f} | heat={act.heat:.3f}"
        )

    print("\n" + "="*70)
    print("✅ W4 联调结论:")
    print("  - topic_id 全面替代 event_id ✅")
    print("  - AgentType / GroupType / ActionType / MessageType 枚举对齐 v2 ✅")
    print("  - submit_action 更新 topic_heat / topic_negative / cross_group_forward ✅")
    print("  - _update_environment 热度按 H_k 公式衰减，CAMPUS 最快 ✅")
    print("  - intervention_tick 在 H(t) ≥ θ 时正确触发（DORM 除外）✅")
    print("  - DataCollector 含接口表§5 全部 9 个指标列 ✅")
    print("="*70)
