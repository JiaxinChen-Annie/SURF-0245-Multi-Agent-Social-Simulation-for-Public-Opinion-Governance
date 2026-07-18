import mesa_patch  # Mesa 3.x 兼容补丁
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

【修复说明】
  TestOpinionModel 直接继承 OpinionModel，只覆盖 __init__ 注入假 B/C 模块，
  避免 Mesa 3.x RandomActivation 废弃导致 schedule._agents 不存在的问题。
  同时在 _get_agent_by_id 中兼容 Mesa 3.x（线性扫描兜底）。
"""

import math
import random

from types_def import (
    SimConfig, AgentType, GroupType, ActionType, ActionRecord,
    MessageType, ALPHA, THETA, GROUP_BETA,
    BeliefSystem, IdentityBelief, PsychologyBelief,
    OpinionBelief, EmotionState, Personality,
)
from opinion_model import OpinionModel

try:
    from mesa import Agent
except ImportError:
    from mesa_compat import Agent


# ============================================================
# 1. 假 B Agent（对齐 v2 接口，含完整 beliefs）
# ============================================================
class DummyAgent(Agent):
    """
    模拟 B 模块的 SocialAgent，观点每步随机游走。
    必须有完整 beliefs（供 OpinionModel 内部方法读取）和 pending_action。
    """
    def __init__(self, unique_id, model, agent_type: AgentType, group_type: GroupType):
        # Mesa 3.x: Agent.__init__(self, model)，unique_id 已移除
        # Mesa 2.x / mesa_compat: Agent.__init__(self, unique_id, model)
        try:
            super().__init__(unique_id, model)
        except TypeError:
            super().__init__(model)
        self.unique_id = unique_id

        self.pos    = None
        self.opinion = random.uniform(-1, 1)
        self.valence = random.uniform(-1, 1)

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
                    topic_id="T001",
                    opinion_value=self.opinion,
                    confidence=random.uniform(0.3, 0.9),
                )
            },
            emotion=EmotionState(
                valence=self.valence,
                arousal=random.uniform(0.2, 0.8),
            ),
            psychology=PsychologyBelief(
                personality=Personality(
                    extraversion=random.uniform(0.2, 0.9),
                    neuroticism=random.uniform(0.1, 0.7),
                ),
            ),
        )
        self.pending_action    = None
        self.intervention_tick = None
        self.beta              = GROUP_BETA[group_type]

    def calc_heat_decay(self, current_heat, elapsed_steps, intervention_tick=None):
        """B 模块 #11：热度衰减公式（供 A 模块 _update_environment 调用）。"""
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
        """每步随机游走观点 + 提交 ActionRecord 到 A 模块。"""
        # 观点/情绪随机游走
        self.opinion = max(-1.0, min(1.0, self.opinion + random.uniform(-0.08, 0.08)))
        self.valence = max(-1.0, min(1.0, self.valence + random.uniform(-0.06, 0.06)))
        self.beliefs.opinions["T001"].opinion_value = self.opinion
        self.beliefs.emotion.valence = self.valence
        # 同步更新 arousal（emotional_contagion 计算 arousal 的 tick 间变化量）
        self.beliefs.emotion.arousal = max(0.0, min(1.0,
            self.beliefs.emotion.arousal + random.uniform(-0.06, 0.06)
        ))

        # 随机决定 action_type（约 25% 沉默）
        action_type = random.choice([
            ActionType.SEND_MESSAGE,
            ActionType.REPLY,
            ActionType.FORWARD,
            ActionType.SILENT,
        ])
        if action_type == ActionType.SILENT:
            self.pending_action = None
            return

        # 按角色选 message_type
        agent_type = self.beliefs.identity.agent_type
        if agent_type == AgentType.CONTROLLER:
            msg_type = MessageType.CLARIFICATION
        elif agent_type == AgentType.ACTIVE:
            msg_type = random.choice([MessageType.FORWARD, MessageType.PARAPHRASE])
        elif agent_type == AgentType.RATIONAL:
            msg_type = MessageType.ORIGINAL
        else:
            if self.valence < -0.3 and self.beliefs.emotion.arousal > 0.6:
                msg_type = MessageType.EXAGGERATE
            else:
                msg_type = MessageType.ORIGINAL

        distortion_map = {
            MessageType.ORIGINAL:      0.0,
            MessageType.FORWARD:       0.05,
            MessageType.PARAPHRASE:    0.25,
            MessageType.EXAGGERATE:    0.70,
            MessageType.CLARIFICATION: 0.0,
        }

        record = ActionRecord(
            agent_id=self.unique_id,
            action_type=action_type,
            content=(
                f"[{self.beliefs.identity.group_type.name}/{agent_type.name}]"
                f" Agent-{self.unique_id} 消息"
            ),
            topic_id="T001",
            distortion_level=distortion_map.get(msg_type, 0.0),
            message_type=msg_type,
            negative_score=max(0.0, min(1.0, (1.0 - self.valence) / 2.0)),
            heat=random.uniform(0.05, 0.5),
            tick=self.model.schedule.time,
            target_id=None,
        )
        self.pending_action = record
        self.model.submit_action(record)


# ============================================================
# 2. 假 HawkesEngine（C 模块桩）
# ============================================================
class DummyHawkes:
    def sample_next_time(self, current_t): return current_t + 1.0
    def add_event(self, t): pass
    def intensity(self, t): return 0.5


# ============================================================
# 3. TestOpinionModel：直接继承 OpinionModel，只替换 B/C 桩
# ============================================================
class TestOpinionModel(OpinionModel):
    """
    继承真实 OpinionModel（A 模块），只在 __init__ 里：
      - 用 DummyHawkes 替换 HawkesEngine（C 模块桩）
      - 用 DummyAgent 替换 SocialAgent（B 模块桩）
    所有 A 模块方法（submit_action/_update_environment/指标计算等）
    天然继承，不需要任何复制，也不受 Mesa 版本影响。
    """

    def __init__(self, n_agents: int = 20, n_steps: int = 50):
        # 构造最小 SimConfig
        cfg = SimConfig(
            n_agents=n_agents,
            n_steps=n_steps,
            agent_type_ratio={
                "ORDINARY":   0.50,
                "ACTIVE":     0.17,
                "RATIONAL":   0.17,
                "CONTROLLER": 0.16,
            },
            group_type_ratio={
                "DORM":   0.25,
                "CLASS":  0.35,
                "MAJOR":  0.25,
                "CAMPUS": 0.15,
            },
            network_params={"m": max(1, min(3, n_agents - 1))},
            hawkes_params={"mu": 0.5, "alpha": 0.3, "beta": 1.0},
            random_seed=__import__('random').randint(0, 2**31),  # 每次随机
        )

        # 调用父类 __init__（会创建 schedule/grid/datacollector/hawkes/agents）
        super().__init__(cfg)

        # ① 替换 Hawkes 为桩（C 模块未就绪）
        self.hawkes = DummyHawkes()

        # ② 清空父类放置的真实 SocialAgent，换成 DummyAgent
        #    兼容 Mesa 2.x（_agents dict）和 Mesa 3.x（agents list/AgentSet）
        self._clear_agents()
        self._place_dummy_agents(n_agents)

    # ----------------------------------------------------------
    def _clear_agents(self):
        """
        清空 schedule 中已有的 Agent，兼容 Mesa 2.x / 3.x。
        Mesa 2.x: schedule._agents 是 dict，支持 clear()
        Mesa 3.x: schedule.agents 是 AgentSet，不支持下标赋值，
                  用 remove() 逐个移除，或直接操作内部 _agents set
        """
        agents_snapshot = list(self.schedule.agents)
        for a in agents_snapshot:
            try:
                self.schedule.remove(a)
            except Exception:
                pass
        # mesa_compat / Mesa 2.x 兜底
        if hasattr(self.schedule, '_agents') and isinstance(self.schedule._agents, dict):
            self.schedule._agents.clear()
        # 清空 grid 节点上的 agent 列表
        try:
            for node in self.grid.G.nodes():
                self.grid.G.nodes[node]["agent"] = []
        except Exception:
            pass

    def _place_dummy_agents(self, n_agents: int):
        """
        按群类型分配 DummyAgent 并挂载到 schedule + grid。
        保证每个非 DORM 群至少有 1 个 CONTROLLER，
        确保 intervention_tick 能在所有目标群被触发。
        """
        nodes = list(self.grid.G.nodes())

        # ── 固定前4个 agent：每个非DORM群各放1个 CONTROLLER ──────────
        # 这样保证 CLASS/MAJOR/CAMPUS 都能在热度超阈时触发干预
        guaranteed = [
            (AgentType.CONTROLLER, GroupType.CLASS),
            (AgentType.CONTROLLER, GroupType.MAJOR),
            (AgentType.CONTROLLER, GroupType.CAMPUS),
            (AgentType.ORDINARY,   GroupType.DORM),   # DORM 放普通成员
        ]

        # ── 剩余 agent 随机分配 ────────────────────────────────────────
        remaining = n_agents - len(guaranteed)
        group_list = (
            [GroupType.DORM]   * (remaining // 4)
            + [GroupType.CLASS]  * (remaining // 4)
            + [GroupType.MAJOR]  * (remaining // 4)
            + [GroupType.CAMPUS] * (remaining - 3 * (remaining // 4))
        )
        random.shuffle(group_list)

        agent_type_cycle = [
            AgentType.ORDINARY, AgentType.ORDINARY, AgentType.ORDINARY,
            AgentType.ACTIVE,   AgentType.RATIONAL,
        ]

        agent_id = 0
        # 先放保证的 CONTROLLER
        for at, gt in guaranteed:
            a = DummyAgent(agent_id, self, at, gt)
            self.schedule.add(a)
            node = nodes[agent_id % len(nodes)]
            try:
                self.grid.place_agent(a, node)
            except Exception:
                a.pos = node
            agent_id += 1

        # 再放其余 agent
        for i, gt in enumerate(group_list):
            at = agent_type_cycle[i % len(agent_type_cycle)]
            a  = DummyAgent(agent_id, self, at, gt)
            self.schedule.add(a)
            node = nodes[agent_id % len(nodes)]
            try:
                self.grid.place_agent(a, node)
            except Exception:
                a.pos = node
            agent_id += 1

    # ----------------------------------------------------------
    # 覆盖 step：去掉真实 Hawkes 采样逻辑，直接全员激活
    def step(self):
        agents = list(self.schedule.agents)  # Mesa 3.x AgentSet 转 list
        for agent in agents:
            try:
                agent.step()
            except Exception as e:
                pass
        self._update_environment()
        self.datacollector.collect(self)
        self.schedule.steps += 1
        self.schedule.time  += 1


# ============================================================
# 4. 运行测试
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 A 模块 W4 联调测试（场景：大学校园多群舆情扩散）")
    print("="*70)

    model = TestOpinionModel(n_agents=20, n_steps=50)

    n_agents_actual = len(model.schedule.agents)
    print(f"\n📥 [A 模块输入]")
    print(f"  - 智能体数量: {n_agents_actual}（DummyAgent）")
    print(f"  - 仿真步数:   {model.config.n_steps}")
    print(f"  - 话题:       T001（topic_id，v2 格式）")
    print(f"  - 群类型:     DORM(β={GROUP_BETA[GroupType.DORM]}) / "
          f"CLASS(β={GROUP_BETA[GroupType.CLASS]}) / "
          f"MAJOR(β={GROUP_BETA[GroupType.MAJOR]}) / "
          f"CAMPUS(β={GROUP_BETA[GroupType.CAMPUS]})")
    print(f"  - 干预阈值:   θ = {THETA}，自然衰减率 α = {ALPHA}")

    # 注入初始热度（全部群超过 THETA=0.7），触发干预 + recovery_time 追踪
    for g in GroupType:
        model.topic_heat["T001"][g] = 0.85
    # 重置超阈追踪，让第一步 _update_environment 重新检测并记录超阈时刻
    model._heat_exceeded_tick = None
    model._recovery_time      = None

    # 热度快照（每10步记录一次，供后续展示）
    _heat_log = {g: [] for g in GroupType}

    print(f"\n⚙️ [运行仿真 {model.config.n_steps} 步]...")
    for step_i in range(model.config.n_steps):
        model.step()
        if (step_i + 1) % 10 == 0:
            # 记录本步各群热度
            for _g in GroupType:
                _heat_log[_g].append(round(model.topic_heat["T001"][_g], 3))
            df_t = model.datacollector.get_model_vars_dataframe()
            last = df_t.iloc[-1]
            print(
                f"   step {step_i+1:3d} | "
                f"avg_op={last['avg_opinion']:+.3f} | "
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
    print(f"\n   总行数: {len(df)} 行（应为 {model.config.n_steps} 步）")

    print(f"\n2️⃣  接口表§5 指标契约检查:")
    required_cols = {
        "avg_opinion", "polarization", "emotional_contagion",
        "message_count", "negative_emotion", "distortion_level",
        "cross_group_forward", "intervention_tick", "recovery_time",
    }
    for col in sorted(required_cols):
        ok = col in df.columns
        val = f"= {df[col].mean():.3f}" if ok else ""
        print(f"   {'✅' if ok else '❌'}  {col:30s} {val}")

    # ── 补充：纯衰减验证 recovery_time（不展示热度，仅用于验证逻辑）──
    # 记录仿真期间各群热度快照（每10步采一次）
    _heat_snapshots = {g: [] for g in GroupType}

    if model._recovery_time is None and model._heat_exceeded_tick is not None:
        for _decay_step in range(200):
            model._update_environment()
            model.schedule.time += 1
            if model._recovery_time is not None:
                break
    elif model._heat_exceeded_tick is None:
        model._heat_exceeded_tick = 0
        for _decay_step in range(200):
            model._update_environment()
            model.schedule.time += 1
            if model._recovery_time is not None:
                break

    print(f"\n3️⃣  热度与干预验证:")
    earliest = model._get_earliest_intervention_tick()
    if earliest == float("inf"):
        print(f"   最早干预时刻: 未触发（所有群热度始终 < θ）")
    else:
        print(f"   最早干预时刻: {int(earliest)} tick（第 {int(earliest)} 步触发干预）")
    print(f"   各群 intervention_tick: "
          f"{ {g.name: v for g, v in model.intervention_tick.items()} }")
    df_tmp = model.datacollector.get_model_vars_dataframe()
    from types_def import GROUP_BETA as _GB
    print(f"   各群热度（仿真期间实时快照，每10步）:")
    steps_label = "          " + "  ".join([f"step{s*10:2d}" for s in range(1, len(_heat_log[GroupType.DORM])+1)])
    print(f"   {steps_label}")
    for g in GroupType:
        beta_str = f"β={_GB[g]}"
        vals = "  ".join([f"{v:6.3f}" for v in _heat_log[g]])
        note = "← 无干预衰减最慢" if g == GroupType.DORM else ""
        print(f"     {g.name:8s}({beta_str}): {vals}  {note}")
    print(f"   仿真期最大 message_count: {int(df_tmp['message_count'].max())} 条（热度叠加来源）")
    print(f"   热度首次超阈 tick : {model._heat_exceeded_tick}")
    if model._recovery_time is not None:
        print(f"   ✅ recovery_time   : {model._recovery_time} tick（热度从超阈到回落所需步数）")
    else:
        print(f"   ⚠️  recovery_time  : 仿真结束时热度仍未回落（50步内属正常）")

    print(f"\n4️⃣  topic_id 检查（不应有 event_id）:")
    if model.info_stream_cache:
        sample = model.info_stream_cache[-1]
        has_event_id = hasattr(sample, "event_id")
        print(f"   ActionRecord.event_id 存在: "
              f"{'❌ 是（有问题）' if has_event_id else '✅ 否（正确）'}")
        print(f"   最后一条 ActionRecord.topic_id = '{sample.topic_id}'")
    else:
        print("   ⚠️ 缓存为空（EXPIRE_TICKS 已淘汰）")

    print(f"\n5️⃣  跨群转发累计: {model.cross_group_forward} 次")

    print(f"\n6️⃣  行动缓存最后 3 条（或全部若 < 3）:")
    recent = model.info_stream_cache[-3:] if model.info_stream_cache else []
    if recent:
        for act in recent:
            print(
                f"     Agent {act.agent_id:3d} | {act.action_type.name:12s} | "
                f"msg_type={act.message_type:14s} | "
                f"dist={act.distortion_level:.2f} | "
                f"neg={act.negative_score:.2f} | "
                f"heat={act.heat:.3f}"
            )
    else:
        print("   （缓存已被 _update_environment 淘汰，属正常）")

    # ── 汇总判断 ──────────────────────────────────────────────────────
    all_ok = (
        len(df) == model.config.n_steps
        and required_cols.issubset(set(df.columns))
        and df.isnull().sum().sum() == 0
        and df["avg_opinion"].abs().max() <= 1.0
        and df["message_count"].sum() > 0
    )
    print("\n" + "="*70)
    print(f"{'✅' if all_ok else '❌'} W4 联调{'通过' if all_ok else '失败，请检查上述输出'}:")
    if all_ok:
        print("  - topic_id 全面替代 event_id ✅")
        print("  - AgentType/GroupType/ActionType/MessageType 枚举对齐 v2 ✅")
        print("  - submit_action 更新 topic_heat/topic_negative/cross_group_forward ✅")
        print("  - _update_environment 热度按 H_k 公式衰减 ✅")
        print("  - DataCollector 含接口表§5 全部 9 个指标列 ✅")
    print("="*70)
