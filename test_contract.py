"""
test_contract.py — 接口契约验证脚本
--------------------------------------
每个人更新自己的模块后，在项目根目录运行：

    python3 test_contract.py

全部绿 ✅ 才算接口兼容，可以交给下游。
本脚本不依赖 pytest，直接 python3 跑即可。

覆盖范围：
  - [types]  types_def.py 共享类型可正常导入
  - [B]      SocialAgent 接口（__init__ / step / beliefs / pending_action）
  - [C]      HawkesEngine 接口（intensity / add_event / sample_next_time）
  - [A]      OpinionModel 核心流程（3-agent × 3-step 全链路）
  - [A]      submit_action 写入缓存
"""

from __future__ import annotations
import sys
import traceback

# ─────────────────────────────────────────────────────────────────── #
# 迷你测试框架（不依赖 pytest）
# ─────────────────────────────────────────────────────────────────── #

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
# [types]  共享类型层
# ═══════════════════════════════════════════════════════════════════ #

def _test_types_import():
    from types_def import (
        SimConfig, AgentType, ActionType, ActionRecord,
        BeliefSystem, Perception, MemoryRecord, EmotionState,
        Desire, Intention, SocialInfo, OpinionBelief,
    )
    cfg = SimConfig()
    assert cfg.n_agents > 0, "SimConfig 默认值异常"
    # 枚举值检查
    assert AgentType.PUBLIC == 0
    assert ActionType.SILENT == 4


# ═══════════════════════════════════════════════════════════════════ #
# [C]  HawkesEngine 接口
# ═══════════════════════════════════════════════════════════════════ #

def _test_hawkes_init():
    """mu / beta 必须 > 0，否则抛 ValueError。"""
    from hawkes_engine import HawkesEngine
    h = HawkesEngine(mu=0.1, alpha=0.5, beta=1.0)
    assert h.history == [], "初始 history 应为空列表"
    # 参数校验
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
    # 无事件时 intensity == mu
    lam0 = h.intensity(0.0)
    assert isinstance(lam0, float),    "intensity 返回值必须是 float"
    assert abs(lam0 - 0.2) < 1e-9,    f"无事件时 intensity 应等于 mu=0.2，实际={lam0}"
    # add_event 后 intensity 应升高
    h.add_event(1.0)
    lam_after = h.intensity(1.5)
    assert lam_after > 0.2,            "add_event 后 intensity 应 > mu"
    # 远处时刻应接近 mu（激励衰减）
    lam_far = h.intensity(100.0)
    assert lam_far < lam_after,        "远处时刻 intensity 应衰减"


def _test_hawkes_sample():
    from hawkes_engine import HawkesEngine
    h = HawkesEngine(mu=0.5, alpha=0.3, beta=1.0)
    t_next = h.sample_next_time(0.0)
    assert isinstance(t_next, float),  "sample_next_time 返回值必须是 float"
    assert t_next > 0.0,               f"sample_next_time 返回值 {t_next} 应 > current_t=0"


def _test_hawkes_add_event():
    from hawkes_engine import HawkesEngine
    h = HawkesEngine(mu=0.1, alpha=0.5, beta=1.0)
    h.add_event(3.0)
    h.add_event(1.0)
    h.add_event(2.0)
    assert h.history == sorted(h.history), "history 应保持升序"


# ═══════════════════════════════════════════════════════════════════ #
# [B]  SocialAgent 接口
# ═══════════════════════════════════════════════════════════════════ #

def _make_single_model():
    """辅助：创建最小 OpinionModel（n=1）供 Agent 测试使用。"""
    from types_def import SimConfig
    from opinion_model import OpinionModel
    return OpinionModel(SimConfig(
        n_agents=1,
        hawkes_params={"mu": 0.5, "alpha": 0.3, "beta": 1.0},
    ))


def _test_agent_attributes():
    """SocialAgent 必须暴露 beliefs / pending_action 属性。"""
    model = _make_single_model()
    agent = model.schedule.agents[0]
    assert hasattr(agent, "beliefs"),         "缺少 beliefs 属性"
    assert hasattr(agent.beliefs, "opinions"), "缺少 beliefs.opinions（Dict[str, OpinionBelief]）"
    assert hasattr(agent.beliefs, "emotion"),  "缺少 beliefs.emotion（EmotionState）"
    assert hasattr(agent.beliefs, "psychology"),"缺少 beliefs.psychology"
    assert hasattr(agent, "pending_action"),   "缺少 pending_action 属性"


def _test_agent_step_no_crash():
    """step() 不能抛出任何异常（异常应在内部降级处理）。"""
    model = _make_single_model()
    agent = model.schedule.agents[0]
    for _ in range(5):
        agent.step()   # 多步连续调用也应稳定


def _test_agent_opinion_range():
    """opinion_value 必须保持在 [-1, 1]。"""
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
        assert -1.0 <= e.valence <= 1.0, f"valence={e.valence} 超出范围"
        assert  0.0 <= e.arousal <= 1.0, f"arousal={e.arousal} 超出范围"


# ═══════════════════════════════════════════════════════════════════ #
# [A]  OpinionModel 核心流程
# ═══════════════════════════════════════════════════════════════════ #

def _test_model_datacollector_format():
    """DataCollector 格式必须与 E 模块约定完全一致。"""
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
    # 行数 == 步数
    assert len(df) == N_STEPS, f"DataFrame 行数 {len(df)} ≠ 步数 {N_STEPS}"
    # 必需列
    required = {"avg_opinion", "polarization", "emotional_contagion"}
    missing = required - set(df.columns)
    assert not missing, f"缺少必需列：{missing}"
    # 无 NaN
    assert df.isnull().sum().sum() == 0, "DataCollector 存在 NaN"
    # avg_opinion 在合法范围
    assert (df["avg_opinion"].abs() <= 1.0).all(), "avg_opinion 超出 [-1,1]"
    # polarization >= 0
    assert (df["polarization"] >= 0).all(), "polarization 出现负值"


def _test_model_submit_action():
    """submit_action 应正确写入 info_stream_cache。"""
    from types_def import SimConfig, ActionRecord, ActionType
    from opinion_model import OpinionModel
    model = OpinionModel(SimConfig(
        n_agents=1,
        hawkes_params={"mu": 0.5, "alpha": 0.3, "beta": 1.0},
    ))
    before = len(model.info_stream_cache)
    r = ActionRecord(
        agent_id=0, action_type=ActionType.POST,
        content="test", event_id="E001",
        opinion_value=0.5, tick=0,
    )
    model.submit_action(r)
    assert len(model.info_stream_cache) == before + 1, \
        "submit_action 后缓存长度应 +1"


def _test_model_single_node():
    """n=1 单节点不能崩溃（边界值测试）。"""
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


# ═══════════════════════════════════════════════════════════════════ #
# 执行所有测试
# ═══════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  接口契约验证  (test_contract.py)")
    print("═" * 55)

    print("\n── [types]  types_def.py 共享类型")
    _check("types_def 全量导入 + 枚举值正确", _test_types_import)

    print("\n── [C]  HawkesEngine 接口")
    _check("__init__ 含参数校验", _test_hawkes_init)
    _check("intensity(t) 返回 float，无事件=mu", _test_hawkes_intensity)
    _check("sample_next_time > current_t", _test_hawkes_sample)
    _check("add_event 保持 history 升序", _test_hawkes_add_event)

    print("\n── [B]  SocialAgent 接口")
    _check("beliefs / pending_action 属性存在", _test_agent_attributes)
    _check("step() 连续5次不崩溃", _test_agent_step_no_crash)
    _check("20步后 opinion/emotion 值域 [-1,1]", _test_agent_opinion_range)

    print("\n── [A]  OpinionModel 核心流程")
    _check("DataCollector 列名 / 行数 / 无NaN", _test_model_datacollector_format)
    _check("submit_action 写入 info_stream_cache", _test_model_submit_action)
    _check("n=1 单节点 5步不崩溃", _test_model_single_node)

    # ── 汇总 ─────────────────────────────────────────────────────── #
    n_pass = sum(1 for r in _results if r[0] == "✅")
    n_fail = sum(1 for r in _results if r[0] == "❌")
    total  = len(_results)

    print("\n" + "─" * 55)
    print(f"  通过 {n_pass}/{total}    失败 {n_fail}/{total}")
    if n_fail == 0:
        print("  ✅  全部通过，接口兼容，可交付")
    else:
        fails = [r[1] for r in _results if r[0] == "❌"]
        print(f"  ❌  失败项：{fails}")
        print("  请修复后再交付给下游成员")
    print("─" * 55 + "\n")

    sys.exit(0 if n_fail == 0 else 1)
