# 联调_ABCDE — 全量接入说明（本机）

更新：2026-07-30 已覆盖 **A 模块 v3**（见 `交接说明_A模块_v3.md` / `SESSION_HANDOFF.md`）。

## 用哪些文件

| 来源 | 用哪个 |
|------|--------|
| **A v3** | `opinion_model.py` `types_def.py` `social_agent.py` `event_source.py` `mesa_*.py` `config.yaml` |
| **C** | `hawkes_engine.py`（交付代码）+ `llm_utils.py`（A 底 + MUTE/ANNOUNCE） |
| **D** | `strategy_eval.py` `abc_model_adapter.py` `run_d_to_abc_integration.py` |
| **E** | `utils.py` |
| 联调 | `run_sim.py`（含 `step_seconds`/`run_meta`）`plot_run_report.py` `config_100_*.yaml` |

## v3 关键点

- 负面超阈触发干预；`governance_tick` vs `intervention_tick`
- 事件源独立；四条转发通道 + 方向门槛
- MUTE / ANNOUNCE；群曝光 `GROUP_EXPOSURE`

## 推荐跑法

```bat
conda activate dts209tc
cd /d "E:\SURF\SURF C\联调_ABCDE"

python test_contract.py
python run_sim.py --config config_100_mock.yaml
python plot_run_report.py

set DEEPSEEK_API_KEY=你的key
python run_sim.py --config config_100_ds.yaml
```

## 输出

`output/metrics.csv`（含 `step_seconds`、`mute_count`、方向转发列）、`run_meta.json`、`figures/`
