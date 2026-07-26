# 联调_ABCDE — 全量接入说明（本机）

## 用哪些文件（已按此组装本目录）

| 来源 | 用哪个 | 不用哪个 |
|------|--------|----------|
| **A** | `opinion_model.py` `types_def.py` `run_sim.py` `mesa_*.py` `config*.yaml` | — |
| **B** | **使用 A 包内已对齐的 `social_agent.py`**（读 `model._llm_client`） | B 仓库单独那份若仍是 `model.llm_utils.get_client()`，先别直接覆盖 |
| **C** | `hawkes_engine.py` `llm_utils.py`（本目录已用你的交付代码覆盖） | A 自带旧 hawkes/llm 已覆盖 |
| **E** | `utils.py`（save / visualize / logger） | 不要用 E 的 LLM 覆盖 C |
| **D** | `strategy_eval.py` `abc_model_adapter.py` `run_d_to_abc_integration.py` | 仿真热路径外离线跑 |

## 推荐跑法（顺序）

### 1）先 mock 压力（必做）

```bat
conda activate dts209tc
cd /d "E:\SURF\SURF C\联调_ABCDE"
python run_sim.py --config config_100_mock.yaml
python plot_run_report.py
```

### 2）再真 Key（100 agents，建议 20 步）

```bat
set DEEPSEEK_API_KEY=你的key
python run_sim.py --config config_100_ds.yaml
python plot_run_report.py
```

为何 **20 步** 而不是 50：每 tick 约激活 10%~80% agent，全真 LLM 时 100×50 可能极慢/很贵。20 步够看趋势与稳定性；mock 用 50 步做压力。

### 3）D 评估（可选，离线）

```bat
python run_d_to_abc_integration.py --agents 100 --steps 20 --seed 42 --out d_results
```

## 输出

- `output/metrics.csv` + `run_meta.json`（运行时间等）
- `output/figures/fig1_metrics_trends.png` …
- `output/simulation_trends_e.png`（E 原版）
- `output/simulation_trends_w4.png`（A 原版）
