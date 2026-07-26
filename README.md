# 实验结果文件说明

本目录是 **ABCDE 模块联调压力测试** 的结果包（不含 API Key、不含完整仿真源码），方便写汇报的同学和各模块组员直接取用。

实验对比两条路径：


| 实验       | 含义                       | 规模                    | 对应目录                  |
| -------- | ------------------------ | --------------------- | --------------------- |
| Mock     | 规则 / MockLLM，快速压力与骨架验收   | 100 agents × 50 steps | `output_mock_100x50/` |
| DeepSeek | 真实 deepseek-v4-flash API | 100 agents × 20 steps | `output_ds_100x20/`   |


随机种子均为 **42**，前 20 步指标可逐点对照。

---

## 1. 包里有什么

```text
联调成果_上传包/
├── README.md                          ← 本说明（先看这个）
├── 实验报告_Mock与DeepSeek对比.md      ← 正式文字报告（目的/设置/结果/分析）
├── README_INTEGRATION.md              ← 联调怎么跑（复现参考，本包无源码）
├── config_100_mock.yaml               ← Mock 实验配置
├── config_100_ds.yaml                 ← DeepSeek 实验配置
├── output_mock_100x50/
│   ├── metrics.csv                    ← 逐步指标主表
│   ├── run_meta.json                  ← 运行元信息（耗时、模型、末态摘要）
│   ├── step_times.csv                 ← 每步秒数
│   └── figures/                       ← 出图 fig1–fig6
└── output_ds_100x20/
    ├── metrics.csv
    ├── run_meta.json
    ├── step_times.csv
    └── figures/                       ← 出图 fig1–fig6
```

各目录 `figures/` 中图含义：


| 文件                             | 内容                     |
| ------------------------------ | ---------------------- |
| `fig1_bounded_trends.png`      | 有界指标趋势（观点、极化、负面情绪、失真等） |
| `fig2_count_trends.png`        | 计数类趋势（消息量、跨群累计等）       |
| `fig3_runtime_bars.png`        | 运行耗时柱状图                |
| `fig4_metric_panels.png`       | 多指标分面总览                |
| `fig5_emotional_contagion.png` | 情绪传染强度                 |
| `fig6_step_seconds.png`        | 每 tick 耗时折线            |


---

## 2. 内容主要看哪些文件（按优先级）

1. `**实验报告_Mock与DeepSeek对比.md**` — 实验思路、设置、结果表、分析结论（文字主源）
2. **两边的 `figures/`** — 直接贴进 PPT / 文档的图
3. **两边的 `run_meta.json`** — 总耗时、模型名、末态指标一眼摘要
4. **两边的 `metrics.csv`** — 需要查某一步数值、自己画图或核对报告时再用
5. `**config_*.yaml**` — 确认实验规模与超参（可选）
6. `**README_INTEGRATION.md**` — 想知道当时怎么跑的（本包不能直接复现源码）

一般 **不必** 先啃 `step_times.csv`；做性能对比时再看即可。

---

## 3. 写汇报的人看什么

**主线：报告 + 图**


| 用途            | 打开                                                                    |
| ------------- | --------------------------------------------------------------------- |
| 写背景 / 思路 / 结论 | `实验报告_Mock与DeepSeek对比.md`（第 1–2、7–9 节）                                |
| 写实验设置         | 报告第 3 节 + `config_100_mock.yaml` / `config_100_ds.yaml`               |
| 写结果（表）        | 报告第 5 节；细数查 `metrics.csv` / `run_meta.json`                           |
| 贴图            | `output_mock_100x50/figures/` 与 `output_ds_100x20/figures/` 对照选用      |
| 强调性能          | `run_meta.json` 里的 `run_seconds`、`seconds_per_step` + `fig3` / `fig6` |


建议汇报结构：联调目的 → Mock vs DeepSeek 设置差异 → 关键曲线/末态对比 → 结论与局限（跨群指标语义等报告里已写明）。

---

## 4. 别的组员看什么


| 角色                        | 建议看                                                                     |
| ------------------------- | ----------------------------------------------------------------------- |
| **只想了解联调有没有跑通**           | 报告第 5、8 节 + 任选几张 `figures`                                              |
| **A（调度 / DataCollector）** | `metrics.csv` 列是否齐全；`run_meta.json`；报告里对跨群 / 干预指标语义的说明                  |
| **B（BDI / Agent）**        | 报告中观点、极化、负面情绪：Mock 冻结 vs DeepSeek 有演化                                   |
| **C（Hawkes / LLM）**       | DeepSeek 的 `run_meta.json`（模型、耗时）；`config_100_ds.yaml`；`fig3`/`fig6` 性能 |
| **D（策略评估）**               | 本包无 D 离线评估结果；可先用两边 `metrics.csv` 作输入参考                                  |
| **E（导出 / 可视化）**           | `figures/` 与 `metrics.csv` 是否满足出图需求                                     |
| **想自己复现实验**               | 先读 `README_INTEGRATION.md` + 两个 config；**完整 `.py` 不在本包**，需另取联调代码目录      |


---

## 5. 本包不含什么

- 真实 API Key / `.env`
- 完整联调 Python 源码（`opinion_model.py`、`social_agent.py` 等）

有 Key、有源码的同学若要复跑，按 `README_INTEGRATION.md` 在完整联调目录中执行即可。