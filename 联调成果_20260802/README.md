# 联调成果 · 2026-08-02

本目录为 SURF-0245「多智能体舆情治理仿真」**本周联调成果快照**，对应 GitHub 仓库分支 `联调成果`。

- 仓库：https://github.com/JiaxinChen-Annie/SURF-0245-Multi-Agent-Social-Simulation-for-Public-Opinion-Governance
- 目标分支：`联调成果`
- 建议远程路径：`联调成果_20260802/`（本文件夹整体上传）

---

## 目录结构

```
联调成果_20260802/
├── README.md                 # 本说明（总描述）
├── 本周工作纪要.md            # ABCDE 分工 + 本周优化（可直接贴进周报）
├── 联调代码/                 # 本周联调后的完整可运行代码快照
└── 联调结果/                 # 本周联调生成的运行结果
    ├── mock_100x50/          # 规则 mock：100 智能体 × 50 步（全流程冒烟）
    ├── bench_V1_10x5/        # DeepSeek 小规模验收：10×5（FORWARD 修复后）
    └── ds_100x20/            # DeepSeek 主跑：100×20（联调主结果）
```

### 为什么同时上传「联调代码」和「联调结果」

本周联调不是简单拼接：A/B/C/D/E 接口、记忆、LLM 预算、转发规则、治理节拍、报表字段均有改动。  
只上传结果、不上传代码，后人无法复现；只上传代码、不上传结果，无法对照指标。因此本包 **代码 + 结果一并归档**。

---

## 本周联调做了什么（总览）

1. **A 底座合入联调**：负面阈值干预、MUTE/ANNOUNCE、分层转发门控、GROUP_EXPOSURE、SimClock（`tick_seconds` / `place` / `time_slot`）、`governance_tick` 与 `intervention_tick` 分工。
2. **B 记忆脑移植**：在不破坏 A 枚举 / C LLM 接线的前提下，并入短期记忆 + 长期摘要 + 高冲击记忆 + 信任微衰减；`env_info` 增加 `memory_summary` / `trust`。
3. **C LLM 提速与转发修复**：提示词预算、并发、`user_id`、json_mode、思考链关闭；修复 FORWARD 被 ANNOUNCE 淹没的问题，恢复跨组转发。
4. **D 评估对齐**：`governance_tick`、滚动窗口、GroupType/AgentType 切片、`time_profile`；适配器与集成脚本合入。
5. **E 报表增强**：宽表 `metrics.csv` 可画；新增按 AgentType / GroupType 的分层图（fig7/fig8）；DataCollector 增加分组均值列。

### 相对上周的主要优化

| 维度 | 上周（对照 run_meta） | 本周 |
|------|----------------------|------|
| 验收 `all_pass` | false | true（主跑 100×20） |
| `cross_group_forward` | ~29 | ~1194（主跑） |
| 通道可观测性 | 较薄 | 组内 / 上向 / 横向 / 下向分流可统计 |
| 记忆 | 偏弱 / 未与 A 统一 | A 兼容的 B 记忆系统 |
| LLM | 易超时、提示过长 | 预算截断 + 并发 + 稳定 JSON |
| 报表 | 宽表难出图 | fig7/fig8 + 分组列 |

主跑粗算：100 智能体 × 20 步，DeepSeek，约 44 分钟（~132s/step），消息与转发明显更活跃。

---

## 联调代码说明

`联调代码/` 建议从本机工作区拷贝（保持可运行结构）：

- 来源建议：`E:\SURF\SURF C\联调_ABCDE\` 全量代码快照  
  （含 `opinion_model.py`、`social_agent.py`、`hawkes_engine.py`、`llm_utils.py`、`strategy_eval.py`、`plot_run_report.py`、configs、scripts 等）
- 运行环境：Conda `dts209tc`；真 LLM 需设置 `DEEPSEEK_API_KEY`
- 主配置参考：`config_100_ds.yaml`（或包内同名配置）

> 上传前请自行确认：勿把含密钥的 `.env`、本地绝对路径笔记误传；API Key 只放环境变量。

---

## 联调结果说明

| 子目录 | 含义 | 关注文件 |
|--------|------|----------|
| `mock_100x50/` | 无 LLM 全流程冒烟 | `metrics.csv`、`run_meta.json`、`figures/` |
| `bench_V1_10x5/` | 小规模真 LLM 验收（FORWARD 修复后） | `run_meta.json`、指标与验收字段 |
| `ds_100x20/` | **本周主结果** | `run_meta.json`、`metrics.csv`、`figures/`（含 fig7/fig8） |

解读时优先看 `run_meta.json` 中的：`all_pass`、`cross_group_forward`、通道分流、announce/mute 计数、耗时。

---

## 模块边界（简记）

| 模块 | 职责 |
|------|------|
| A | 仿真骨架、事件、治理动作、社交拓扑与转发规则、时钟 |
| B | 智能体记忆 / 信任 / 感知与决策侧状态 |
| C | Hawkes + LLM 调用与动作解析 |
| D | 策略评估、窗口与切片指标 |
| E | 报表与可视化 |

更细的「本周 A/B/C/D/E 分别干了什么」见同目录 **`本周工作纪要.md`**（可直接复制进周报）。

---

## 复现（摘要）

```bat
conda activate dts209tc
cd 联调代码
set DEEPSEEK_API_KEY=你的密钥
python run_sim.py --config configs/你的配置.yaml
python plot_run_report.py --run-dir 输出目录
```

具体入口与配置名以 `联调代码/` 内实际文件为准。

---

## 提交信息建议

- **标题**：`联调成果 2026-08-02：ABCDE 合入、LLM 转发修复与 100×20 主跑归档`
- **描述**：见下方「上传用 commit message」或 `本周工作纪要.md` 首段。
