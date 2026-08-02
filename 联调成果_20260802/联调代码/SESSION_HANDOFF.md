# 会话交接摘要（SURF C / ABCDE 联调）— 防止上下文截断后断层

更新日：2026-08-02

## 1. 项目与你的职责

- 课题：校园多层群聊多智能体舆情仿真
- **你负责 C**：`HawkesEngine` + LLM 工具链
- 热度公式归 **A**；Hawkes ≠ 群热度 H(t)
- C 正式交付：`E:\SURF\SURF C\交付代码`（GitHub 只传这里）
- DeepSeek 默认：`deepseek-v4-flash`，`thinking: disabled`

## 2. 联调总目录（当前权威）

`E:\SURF\SURF C\联调_ABCDE`

### 2026-07-30 已覆盖（A 模块 v3）

源包：`e:\SURF\download\A`（`交接说明_A模块_v3.md`）

| 文件 | 来源 |
|------|------|
| `opinion_model.py` `types_def.py` `social_agent.py` `event_source.py` `mesa_*` | **A v3** |
| `config.yaml` `config_1000.yaml` `config_single.yaml` | **A v3** |
| `test_contract.py` `test_module_a.py` | **A v3** |
| `llm_utils.py` | **A 底版 + 联调补 MUTE/ANNOUNCE** |
| `hawkes_engine.py` | **C 交付代码** |
| `run_sim.py` | **A v3 + 联调增强**（`step_seconds` / `run_meta.json` / 真 LLM 不套 30s） |
| `config_100_mock.yaml` / `config_100_ds.yaml` | 已写入 v3 `scenario_params` |
| `strategy_eval.py` `abc_model_adapter.py` `run_d_to_abc_integration.py` | **D** |
| `utils.py` | **E**（联调版） |

### v3 关键语义

- 干预默认按**负面超阈**（`intervention_trigger: negative`）；`governance_tick` ≠ `intervention_tick`
- 独立 `event_source.py`（初始事件 + 二次爆料）
- 四条转发通道：群内 / 小→大 / DORM↔CLASS 横向 / 大→小；门槛按方向分级
- Controller 真执行 **MUTE / ANNOUNCE**；指标含 `mute_count`
- `GROUP_EXPOSURE` 影响热度增益、宏观溅射、注意力配额
- 降温统一为「基础降温 × control_level」

## 3. 推荐跑法

```bat
conda activate dts209tc
cd /d "E:\SURF\SURF C\联调_ABCDE"
python test_contract.py
python run_sim.py --config config_100_mock.yaml
python plot_run_report.py
```

真 Key：

```bat
set DEEPSEEK_API_KEY=你的key
python run_sim.py --config config_100_ds.yaml
```

## 4. 注意

- 旧报告（`output_上传包`）基于 v3 之前版本，结论需用新结果重跑后再写
- `交付代码/llm_utils.py` 若要正式交付 C，需把 MUTE/ANNOUNCE、窗口与限速旋钮合回
- Mesa 对齐 3.0.3
- D：消融实验应看 `governance_tick`（真正动手），不要只看 `intervention_tick`
- E：`negative_emotion` / 哨兵指标定义可能需对照 v3 群负面重算

## 5. C 小测（速度 vs 真实感）

先别上 1000。在 Anaconda Prompt：

```bat
conda activate dts209tc
cd /d "E:\SURF\SURF C\联调_ABCDE"
set DEEPSEEK_API_KEY=你的key
python bench_c_llm_variants.py --skip-thinking
```

对比结果在 `output_bench_c/compare.json`。把该文件发我一起判。

**多 Key**：官方并发按**账号**计，同账号多 Key **不加倍**；要更高并发需账号扩容工单或另开账号（费用/合规另议）。1000 agent 优先靠：短窗口 + 客户端 `max_concurrency` + 降激活率/mock 压测骨架。

## 6. 2026-08-02 覆盖决策（A/B 撞车）

| 文件 | 用谁 | 原因 |
|------|------|------|
| `opinion_model.py` `types_def.py` `event_source.py` | **A**（已含 SimClock） | 环境/时钟/时段 |
| `social_agent.py` | **A 底 + C 窗口/user_id** | B 的 `social_agent(8.02).py` 自带枚举，不能整文件盖 |
| `llm_utils.py` `hawkes_engine.py` | **C** | 短窗/FORWARD/限流 |
| `strategy_eval.py` 等 | **D 新版** | governance_tick + 分群分角色评估窗 |
| 出图 | **`plot_run_report.py`**（fig7/8） | 优于 E 的 `generate_e_report.py`；旧 metrics 无分面列会 skip |

B 记忆升级：**已并入**联调 `social_agent.py`（`AgentMemorySystem`：短期+长期摘要+高影响固化+trust 微降）；仍不要用下载包 `social_agent(8.02).py` 整文件覆盖。

## 7. 2026-08-02 联调符合性检查（老师 6 事 + 项目接线）

| 项 | 状态 | 落点 |
|----|------|------|
| C 上下文压缩 / max_tokens / JSON / 限流 / user_id | ✅ | `llm_utils` + `config_100_ds.yaml` |
| A tick→时钟 / place / 时段活跃度 | ✅ | `SimClock` + Perception + Hawkes μ |
| B 短/长期记忆 + 感知分离 + belief 迁移钩子 | ✅ 已并入接线版 | `social_agent.AgentMemorySystem` |
| D governance_tick + 评估窗 + 分群分角色 | ✅ | `strategy_eval.py` 等 |
| E 分群分角色出图 | ✅ | `plot_run_report` fig7/8 + DataCollector 分面列 |
| 流程图闭环 PPT | ⬜ 文档/汇报 | 非代码 |
| 1000 全真 LLM | ⬜ 下一步压测 | 先 mock `config_1000.yaml` |
| 群聊原文导出 | ⬜ 可选 | 尚无 messages.csv |

项目接线要求：共用 `types_def`、MUTE/多群 FORWARD、`_llm_client`、DataCollector 列 — **满足**。

### 建议下一步
1. `python run_sim.py --config config_100_mock.yaml` → `plot_run_report.py`（确认 fig7/8）
2. 小测真 LLM：`bench_c_llm_variants.py --only V1_teacher`（确认 FORWARD 仍 >0）
3. 可选：`run_d_to_abc_integration.py` 看 D 分面评估
4. 再考虑 100×20 / 1000 mock
