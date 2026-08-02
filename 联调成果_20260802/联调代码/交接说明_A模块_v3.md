# A 模块 v3 交接说明 —— 8 个场景不一致问题的修改记录

负责人：A（模块二 OpinionModel / 系统骨架）
基准文档：《场景设定_20260718_v2》《分工规范_20260630》+ 流程图
验证状态：`test_contract.py` 29/29 通过；`test_module_a.py` 通过；
100 Agent×50 步 2.2s，1000 Agent×50 步 24.8s（验收线 30s）。

---

## 一、逐条修改

### ① 干预触发条件（最高优先级）

**问题**：设定写「负面程度超过阈值 → controller 干预」，代码主要按热度 H ≥ θ 触发，
负面高但不热就不干预。

**根因**不是判断语句写错，而是**负面程度这个量本身越不过阈值**：旧实现只在
`submit_action` 里做 EMA(0.2) 累加，群里没消息也不回落，单条极端负面消息被稀释。

**改法**
| 位置 | 内容 |
| --- | --- |
| `opinion_model._refresh_group_negative()` | 群负面重定义为「群内存活消息 `negative_score` 的加权均值」，权重 = 新鲜度(τ=4) × 传播力(1+0.8·forward_count) × 失真度(1+0.4·distortion)。一条被转了 5 次的夸大消息，对「这个群现在有多负面」的贡献远大于一条没人理的日常发言。 |
| `opinion_model._evaluate_intervention_triggers()` | 默认 `intervention_trigger: negative`；`heat_threshold` 从 `THETA` 单列出来，不再和负面阈值混用；支持 `heat` / `heat_or_negative` / `heat_and_negative` 切换。 |
| `step()` 时序 | 触发判定移到 Agent 激活**之前**，Controller 当拍就能响应，不再延迟一 tick。 |
| 新参数 `intervention_min_messages: 4` | 最小证据量。否则开局第一条 original（neg=0.72）会把整群判成超阈，全场从 tick 0 就进恢复期。 |
| 新增 `governance_tick` | 与 `intervention_tick` 分开：前者是 Controller **真正执行**了公告/禁言的 tick，《场景设定》§3「Controller 干预后 → 负面与失真下降」以它为准。原来只要阈值一到负面就自动回落，Controller 还没动手事情就没了，D 模块的干预消融实验会失效。 |

新指标列 `group_negative_max` 就是触发判定的直接观测量，可以直接和 0.65 阈值对照看。

### ② 事件发起

新文件 **`event_source.py`**，对应流程图第一个节点。`EventSource` 有独立生命周期：
- `inject_initial()`：`__init__` 末尾投放，`heat` 按 **H₀ 绝对值**写入目标群
  （`submit_action(..., absolute_heat=True)`），不是当普通消息加增量；
- `step()`：每 tick 调用，按 `secondary_event_probability` 追加「后续爆料」，
  舆情因此可能出现二次高峰而不是单峰衰减；
- 小群 = {DORM, CLASS}，大群 = {MAJOR, CAMPUS}，不再写死 DORM/CAMPUS；
- 产出 `EventRecord` 列表，`get_final_summary()["event_source"]` 直接可读。

### ③ 跨群条件

- `ActionRecord` 新增 `forward_count`（转发链深度）、`root_message_id`（谱系根）、
  `forward_direction`；
- `lineage_forward_total()` 取 `max(链深度, 该根消息全网被转总次数)`，
  前者刻画链式接力，后者刻画广度扩散，任一达标即认为具备跨群动能；
- **群内 FORWARD 始终允许**，这是攒够转发次数的唯一途径；
- 跨群需达门槛，或源群热度已 ≥ θ（`cross_group_heat_bypass`，对应设定里
  「消息如果热度高 → 有概率进入其他任意群」）；
- 宏观热度溅射同样受门槛约束（`macro_spread_requires_threshold`）。

### ④ 小群互转

`ForwardDirection` 四类通道，门槛与权重都分级：

| 通道 | 判定 | 转发次数门槛 | 权重 |
| --- | --- | --- | --- |
| INTRA 群内 | src == dst | 0 | 0.60 |
| LATERAL 小群互转 | **DORM ↔ CLASS**（流程图里唯一那条横向箭头） | 0 | 0.55 |
| UPWARD 小→大 | 群规模变大，含 MAJOR→CAMPUS | 1 | 1.00 |
| DOWNWARD 大→小 | 群规模变小 | 3 | 0.15 |

注意 **MAJOR→CAMPUS 判为 UPWARD 而不是横向** —— 专业群→校园群本来就是「小→大」，
早期版本按「大群带内」把它算成横向，导致主通道统计几乎为 0。

`allow_downward_forward` 默认改为 `true`（原来是 false，向下完全不可能）。
目标群按 `方向权重 × 曝光因子` 加权抽样，`forward_exposure_weight: 0.5` 控制曝光的
影响力度 —— 取 1.0 会让宿舍群被校园群彻底压死，横向通道有名无实。

实测 100 Agent×50 步：群内 124 / 小→大 13 / 小群互转 4 / 向下 3，符合
「主要是小→大，横向和向下弱」。

### ⑤ 禁言

- `ActionType` 新增 `MUTE=4` / `ANNOUNCE=5`；
- 环境层真正执行：`muted_until[(agent_id, group_id)]`，被禁言者在该群的一切投稿被
  `submit_action` 拒绝（计入 `blocked_by_mute`）；
- **DORM 群不可禁言**（control_level 最低，Controller 介入不了），非 CONTROLLER 越权
  会被拒；
- 禁言时长 = `mute_duration_ticks × control_level / 2`，管理强的群禁得久；
- Controller 走升级阶梯：先公告 → 压不住再禁言「失真×0.5 + 负面×0.5」最高的传播者，
  同群治理动作有 3 tick 冷却期；
- 新指标 `mute_count` / `active_mutes`。

### ⑥ 曝光

`types_def.GROUP_EXPOSURE = {DORM 0.20, CLASS 0.50, MAJOR 0.75, CAMPUS 1.00}`，
取自《场景设定》§1 群类型表的「曝光」列。三处生效：

1. **热度增益**：同一条消息在校园群掀起的热度显著高于宿舍群；
2. **宏观溅射目标选择**：高曝光群更容易被外部舆情渗透；
3. **注意力配额**：`get_group_messages` 改成按曝光**分摊名额**，而不是「谁的消息新谁占满」。
   这一条最关键 —— 不改的话小群消息永远被校园群刷掉，小群互转和小→大扩散都无从发生。

### ⑦ 降温公式表述

两处表述本来就代数等价，现在统一到 `types_def.heat_decay()`：

```
未干预：H(t+1) = H(t) · exp(−α)                      「基础降温」
已干预：H(t+1) = H(t) · exp(−α · control_level_k)     「基础降温 × control_level」
              ≡ H(t) · exp(−α) · exp(−β_k)            （§4.2 式(2)）
因为 β_k = α · (control_level_k − 1)
```

`GROUP_CONTROL_LEVEL = {1.333, 1.80, 2.333, 3.00}` 是**唯一可调参数**，
`GROUP_BETA` 由它派生，数值仍是接口表里的 0.05 / 0.12 / 0.20 / 0.30，不用改接口表。
`test_contract` 里有一条测试逐位比对两种写法。

### ⑧ 次要

- A 模块全链路无 `event_id` 残留，一律 `topic_id`；
- `PsychologyBelief` 新增 `trust` / `confirmation_bias`，按角色分层
  （RATIONAL 低信任低偏误、ACTIVE 高偏误、ORDINARY 高信任），接进置信区间、
  学习率、转发意愿、情绪易感度四处；
- 新增 `_absorb_emotional_contagion()`：没有它，agent 情绪只会向基线回归，
  群负面永远停在 0.5 附近，①的触发条件形同虚设。

---

## 二、需要各模块负责人接手的事

### B（SocialAgent）
我动了你的存根，请接手并保持语义：
- `_infer_desires`：新增 mute / announce 阶梯与群内转发分支；Controller 以
  `perception.intervened_groups`（A 模块权威）为准决定是否出手，不再自己算阈值；
- `_init_psychology`：新增 trust / confirmation_bias 分角色区间；
- 新增 `_absorb_emotional_contagion()`，在 `step()` 里 `_update_beliefs` 之后调用，
  与 LLM / 规则两条路径正交；
- `calc_heat_decay` 改成转调 `types_def.heat_decay`，**别再自己写指数式**；
- 治理动作不走 `p_act` 概率门限（那是职责不是性格）。

**遗留问题**：观点动力学仍然很弱（50 步 `avg_opinion` 只从 +0.133 走到 +0.120，
`polarization` 几乎不动）。原因是规则路径把「邻居观点」用 `negative_score` 反推，
信息量太少。这条不在本次 8 个问题里，但接 LLM 之前建议你先补一个真实的观点交换机制。

### C（HawkesEngine + LLM 工具链）
- `llm_utils.py` 里 `event_id` 别名还在（`_normalize_response` 约 373–376 行、
  `_silent_response` 约 439 行），主路径已全 `topic_id`，请清掉；
- prompt schema 需要加 `MUTE` / `ANNOUNCE` 两个 action（A 侧会校验只有 CONTROLLER
  能用，越权自动降级为 `SEND_MESSAGE`）；
- 必须让模型输出 `source_message_id` / `destination_group_id`，A 侧做合法性校验，
  模型编的 ID / 非成员群会被直接丢弃；
- `MockLLMClient` 固定返回 SILENT，所以三份 config 的 `llm_config` 都留空 `{}`。

### D（StrategyEvaluator）
- **`intervention_tick` 和 `governance_tick` 是两回事**：前者是阈值触发时刻（进热度
  公式的 𝟙[t ≥ t_k^int]），后者是 Controller 实际动手时刻（进负面/失真恢复）。
  做干预消融时用 `governance_tick`，否则「有无干预」两组会没有差别；
- `EventSource` 可以直接复用做「事件注入型干预」：
  `model.submit_action(record, bypass_mute=True, absolute_heat=True)`；
- 节点控制型干预可以直接用 `MUTE`；平台参数型干预改 `scenario_params` 即可，
  所有阈值/权重都集中在那里。

### E（Utils / 可视化）
两个**破坏性变更**，请同步：
1. `intervention_tick` 未触发从 `inf` 改成 **`-1.0`** 哨兵（`inf` 会污染 `describe()`）；
2. `negative_emotion` 从「valence<0 的 agent 比例」改成「平均负性强度 (1−valence)/2」。
   旧定义在爆发期会迅速饱和到 1.0，看不出强度差异也看不出干预后的回落。

新增列：`upward_forward`、`lateral_forward`、`downward_forward`、`intra_group_forward`、
`group_negative_max`、`group_heat_max`、`mute_count`、`active_mutes`、`n_intervened_groups`。

`get_final_summary()` 现在包含事件源投放明细、四条通道分解、control_level 与 exposure
对照表，可以直接进报告。

---

## 三、已知限制（下一轮再处理）

1. **参数对规模敏感**：`base_heat_gain` 是按 100 Agent 标定的。1000 Agent 时消息量
   大得多，群负面均值被稀释（实测峰值只有 0.52，触发不了 0.65），而热度反而顶到
   2.58。压测配置目前只用来验性能，不用来看舆情结论。建议把 `base_heat_gain` 和
   `negative` 的统计口径改成按群成员数归一化。
2. **触发有随机性**：20 个种子里 15 个触发干预、10 个出现完整恢复。这在建模上是合理
   的（不是每个事件都会变成危机），但如果实验需要稳定的对照组，请固定 `random_seed`
   并先跑一遍确认该种子会触发。`config.yaml` 已固定为 `random_seed: 42`（会触发）。
3. **1000 Agent 24.8s**，离 30s 验收线只剩 5s 余量。瓶颈在每 Agent 每 tick 的
   `get_group_messages` 全缓存扫描。如果之后要加 LLM 或扩大规模，这里需要改成
   按群维护倒排索引。
4. **群实例只有一个/类**：现在 `GROUP_DORM` 是单一频道，所以「两个不同宿舍群之间
   互转」建模不了，横向只能是 DORM↔CLASS。若要更真实，需要支持每类群多个实例
   （`GROUP_DORM_00/01/...`）并建立群的层级树，改动面较大。

---

## 四、文件清单

| 文件 | 状态 | 归属 |
| --- | --- | --- |
| `types_def.py` | 大改（新增枚举/常量/字段/默认参数表） | A 牵头冻结，全队共用 |
| `opinion_model.py` | 大改 | A |
| `event_source.py` | **新增** | A |
| `run_sim.py` | 改（新增场景一致性输出与两张图的叠加曲线） | A |
| `config.yaml` / `config_1000.yaml` / `config_single.yaml` | 重写 | A |
| `test_contract.py` | 改 + 新增 9 条 v3 场景一致性测试 | A |
| `test_module_a.py` | 改（渐进触发探针改走负面通道） | A |
| `social_agent.py` | 改（存根，需 B 接手） | B |
| `hawkes_engine.py` / `llm_utils.py` / `mesa_compat.py` / `mesa_patch.py` | 未改动 | C / A |

运行方式：

```bash
python3 test_contract.py                              # 29 项接口 + 场景一致性
python3 test_module_a.py                              # A 模块联调
python3 run_sim.py --config config.yaml               # 默认 100 Agent，出图
python3 run_sim.py --config config_1000.yaml --no-plot   # 压测
```
