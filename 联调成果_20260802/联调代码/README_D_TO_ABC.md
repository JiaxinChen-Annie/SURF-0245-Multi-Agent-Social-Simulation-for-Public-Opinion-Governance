# D 到 ABC 联调说明

本目录用于将 D 模块 `StrategyEvaluator` 接入 ABC 主线输出。D 模块不修改 ABC 的核心逻辑，只读取 ABC 产生的时序指标、行为记录和 DataCollector 状态，并在此基础上完成干预策略评估。

## 本次修改对应老师建议

1. 干预有效性不再只看 `intervention_tick`，而是优先使用 `governance_tick` 作为真实官方介入时间点。
2. 评估报告新增 `GroupType` 与 `AgentType` 分组结果，不再只输出全网平均值。
3. 新增评估窗口：以 `governance_tick` 为中心，比较干预前 N tick 与干预后 N tick 的负面程度、情绪传染等指标。
4. 新增 rolling window：对 `negative_emotion` 与 `emotional_contagion` 做滑动窗口均值，降低单个 tick 抖动对结论的影响。
5. 支持 `time_profile`，可设置课堂时段、晚间时段等不同干预强度。

## 运行方式

先安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行默认联调：

```bash
python run_d_to_abc_integration.py
```

也可以指定参数：

```bash
python run_d_to_abc_integration.py --agents 100 --steps 50 --seed 42 --governance-tick 10 --eval-window 10 --rolling-window 5
```

参数含义：

- `--governance-tick`：真实官方干预开始的 tick。
- `--eval-window`：干预前后各取多少 tick 做均值对比。
- `--rolling-window`：负面情绪和情绪传染的滑动窗口长度。
- `--agents`：仿真智能体数量。
- `--steps`：仿真总步数。
- `--seed`：随机种子。

## 联调链路

```text
B 生成 ActionRecord
  -> A.submit_action 写入环境
  -> A.step 调用 C/Hawkes 更新传播过程
  -> A.DataCollector 采集 avg_opinion / polarization / negative_emotion / emotional_contagion
  -> abc_model_adapter 转换成 D 所需 metrics
  -> D.StrategyEvaluator 建立 baseline 并评估三类官方干预策略
```

## 输出结果

默认输出目录为 `d_to_abc_results`：

- `baseline.csv`：无干预基线结果。
- `event_injection_public_info.csv`：信息发布类干预结果。
- `node_control_leaders.csv`：关键节点控制类干预结果。
- `platform_param_downrank.csv`：平台降权类干预结果。
- `evaluation_summary.json`：结构化评估摘要。
- `d_to_abc_integration_report.txt`：可直接放入汇报材料的文字版报告。

报告中重点看：

- `governance_tick`：真实官方介入时刻。
- `negative_drop`：干预后负面程度相对 baseline 的下降幅度。
- `contagion_drop`：干预后情绪传染相对 baseline 的下降幅度。
- `GroupType report`：宿舍群、班级群、专业群、校园群的分组指标。
- `AgentType report`：普通用户、活跃用户、理性用户、控制者的分角色指标。

## 注意

如果只验证 D 模块，可直接使用 `MockOpinionModel`。如果运行 `run_d_to_abc_integration.py`，需要同目录或 Python 路径中存在 ABC 主线文件，例如 `opinion_model.py`、`social_agent.py`、`hawkes_engine.py`、`types_def.py`。
