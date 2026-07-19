# D 到 ABC 联调说明

本目录用于 Week4 并行模块联调：以 A 包中的 Week4 统一版 ABC 主线为运行环境，将 D 模块的 `StrategyEvaluator` 接入真实 `OpinionModel` 输出。

## 文件说明

- `types_def.py`：ABC 共用数据结构，包含 `ActionRecord`、`SimConfig`、`InterventionType` 等接口字段。
- `opinion_model.py`：A 模块环境层，负责调度 Agent、更新环境、汇总 DataCollector 指标。
- `social_agent.py`：B 模块规则桩，负责 Agent 感知、信念更新、行动生成。
- `hawkes_engine.py`：C 模块 Hawkes 事件强度与事件回写。
- `strategy_eval.py`：D 模块策略评估原始代码。
- `abc_model_adapter.py`：本次新增适配层，把 ABC 的 DataCollector 指标转换为 D 需要的评估输入。
- `run_d_to_abc_integration.py`：本次新增联调运行脚本。

## 运行方式

在 PyCharm 中打开本目录，选择 Python 解释器后运行：

```bash
python run_d_to_abc_integration.py
```

也可以指定参数：

```bash
python run_d_to_abc_integration.py --agents 50 --steps 50 --seed 42
```

## 联调链路

```text
B 生成 ActionRecord
  -> A.submit_action 写入环境缓存
  -> A.step 调用 C/Hawkes 更新传播节奏
  -> A.DataCollector 汇总 avg_opinion / polarization / message_count 等指标
  -> abc_model_adapter 转换为 D 所需 metrics
  -> D.StrategyEvaluator 建立 baseline 并评估三类干预
```

## 输出结果

默认输出到 `d_to_abc_results`：

- `baseline.csv`
- `event_injection_public_info.csv`
- `node_control_leaders.csv`
- `platform_param_downrank.csv`
- `evaluation_summary.json`
- `d_to_abc_integration_report.txt`

## 当前说明

单独的 B zip 仍是早期字段命名；本次为了保证 ABCD 接口一致，使用 A 包内已对齐 Week4 v2 接口的 `social_agent.py` 作为 B 模块联调桩。
