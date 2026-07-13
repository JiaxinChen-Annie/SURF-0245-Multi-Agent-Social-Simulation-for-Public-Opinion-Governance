# D 部分：StrategyEvaluator 单独运行版

这个目录实现你负责的 D 模块，按第一周接口定义包含：

- `StrategyEvaluator.__init__(model)`
- `set_baseline()`
- `apply_intervention(intervention_type, params)`
- `evaluate()`
- `_analyze_behavior()`
- `_analyze_content()`
- `_analyze_topology()`

为了在 A/B/C 未完成前也能单独运行，代码内置了 `MockOpinionModel`。后续接入真实 `OpinionModel` 时，只要真实模型提供：

```python
clone_with_intervention(intervention_type, params, seed)
run(steps) -> pandas.DataFrame
```

并返回同名指标列，D 模块就可以复用。

## 运行

在本目录下执行：

```powershell
python run_strategy_eval.py
```

如果环境没有依赖，先安装：

```powershell
pip install -r requirements.txt
```

或指定参数：

```powershell
python run_strategy_eval.py --steps 50 --seed 42 --agents 1000 --out strategy_eval_results
```

会生成：

- `baseline.csv`
- 三类干预策略的 CSV
- `evaluation_summary.json`

## 三类干预

- `EVENT_INJECTION`：事件注入 / 官方信息公开
- `NODE_CONTROL`：节点控制 / 意见领袖降权
- `PLATFORM_PARAM`：平台参数调整 / 降低转发扩散

## 三维度评估

- 行为维度：参与率、传播速度、互动密度
- 内容维度：情感均值、情感方差、主题迁移、极化度
- 拓扑维度：网络密度、模块度、意见领袖中心性
