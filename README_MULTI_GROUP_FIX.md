# 多群真实转发修复说明

本版本针对“一人一群、同群感知、伪跨群指标、LLM 行动未执行”四个建模缺口做了联动修复。A/B/C 的职责仍保持不变：A 负责成员关系、消息路由与指标；B 负责感知、欲望、意图和执行；C 负责 LLM/Hawkes 工具。

## 1. 成员关系：`group_type` 保留兼容，`group_ids` 才是真实成员关系

`IdentityBelief` 新增：

- `primary_group_id`：主群 ID；
- `group_ids`：Agent 实际加入的全部群。

默认 `group_membership_mode: hierarchical`，把 `group_type_ratio` 解释为“主群/最小群”的比例，并建立向上的层级成员关系：

```text
DORM   -> GROUP_DORM + GROUP_CLASS + GROUP_MAJOR + GROUP_CAMPUS
CLASS  -> GROUP_CLASS + GROUP_MAJOR + GROUP_CAMPUS
MAJOR  -> GROUP_MAJOR + GROUP_CAMPUS
CAMPUS -> GROUP_CAMPUS
```

因此默认 100 Agent 配置下，主群比例仍是 DORM 25 / CLASS 35 / MAJOR 25 / CAMPUS 15，但真实成员数为 DORM 25 / CLASS 60 / MAJOR 85 / CAMPUS 100。

需要做旧模型对照实验时，可将 `group_membership_mode` 设为 `single`。

## 2. 消息路由：消息属于目标群，不再属于“发送者主群”

`ActionRecord` / `SocialInfo` 新增：

- `message_id`：环境分配的唯一消息 ID；
- `group_id`：该条消息真正投递到的群；
- `source_message_id`：FORWARD 引用的真实源消息；
- `source_group_id`：源消息所在群。

`OpinionModel.submit_action()` 会校验：

1. 发送者存在；
2. 目标群存在；
3. 发送者是目标群成员；
4. FORWARD 引用的源消息真实存在且发送者可见。

校验失败的行动不会进入信息流。FORWARD 成功后，会在目标群创建一条新的 `ActionRecord`，目标群成员下一 tick 才能感知到它。

`get_group_messages()` 现在只返回 Agent 所在群中的消息。旧参数 `cross_group_visibility` 已弃用并忽略，不再随机泄漏异群消息。

## 3. BDI：规则转发绑定完整来源与目标

`Desire` / `Intention` 新增：

- `source_message_id`；
- `source_group_id`；
- `destination_group_id`。

规则链的 `share` 会：

1. 从本轮真实可见消息中选择源消息；
2. 查询发送者可投递的其他成员群；
3. 默认按 `next_larger` 逐层选择目标群；
4. 将完整路由字段传到 `ActionRecord`。

可配置策略：

```yaml
network_params:
  group_membership_mode: hierarchical
  forward_destination_strategy: next_larger  # next_larger | largest | random
  allow_downward_forward: false
```

## 4. LLM：行动结果不再被丢弃

LLM 输出 schema 新增：

```json
{
  "action_type": "FORWARD",
  "target_id": 12,
  "source_message_id": "M00000042",
  "source_group_id": "GROUP_DORM",
  "destination_group_id": "GROUP_CAMPUS"
}
```

`SocialAgent._update_beliefs()` 在更新观点和情绪后，会把合法 LLM 行动转换为 `Intention`；`step()` 优先执行该意图，不再重新走规则 `_infer_desires -> _plan_intentions` 覆盖它。

安全校验仍由本地代码完成：LLM 编造的消息 ID、Agent ID 或非成员群不会被执行。`FORWARD`/`REPLY` 必须命中本轮可见消息。

`llm_config.use_actions` 默认为 `true`。若只想让 LLM 更新观点/情绪而保留规则行动，可显式设置：

```yaml
llm_config:
  provider: deepseek
  use_actions: false
```

注意：`MockLLMClient` 固定返回 `SILENT`，所以默认 `config.yaml` 使用 `llm_config: {}`，避免冒烟仿真被全部静默。

## 5. 指标定义已拆清

- `cross_group_forward`：仅统计 Agent 实际 FORWARD，且 `source_group_id != group_id`；
- `cross_group_spread`：仅统计环境层热度从一个 GroupType 溅射到另一个 GroupType；
- 消息热度和负面值归入 `record.group_id` 对应的目标群，而不是发送者主群。

因此图中的两个指标现在具有不同、可解释的含义，不会再把环境热度扩散冒充成人际转发。

## 6. 运行与验证

```bash
python3 test_contract.py
python3 test_module_a.py
python3 run_sim.py --config config.yaml --no-plot
python3 run_sim.py --config config_1000.yaml --no-plot
```

本次修改后的验证结果：

- `test_contract.py`：20/20 通过；
- `test_module_a.py`：通过；
- 100 Agent × 50 步：通过，出现真实跨群转发；
- 1000 Agent × 50 步：最终打包回归（seed=1260807164）仿真主体约 4.27 秒，低于 30 秒验收线；记录到 371 次真实 Agent 跨群转发和 96 次宏观热度扩散。

## 7. 最小真实转发流程

```python
# 1) 原消息进入宿舍群
original = ActionRecord(
    agent_id=origin_id,
    action_type=ActionType.SEND_MESSAGE,
    content="宿舍群原文",
    group_id="GROUP_DORM",
)
model.submit_action(original)

# 2) 同时属于宿舍群与班级群的桥接成员转发
forwarded = ActionRecord(
    agent_id=bridge_id,
    action_type=ActionType.FORWARD,
    content="转发到班级群",
    group_id="GROUP_CLASS",
    source_message_id=original.message_id,
)
model.submit_action(forwarded)
```

班级群成员看不到第一条宿舍群原文，但能看到第二条真正投递到 `GROUP_CLASS` 的转发记录。
