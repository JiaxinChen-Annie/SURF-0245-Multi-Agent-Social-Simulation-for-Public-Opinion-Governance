# Module C Quick Start（接口表 20260718_v2）

C 负责：
- `HawkesEngine`（接口 21–24）：Agent 激活强度，**不是**群热度 H(t)
- LLM 工具链（接口 33–35）：`setup_llm_client` / `build_prompt` / `parse_llm_response`

群热度公式 `H·e^{-α}·e^{-β_k}` 由 A 维护，B 的 `calc_heat_decay` 只供 A 调用。

---

## 1) 安装依赖

```bash
cd "E:\SURF\SURF C\交付代码"
pip install -r requirements.txt
```

## 2) 本地冒烟（不需要 Key）——验证 prompt/parse 是否正确

```bash
python smoke_test_c.py
```

**通过标准（最后一行）：**

```text
C module smoke test passed.
```

中间应看到类似：

- `Hawkes history: [...]`
- `Prompt length: <较大正数>`
- `Parsed from mock: {... "action_type_name": "SILENT" ...}`
- `Parsed from noisy: {... "action_type_name": "SEND_MESSAGE", "message_type": "clarification" ...}`

若 Assert 失败，说明 v2 解析或 prompt 字段坏了。

### 在 Python 里手工验 prompt（可选）

```python
from llm_utils import build_prompt, parse_llm_response

belief = {
    "identity": {"agent_type": "RATIONAL", "group_type": "MAJOR", "nickname": "小李"},
    "emotion": {"valence": -0.2, "arousal": 0.55},
    "opinions": {"T001": {"topic_id": "T001", "opinion_value": -0.3, "confidence": 0.5}},
}
memory = [{"tick": 1, "info": {"topic_id": "T001", "message_type": "exaggerate", "content": "全校都完了"}}]
env = {
    "group_type": "MAJOR",
    "beta": 0.20,
    "role": "RATIONAL",
    "nickname": "小李",
    "recent_messages": ["[exaggerate] 阿强: 全校都完了"],
    "mentions": [],
    "topic_heat": {"T001": 0.9},
    "topic_negative": {"T001": 0.75},
}

p = build_prompt(belief, memory, env)
print(p)  # 应含：角色提示、群类型、topic_heat、OUTPUT JSON SCHEMA、SEND_MESSAGE/REPLY/FORWARD/SILENT

# 假 LLM 输出
raw = '{"action_type":"REPLY","message_type":"paraphrase","content":"先核实来源","topic_id":"T001","opinion_updates":{"T001":-0.1},"emotion_delta":{"valence":0.0,"arousal":-0.05}}'
print(parse_llm_response(raw))
```

**正确 parse 至少应有：**

| 字段 | 要求 |
|------|------|
| `action_type_name` | SEND_MESSAGE / REPLY / FORWARD / SILENT |
| `message_type` | original/forward/paraphrase/exaggerate/clarification（或合理默认） |
| `topic_id` | 字符串（兼容旧字段 `event_id`） |
| `opinion_updates` | dict |
| `emotion_delta` | 含 valence / arousal |
| 非法 JSON | 降级为 SILENT，不抛异常 |

---

## 3) 填 Key 连真 LLM（DeepSeek）

**不要把 Key 写进代码或提交 Git。**

### Anaconda Prompt / CMD

```bat
cd /d "E:\SURF\SURF C\交付代码"
set DEEPSEEK_API_KEY=你的真实key
set DEEPSEEK_MODEL=deepseek-v4-flash
python deepseek_smoke.py
```

### PowerShell

```powershell
cd "E:\SURF\SURF C\交付代码"
$env:DEEPSEEK_API_KEY="你的真实key"
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
python deepseek_smoke.py
```

说明：`base_url` 仍是 `https://api.deepseek.com`（OpenAI 兼容格式）。  
`deepseek-chat` / `deepseek-reasoner` 将于 **2026/07/24** 弃用；请用 `deepseek-v4-flash`（默认）或 `deepseek-v4-pro`。

**通过标准：**

1. 打印 `=== RAW ===`（模型原文，允许带 markdown 代码块）
2. 打印 `=== PARSED ===`（字典，含 `action_type_name` / `opinion_updates` / `emotion_delta`）
3. 最后一行：`DeepSeek live smoke OK.`
4. CONTROLLER + 高热度场景下，常见合理结果是 `SEND_MESSAGE` + `message_type=clarification`（不强制，但不应崩溃）

若报错：

| 现象 | 处理 |
|------|------|
| `DEEPSEEK_API_KEY is empty` | 当前终端没设环境变量；重新 `set` 后再跑 |
| 401 / Authentication | Key 错或过期 |
| 404 model | 改 `DEEPSEEK_MODEL`（试 `deepseek-v4-flash` / `deepseek-v4-pro`） |
| 超时 / 连接失败 | 检查网络；可增大 `timeout` |

---

## 4) 和 A/B 联调时注意

- A：`HawkesEngine` 只管激活强度；群 `topic_heat` 走 A 的 `_update_environment`
- B：`env_info` 建议带 `group_type / beta / role / recent_messages / topic_heat / topic_negative`
- B 读 LLM：优先 `opinion_updates` + `emotion_delta`；失败则保持原信念
