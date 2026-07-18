# C 模块：HawkesEngine + LLM 工具链（v2）

对齐接口表：**多智能体舆情仿真_全函数接口表_20260718_v2**

场景：校园多层群聊（DORM/CLASS/MAJOR/CAMPUS）+ 角色（ORDINARY/ACTIVE/RATIONAL/CONTROLLER）

## 文件

```text
hawkes_engine.py    # 接口 21–24：Agent 激活 Hawkes（≠ 群热度 H(t)）
llm_utils.py        # 接口 33–35：setup_llm_client / build_prompt / parse_llm_response
smoke_test_c.py     # 本地冒烟（无 Key）
deepseek_smoke.py   # 真 LLM 冒烟（需 DEEPSEEK_API_KEY）
requirements.txt
.env.example
HOW_TO_RUN.md       # 详细验证步骤（推荐先看）
```

## 接口

```python
HawkesEngine(mu, alpha, beta)  # 可选 rng=
.intensity(t) -> float
.sample_next_time(current_t) -> float
.add_event(t) -> None

setup_llm_client(llm_config) -> LLMClient  # .chat(str)->str
build_prompt(belief, memory, env_info) -> str
parse_llm_response(raw) -> Dict  # 含 opinion_updates / emotion_delta
```

## 快速验证

```bash
pip install -r requirements.txt
python smoke_test_c.py
```

填 Key 后：

```bat
set DEEPSEEK_API_KEY=your_real_key
set DEEPSEEK_MODEL=deepseek-v4-flash
python deepseek_smoke.py
```

模型说明：官网 OpenAI 兼容 `base_url=https://api.deepseek.com`；推荐 `deepseek-v4-flash`（或 `deepseek-v4-pro`）。`deepseek-chat` 将于 2026/07/24 弃用。

细节与验收标准见 `HOW_TO_RUN.md`。

## 安全

不要提交真实 API Key；不要上传 `.env`。
