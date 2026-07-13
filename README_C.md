# C 模块：HawkesEngine + LLM 工具链

本目录是 SURF 项目 **Multi-Agent Social Simulation for Public Opinion Governance** 的 C 模块交付代码。

C 模块主要提供两部分能力：

- `HawkesEngine`：舆情事件强度计算、下一个事件时间采样、事件历史记录。
- `LLM Toolchain`：DeepSeek/OpenAI-compatible 客户端初始化、prompt 构建、LLM 返回解析。

## 文件说明

```text
hawkes_engine.py    # Hawkes process engine, interfaces 19-22
llm_utils.py        # LLM client, prompt builder, response parser, interfaces 31-33
smoke_test_c.py     # Local smoke test, no API key required
deepseek_smoke.py   # DeepSeek live API smoke test, API key required
requirements.txt    # Python dependencies
.env.example        # Example environment variable file, no real key included
.gitignore          # Ignores .env, __pycache__, and .pyc files
HOW_TO_RUN.md       # Detailed run instructions
```

## 已实现接口

### HawkesEngine

```python
HawkesEngine.__init__(mu: float, alpha: float, beta: float)
HawkesEngine.intensity(t: float) -> float
HawkesEngine.sample_next_time(current_t: float) -> float
HawkesEngine.add_event(t: float) -> None
```

### LLM Toolchain

```python
setup_llm_client(llm_config: Dict[str, Any]) -> LLMClient
build_prompt(belief, memory, env_info) -> str
parse_llm_response(raw: str) -> Dict[str, Any]
```

## 快速运行

安装依赖：

```bash
pip install -r requirements.txt
```

运行本地冒烟测试（不需要 DeepSeek Key）：

```bash
python smoke_test_c.py
```

期望最后输出：

```text
C module smoke test passed.
```

## DeepSeek API 测试

请在终端中设置 API Key。**不要把真实 Key 写进代码或上传到 GitHub。**

Windows CMD / Anaconda Prompt:

```bat
set DEEPSEEK_API_KEY=your_real_key
python deepseek_smoke.py
```

PowerShell:

```powershell
$env:DEEPSEEK_API_KEY="your_real_key"
python deepseek_smoke.py
```

## 和其他模块的联调说明

### 和 A 模块 `OpinionModel`

A 模块可以把 `HawkesEngine` 挂到 `self.hawkes` 上：

- 调用 `intensity(t)` 获取当前舆情事件强度。
- 调用 `add_event(t)` 记录一次事件/行动发生。

### 和 B 模块 `SocialAgent`

B 模块可以把 `LLMUtils` 挂到 `self.model.llm_utils` 上。

推荐调用流程：

```text
build_prompt(...) -> client.chat(prompt) -> parse_llm_response(...)
```

## 安全注意事项

- 不要上传 `.env`。
- 不要提交任何真实 DeepSeek API Key。
- `.env.example` 只是模板，可以上传。
