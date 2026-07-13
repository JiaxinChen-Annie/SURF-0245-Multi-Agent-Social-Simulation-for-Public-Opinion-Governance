# Module C Quick Start

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Run local smoke test (no API needed)

```bash
python smoke_test_c.py
```

## 3) Use DeepSeek API

Set environment variable (Windows PowerShell):

```bash
$env:DEEPSEEK_API_KEY="your_real_key"
```

Then configure and create client:

```python
from llm_utils import setup_llm_client

client = setup_llm_client(
    {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "timeout": 30,
        "max_retries": 2,
        "temperature": 0.3,
    }
)
```

Optional live smoke test:

```bash
python deepseek_smoke.py
```

## 4) Integration expectations

- `HawkesEngine` provides:
  - `__init__(mu, alpha, beta)`
  - `intensity(t)`
  - `sample_next_time(current_t)`
  - `add_event(t)`
- LLM toolchain provides:
  - `setup_llm_client(llm_config)`
  - `build_prompt(belief, memory, env_info)`
  - `parse_llm_response(raw)`
