"""
bench_c_llm_variants.py — C 模块小规模 A/B（速度 vs 真实感）

目的：用 10 agents × 5 steps 对比几种 LLM 配置，别一上来 100/1000。
把 terminal 输出 + output_bench_c/compare.json 发给我，我帮你判哪套更合理。

用法（Anaconda Prompt）：
  conda activate dts209tc
  cd /d "E:\\SURF\\SURF C\\联调_ABCDE"
  set DEEPSEEK_API_KEY=你的key
  python bench_c_llm_variants.py

可选：
  python bench_c_llm_variants.py --only V1_teacher
  python bench_c_llm_variants.py --ping-only   # 只打 3 次 API，不跑仿真
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# 对比矩阵：同一 seed / 同一 base，只改 llm_config
VARIANTS: dict[str, dict] = {
    # 接近「旧宽松」：窗口大、不强制 JSON、输出上限高
    "V0_loose": {
        "label": "宽松长窗口（旧习惯对照）",
        "thinking": False,
        "json_mode": False,
        "max_tokens": 2048,
        "max_prompt_chars": 24000,
        "max_memory_items": 12,
        "max_recent_messages": 12,
        "max_concurrency": 32,
        "temperature": 0.3,
    },
    # 老师建议方向：短当前窗 + 强制 JSON + 小输出
    "V1_teacher": {
        "label": "老师短窗口+JSON（主推候选）",
        "thinking": False,
        "json_mode": True,
        "max_tokens": 512,
        "max_prompt_chars": 6000,
        "max_memory_items": 4,
        "max_recent_messages": 6,
        "max_concurrency": 8,
        "temperature": 0.2,
    },
    # 稍丰富：真实感可能更好，仍远小于 1M
    "V2_richer": {
        "label": "稍丰富窗口（真实感候选）",
        "thinking": False,
        "json_mode": True,
        "max_tokens": 1024,
        "max_prompt_chars": 8000,
        "max_memory_items": 5,
        "max_recent_messages": 8,
        "max_concurrency": 16,
        "temperature": 0.2,
    },
    # 开思考：一般更慢更贵，作反面对照
    "V3_thinking": {
        "label": "开思考模式（速度反面对照）",
        "thinking": True,
        "json_mode": True,
        "max_tokens": 1024,
        "max_prompt_chars": 8000,
        "max_memory_items": 5,
        "max_recent_messages": 5,
        "max_concurrency": 4,
        "temperature": 0.2,
    },
}


def _load_base() -> dict:
    path = _HERE / "config_bench_c_base.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_variant_config(name: str, base: dict, knobs: dict) -> Path:
    cfg = deepcopy(base)
    llm = dict(cfg.get("llm_config") or {})
    llm.update(knobs)
    llm["provider"] = "deepseek"
    llm["model"] = llm.get("model") or "deepseek-v4-flash"
    llm["use_actions"] = True
    llm["user_id_prefix"] = "agent"
    cfg["llm_config"] = llm
    out = _HERE / f"_tmp_bench_{name}.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out


def ping_api(knobs: dict) -> dict:
    """不跑仿真，直接打 3 次 chat，量延迟与 token。"""
    from llm_utils import OpenAICompatibleClient, parse_llm_response

    client = OpenAICompatibleClient(
        {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
            "timeout": 60,
            "max_retries": 2,
            **knobs,
        }
    )
    sample = (
        "【任务】输出一个校园群聊行动 JSON。\n"
        "【OUTPUT JSON SCHEMA】"
        '{"action_type":"SILENT|SEND_MESSAGE","message_type":"original",'
        '"content":"","topic_id":"T001","opinion_value":0,'
        '"target_id":null,"source_message_id":null,"source_group_id":"",'
        '"destination_group_id":"CLASS_0","opinion_updates":{},'
        '"emotion_delta":{"valence":0,"arousal":0}}\n'
        "请做一个 ORDINARY 宿舍群成员在负面话题下的决策。"
    )
    rows = []
    for i in range(3):
        t0 = time.perf_counter()
        raw = client.chat(sample, user_id=f"bench-ping-{i}")
        dt = time.perf_counter() - t0
        parsed = parse_llm_response(raw)
        rows.append(
            {
                "i": i,
                "seconds": round(dt, 4),
                "raw_len": len(raw or ""),
                "action": parsed.get("action_type_name") or str(parsed.get("action_type")),
                "usage": dict(client.last_usage),
            }
        )
    return {"ping_rows": rows, "call_stats": dict(client.call_stats)}


def run_sim_variant(name: str, cfg_path: Path, out_root: Path) -> dict:
    from run_sim import load_config
    from opinion_model import OpinionModel

    config = load_config(str(cfg_path))
    t_init0 = time.perf_counter()
    model = OpinionModel(config)
    t_init = time.perf_counter() - t_init0

    step_times: list[float] = []
    t_run0 = time.perf_counter()
    for _ in range(config.n_steps):
        t0 = time.perf_counter()
        model.step()
        step_times.append(time.perf_counter() - t0)
    t_run = time.perf_counter() - t_run0

    df = model.datacollector.get_model_vars_dataframe()
    summary = model.get_final_summary()
    llm_client = getattr(model, "_llm_client", None)
    call_stats = dict(getattr(llm_client, "call_stats", {}) or {})

    out_dir = out_root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics.csv", index_label="step")

    meta = {
        "variant": name,
        "n_agents": config.n_agents,
        "n_steps": config.n_steps,
        "seed": config.random_seed,
        "llm_config": config.llm_config,
        "init_seconds": round(t_init, 4),
        "run_seconds": round(t_run, 4),
        "step_seconds_mean": round(float(sum(step_times) / max(len(step_times), 1)), 4),
        "step_seconds_max": round(float(max(step_times) if step_times else 0.0), 4),
        "call_stats": call_stats,
        "summary_snip": {
            "cross_group_forward": summary.get("cross_group_forward"),
            "cross_group_spread": summary.get("cross_group_spread"),
            "mute_count": summary.get("mute_count"),
            "announce_count": summary.get("announce_count"),
            "message_count_end": int(df["message_count"].iloc[-1]) if len(df) else None,
            "avg_opinion_end": float(df["avg_opinion"].iloc[-1]) if len(df) else None,
            "negative_emotion_end": float(df["negative_emotion"].iloc[-1]) if len(df) else None,
            "group_negative_max_peak": float(df["group_negative_max"].max()) if len(df) else None,
        },
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="只跑某一个 variant，如 V1_teacher")
    parser.add_argument("--ping-only", action="store_true", help="只 API ping，不跑仿真")
    parser.add_argument(
        "--skip-thinking",
        action="store_true",
        help="跳过 V3_thinking（省时间/钱）",
    )
    args = parser.parse_args()

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("ERROR: 请先 set DEEPSEEK_API_KEY=你的key")
        sys.exit(1)

    out_root = _HERE / "output_bench_c"
    out_root.mkdir(exist_ok=True)
    base = _load_base()

    names = list(VARIANTS.keys())
    if args.only:
        names = [args.only]
    if args.skip_thinking:
        names = [n for n in names if n != "V3_thinking"]

    results = []
    for name in names:
        if name not in VARIANTS:
            print(f"未知 variant: {name}")
            sys.exit(1)
        knobs = VARIANTS[name]
        label = knobs.get("label", name)
        print("\n" + "=" * 64)
        print(f"  {name}: {label}")
        print("=" * 64)

        if args.ping_only:
            meta = {"variant": name, "label": label, **ping_api(knobs)}
        else:
            cfg_path = _write_variant_config(name, base, knobs)
            try:
                meta = run_sim_variant(name, cfg_path, out_root)
                meta["label"] = label
            finally:
                if cfg_path.exists():
                    cfg_path.unlink()

        results.append(meta)
        print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))

    compare_path = out_root / "compare.json"
    compare_path.write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n[完成] 对比汇总 → {compare_path}")
    print("请把该文件内容（或终端输出）发我，我帮你判：速度 / 真实感 / 1000 扩展性。")


if __name__ == "__main__":
    main()
