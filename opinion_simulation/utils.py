"""
多智能体舆情仿真 — Utils 工具函数模块
E (刘嘉铭) 负责

包含: load_config, setup_llm_client, build_prompt,
      parse_llm_response, save_simulation_results,
      visualize_trend, compute_similarity, get_logger
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

# 同级目录导入自定义类型
try:
    from sim_types import (
        SimConfig, ActionType, BeliefSystem, MemoryRecord, Perception,
    )
except ImportError:
    # fallback: 项目尚在搭建阶段时兼容直接执行
    SimConfig = None           # type: ignore
    ActionType = None          # type: ignore
    BeliefSystem = None        # type: ignore
    MemoryRecord = None        # type: ignore
    Perception = None          # type: ignore

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

_LOG = logging.getLogger("utils")
"""模块级日志"""

LOGGER_CACHE: Dict[str, logging.Logger] = {}
"""缓存已创建的 logger, 避免重复配置"""

DEFAULT_FIG_SIZE = (14, 8)
"""可视化默认画布尺寸 (英寸)"""

# ── matplotlib 中文字体配置 ──
# 自动查找系统中文字体, 避免 CJK 字形缺失警告
_CN_FONTS = [
    "PingFang SC", "PingFang TC", "STHeiti", "Heiti SC",
    "Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC",
    "WenQuanYi Micro Hei", "AR PL UMing CN",
]
for _f in _CN_FONTS:
    try:
        plt.rcParams["font.sans-serif"].insert(0, _f)
        plt.rcParams["axes.unicode_minus"] = False
        # 快速验证: 该字体能否渲染一个中文
        fig_test, ax_test = plt.subplots(figsize=(0.1, 0.1))
        ax_test.set_title("测")
        fig_test.canvas.draw()
        plt.close(fig_test)
        break
    except Exception:
        # 尝试下一个字体
        plt.rcParams["font.sans-serif"].pop(0)
        continue

# ──────────────────────────────────────────────
# #30 load_config
# ──────────────────────────────────────────────

def load_config(path: str) -> SimConfig:
    """
    读取实验参数配置文件, 返回 SimConfig 对象。

    支持 YAML (.yaml / .yml) 和 JSON (.json) 格式。

    Parameters
    ----------
    path : str
        配置文件的路径 (绝对或相对路径)。

    Returns
    -------
    SimConfig
        解析后的仿真实验配置对象。

    Raises
    ------
    FileNotFoundError
        文件不存在。
    ValueError
        文件格式不支持或内容解析失败。
    """
    _path = Path(path)

    if not _path.exists():
        raise FileNotFoundError(f"配置文件不存在: {_path.resolve()}")

    suffix = _path.suffix.lower()

    try:
        raw = _path.read_text(encoding="utf-8")

        if suffix in (".yaml", ".yml"):
            data: Dict[str, Any] = yaml.safe_load(raw)
        elif suffix == ".json":
            data = json.loads(raw)
        else:
            raise ValueError(f"不支持的文件格式: {suffix} (支持: .yaml .yml .json)")

    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValueError(f"配置文件解析失败: {exc}") from exc

    return SimConfig.from_dict(data)


# ──────────────────────────────────────────────
# #31 setup_llm_client
# ──────────────────────────────────────────────

class LLMClient:
    """
    LLM 客户端统一协议。

    所有 LLM 接入必须暴露 chat(prompt: str) -> str 方法。
    由 setup_llm_client 工厂函数创建具体实例。
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._client = None
        self._model = config.get("model", "gpt-3.5-turbo")
        self._base_url = config.get("base_url", "")
        self._api_key = config.get("api_key", "")
        self._temperature = config.get("temperature", 0.7)
        self._max_tokens = config.get("max_tokens", 1024)
        self._init_client()

    def _init_client(self) -> None:
        """尝试初始化 OpenAI 兼容客户端。"""
        provider = self.config.get("provider", "openai")

        if provider in ("openai", "openai_compatible"):
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url or None,
                )
            except ImportError:
                _LOG.warning("openai 库未安装, 将回退到 Mock 模式")
                self._client = None
        else:
            _LOG.warning(
                f"不支持的 LLM provider: {provider}, 回退到 Mock 模式"
            )
            self._client = None

    def chat(self, prompt: str) -> str:
        """
        发送 prompt 并返回 LLM 响应文本。

        Parameters
        ----------
        prompt : str
            输入的完整 prompt 字符串。

        Returns
        -------
        str
            LLM 返回的响应文本。
        """
        if self._client is None:
            return self._mock_chat(prompt)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            _LOG.error("LLM 调用失败: %s, 回退到 Mock 模式", exc)
            return self._mock_chat(prompt)

    def _mock_chat(self, prompt: str) -> str:
        """Mock 模式: 返回 JSON 格式的默认响应。"""
        _LOG.debug("LLM Mock chat (prompt 前 80 字): %s...", prompt[:80])
        return '{"action_type": "SILENT", "content": "", "event_id": "", "opinion_value": 0.0}'


def setup_llm_client(llm_config: Dict[str, Any]) -> LLMClient:
    """
    根据配置字典创建 LLM 客户端, 暴露 chat(prompt: str) -> str。

    支持 OpenAI 和 OpenAI 兼容 API。
    若 openai 库未安装或调用失败, 自动回退到 Mock 模式。

    Parameters
    ----------
    llm_config : Dict[str, Any]
        包含 provider / model / base_url / api_key 等字段的字典。

    Returns
    -------
    LLMClient
        已初始化的 LLM 客户端对象。
    """
    return LLMClient(llm_config)


# ──────────────────────────────────────────────
# #32 build_prompt
# ──────────────────────────────────────────────

def build_prompt(belief: "BeliefSystem",
                 memory: List["MemoryRecord"],
                 env_info: "Perception") -> str:
    """
    将智能体的信念、记忆和当前感知组装成 LLM 调用 prompt。

    生成结构化提示, 引导 LLM 以 JSON 格式输出下一步行动决策。

    Parameters
    ----------
    belief : BeliefSystem
        智能体当前的信念系统 (身份、心理、观点、情绪)。
    memory : List[MemoryRecord]
        智能体的近期记忆列表。
    env_info : Perception
        当前环境感知 (邻居发言、热门话题、被提及的消息)。

    Returns
    -------
    str
        可直接传给 LLMClient.chat() 的完整 prompt 字符串。
    """
    # ── 身份与心理 ──
    identity = belief.identity
    psych = belief.psychology
    emotion = belief.emotion

    identity_str = (
        f"角色类型: {identity.agent_type.name}\n"
        f"角色描述: {identity.role_desc}\n"
        f"初始立场: {identity.stance_prior:.2f}\n"
    )

    personality = psych.personality
    psych_str = (
        f"人格 (大五): O={personality.openness:.2f} "
        f"C={personality.conscientiousness:.2f} "
        f"E={personality.extraversion:.2f} "
        f"A={personality.agreeableness:.2f} "
        f"N={personality.neuroticism:.2f}\n"
        f"风险规避: {psych.risk_aversion:.2f}\n"
    )

    emotion_str = (
        f"当前情绪: valence={emotion.valence:.2f}, "
        f"arousal={emotion.arousal:.2f}\n"
    )

    # ── 观点状态 ──
    views = []
    for eid, ob in belief.opinions.items():
        views.append(
            f"  - 事件 {eid}: 立场={ob.opinion_value:.2f}, "
            f"置信度={ob.confidence:.2f}"
        )
    opinion_str = "当前观点:\n" + "\n".join(views) if views else "当前观点: (无)"

    # ── 记忆 ──
    mem_lines = []
    for rec in memory[-5:]:  # 只取最近 5 条
        info = rec.info
        mem_lines.append(
            f"  [tick={rec.tick}] agent_{info.source_id} "
            f"{info.action_type.name}: \"{info.content[:60]}\" "
            f"(opinion={info.opinion_value:.2f}, rel={rec.relevance:.2f})"
        )
    mem_str = "近期记忆 (最近 5 条):\n" + "\n".join(mem_lines) if mem_lines else "近期记忆: (空)"

    # ── 环境感知 ──
    env_lines = []
    env_lines.append(f"当前 tick: {env_info.tick}")
    if env_info.trending_topics:
        topics = ", ".join(
            f"{t[0]}(热度={t[1]:.2f})" for t in env_info.trending_topics[:3]
        )
        env_lines.append(f"热门话题: {topics}")
    if env_info.mentions:
        env_lines.append(
            f"被提及 {len(env_info.mentions)} 次"
        )
    env_str = "环境感知:\n" + "\n".join(f"  {line}" for line in env_lines)

    # ── 拼接 ──
    prompt = (
        f"你是一个社交媒体平台上的用户。请根据以下信息, "
        f"决定你下一步的行动。\n\n"
        f"### 角色信息\n{identity_str}\n"
        f"### 心理状态\n{psych_str}\n"
        f"### 情绪状态\n{emotion_str}\n"
        f"### {opinion_str}\n\n"
        f"### {mem_str}\n\n"
        f"### {env_str}\n\n"
        f"### 行动指令\n"
        f"请以 JSON 格式输出你的下一步行动决定:\n"
        f'{{"action_type": "POST|COMMENT|REPOST|LIKE|SILENT", '
        f'"content": "发言内容", '
        f'"event_id": "事件ID", '
        f'"opinion_value": <float in [-1, 1]>, '
        f'"target_id": <目标agent_id 或 null>}}\n\n'
        f"只输出 JSON, 不要包含其他内容。"
    )

    return prompt


# ──────────────────────────────────────────────
# #33 parse_llm_response
# ──────────────────────────────────────────────

def parse_llm_response(raw: str) -> Dict[str, Any]:
    """
    解析 LLM 返回的原始文本, 提取为结构化决策字典。

    支持:
      - 纯净 JSON: {"action_type": "POST", ...}
      - Markdown 包裹: ```json {...} ```
      - 前后多余文本: 自动提取首个 JSON 片段

    Parameters
    ----------
    raw : str
        LLM 返回的原始响应文本。

    Returns
    -------
    Dict[str, Any]
        解析后的字典, 包含 action_type, content, event_id,
        opinion_value, target_id 等字段。
        若解析失败, 返回 SILENT 兜底。

    Notes
    -----
    action_type 会从 str 转换为 ActionType 枚举值。
    """
    import re

    text = raw.strip()

    # ── 尝试 1: 整体解析 ──
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return _normalize_llm_output(data)
    except (json.JSONDecodeError, TypeError):
        pass

    # ── 尝试 2: 提取 Markdown JSON 块 ──
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if md_match:
        try:
            data = json.loads(md_match.group(1).strip())
            if isinstance(data, dict):
                return _normalize_llm_output(data)
        except (json.JSONDecodeError, TypeError):
            pass

    # ── 尝试 3: 查找花括号包裹的 JSON ──
    brace_match = re.search(r"\{[^{}]*\}", text)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            if isinstance(data, dict):
                return _normalize_llm_output(data)
        except (json.JSONDecodeError, TypeError):
            pass

    # ── 兜底: SILENT ──
    _LOG.warning("LLM 响应解析失败, 回退到 SILENT。raw 前 100 字: %s", raw[:100])
    return _default_response()


def _normalize_llm_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """将 LLM 输出的原始 dict 格式化为规范格式。"""
    out = _default_response()

    action_type_str = str(data.get("action_type", "")).upper().strip()
    try:
        out["action_type"] = ActionType[action_type_str]
    except KeyError:
        _LOG.debug("未知 action_type: '%s', 回退到 SILENT", action_type_str)

    if "content" in data:
        out["content"] = str(data["content"])
    if "event_id" in data:
        out["event_id"] = str(data["event_id"])
    if "opinion_value" in data:
        try:
            out["opinion_value"] = float(np.clip(
                float(data["opinion_value"]), -1.0, 1.0
            ))
        except (ValueError, TypeError):
            pass
    if "target_id" in data and data["target_id"] is not None:
        out["target_id"] = int(data["target_id"])

    return out


def _default_response() -> Dict[str, Any]:
    """SILENT 兜底响应。"""
    return {
        "action_type": ActionType.SILENT,
        "content": "",
        "event_id": "",
        "opinion_value": 0.0,
        "target_id": None,
    }


# ──────────────────────────────────────────────
# #34 save_simulation_results
# ──────────────────────────────────────────────

# 16bit 编码映射
_FLOAT_TO_INT16 = 32767    # [-1, 1] → int16: v * 32767
_FLOAT_TO_UINT16 = 65535   # [0, 1]  → uint16: v * 65535

# 应应用 16bit 编码的指标字段名
_METRICS_RANGE_M11 = {       # ∈ [-1, 1], 用 int16 编码
    "avg_opinion", "polarization", "emotional_contagion",
}
_METRICS_RANGE_01 = {        # ∈ [0, 1], 用 uint16 编码
    "participation_rate",
}


def _encode_16bit(col_name: str, value: float) -> int:
    """
    将浮点值按 16bit 编码转换为整数。

    规则:
      - [-1, 1] 范围字段 → int16, v * 32767
      - [0, 1]  范围字段 → uint16, v * 65535
    """
    if col_name in _METRICS_RANGE_M11:
        return int(round(np.clip(value, -1.0, 1.0) * _FLOAT_TO_INT16))
    elif col_name in _METRICS_RANGE_01:
        return int(round(np.clip(value, 0.0, 1.0) * _FLOAT_TO_UINT16))
    # 不在已知列表中的字段, 原样返回
    return int(value) if isinstance(value, (int, float, np.integer, np.floating)) else value


def save_simulation_results(data: pd.DataFrame, path: str,
                            fmt: str = "csv") -> str:
    """
    将时序仿真数据导出为 CSV 或 JSON 文件。

    按数据契约约定, 对度量字段自动应用 16bit 整数编码:
      - avg_opinion / polarization / emotional_contagion → int16 (x32767)
      - participation_rate → uint16 (x65535)

    Parameters
    ----------
    data : pd.DataFrame
        包含时序仿真结果的 DataFrame, 应包含 tick 及各指标列。
    path : str
        输出文件路径 (不含扩展名)。
    fmt : str, optional
        输出格式, "csv" 或 "json" (默认 "csv")。

    Returns
    -------
    str
        实际写入的文件绝对路径。

    Raises
    ------
    ValueError
        fmt 不是 "csv" 或 "json"。
    """
    fmt = fmt.lower()
    if fmt not in ("csv", "json"):
        raise ValueError(f"不支持的导出格式: {fmt} (支持: csv, json)")

    # 深拷贝, 避免修改传入的 DataFrame
    out = data.copy()

    # 对已知指标列应用 16bit 编码
    for col in out.columns:
        if col in _METRICS_RANGE_M11 or col in _METRICS_RANGE_01:
            out[col] = out[col].apply(lambda v: _encode_16bit(col, v))

    out_path = Path(path).with_suffix(f".{fmt}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        out.to_csv(out_path, index=False, encoding="utf-8")
    else:
        out.to_json(out_path, orient="records", force_ascii=False, indent=2)

    logger = get_logger(__name__)
    logger.info("仿真结果已导出: %s (shape=%s)", out_path.resolve(), data.shape)
    return str(out_path.resolve())


# ──────────────────────────────────────────────
# #35 visualize_trend
# ──────────────────────────────────────────────

def visualize_trend(data: pd.DataFrame, metrics: List[str],
                    out_path: str, title: Optional[str] = None) -> str:
    """
    绘制仿真时序指标演化曲线, 保存为图片。

    每列指标作为一条独立曲线, x 轴为 tick, y 轴为指标值。
    如果数据包含 experiment_label 列, 自动按实验对照分组绘制。

    Parameters
    ----------
    data : pd.DataFrame
        时序数据, 必须包含 "tick" 列和 metrics 指定的指标列。
    metrics : List[str]
        要绘制的指标列名列表 (如 ["avg_opinion", "polarization"])。
    out_path : str
        输出图片路径 (不含扩展名, 将保存为 PNG)。
    title : str, optional
        图标题, 不传则自动生成。

    Returns
    -------
    str
        实际写入的图片文件绝对路径。

    Raises
    ------
    KeyError
        data 中缺少 "tick" 列或 metrics 中指定列。
    """
    # ── 校验 ──
    if "tick" not in data.columns:
        raise KeyError("data 必须包含 'tick' 列")

    missing = [c for c in metrics if c not in data.columns]
    if missing:
        raise KeyError(f"data 缺少指标列: {missing}")

    # ── 按实验对照标签分组 ──
    has_label = "experiment_label" in data.columns

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=DEFAULT_FIG_SIZE,
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        if has_label:
            for label, grp in data.groupby("experiment_label"):
                grp_sorted = grp.sort_values("tick")
                ax.plot(grp_sorted["tick"], grp_sorted[metric],
                        label=str(label), marker="", linewidth=1.5)
            ax.legend(fontsize=9)
        else:
            df_sorted = data.sort_values("tick")
            ax.plot(df_sorted["tick"], df_sorted[metric],
                    color="#1f77b4", linewidth=1.5)

        ax.set_ylabel(metric, fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("tick", fontsize=11)

    fig.suptitle(title or "仿真时序指标演化趋势", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # ── 保存 ──
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger = get_logger(__name__)
    logger.info("趋势图已保存: %s", out.resolve())
    return str(out.resolve())


# ──────────────────────────────────────────────
# #36 compute_similarity
# ──────────────────────────────────────────────

# 简易词袋嵌入, 用于无 LLM 环境下的兜底
# 当 sentence-transformers 不可用时自动回退


def _fallback_bow_similarity(text_a: str, text_b: str) -> float:
    """
    基于词袋交并比 (Jaccard) 的简单相似度, 作为兜底方案。
    将文本按空白 + 标点分词, 计算交集 / 并集。

    Returns
    -------
    float
        [0, 1] 区间的 Jaccard 相似度。
    """
    import re
    tokens_a = set(re.findall(r"\w+", text_a.lower()))
    tokens_b = set(re.findall(r"\w+", text_b.lower()))

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def compute_similarity(text_a: str, text_b: str) -> float:
    """
    计算两段文本的余弦相似度。

    优先使用 sentence-transformers 模型嵌入, 不可用时
    自动回退到词袋 Jaccard 相似度。

    Parameters
    ----------
    text_a : str
        第一段文本。
    text_b : str
        第二段文本。

    Returns
    -------
    float
        [0, 1] 区间的相似度 (1 表示完全一致, 0 表示完全不同)。

    Notes
    -----
    如需使用语义嵌入, 安装 sentence-transformers 即可自动启用:
        pip install sentence-transformers
    """
    # ── 尝试 sentence-transformers ──
    try:
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        emb_a = model.encode(text_a, convert_to_tensor=True)
        emb_b = model.encode(text_b, convert_to_tensor=True)
        sim = float(util.pytorch_cos_sim(emb_a, emb_b).item())

        # 归一化到 [0, 1]
        sim = max(0.0, min(1.0, (sim + 1.0) / 2.0))

        logger = get_logger(__name__)
        logger.debug("compute_similarity: 使用 sentence-transformers, "
                      "result=%.4f", sim)
        return sim

    except ImportError:
        pass

    # ── 兜底: Jaccard ──
    sim = _fallback_bow_similarity(text_a, text_b)

    logger = get_logger(__name__)
    logger.debug("compute_similarity: 使用 Jaccard 兜底, result=%.4f", sim)
    return sim


# ──────────────────────────────────────────────
# #37 get_logger
# ──────────────────────────────────────────────

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    获取统一配置的 logger。

    同一 name 返回同一 logger 实例 (带缓存), 避免重复添加 handler。

    Parameters
    ----------
    name : str
        Logger 名称, 通常传入 __name__。
    level : str, optional
        日志级别: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        (默认 "INFO")。

    Returns
    -------
    logging.Logger
        配置好的 Logger 实例。
    """
    if name in LOGGER_CACHE:
        return LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    _level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(_level)

    # ── 只有 root logger 没有 handler 时才添加 ──
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(_level)

        fmt = logging.Formatter(
            "[%(asctime)s] %(name)-24s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.propagate = False
    LOGGER_CACHE[name] = logger
    return logger


# ──────────────────────────────────────────────
# 快捷入口 (直接 python utils.py 可快速测试)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    log = get_logger("utils.demo", "DEBUG")

    # 1. 测试 get_logger
    log.info("get_logger 测试通过")

    # 2. 测试 compute_similarity
    sim = compute_similarity("今天天气真好", "今天天气不错")
    log.info("compute_similarity('今天天气真好', '今天天气不错') = %.4f", sim)

    # 3. 测试 save_simulation_results 与 visualize_trend
    demo_df = pd.DataFrame({
        "tick": list(range(10)),
        "avg_opinion": np.sin(np.linspace(0, 2 * np.pi, 10)),
        "polarization": np.abs(np.random.randn(10)) * 0.3,
        "emotional_contagion": np.random.rand(10) * 2 - 1,
        "participation_rate": np.random.rand(10),
    })

    log.info("demo data shape: %s", demo_df.shape)
    log.info("所有函数就绪 ✅")
