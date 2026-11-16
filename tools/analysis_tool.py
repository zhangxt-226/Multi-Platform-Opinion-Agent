"""analysis_tool.py —— 四维分析工具

对应原项目的 analysis.py，但把“算数据”和“画图”拆开了：
本模块只负责算指标，产出结构化结果（outputs/analysis_result.json）+ 给 LLM 看的文字摘要；
画图交给 chart_tool。

四个维度：
    1) 话题维度：巴勒斯坦/哈马斯、以色列、伊朗、美国、冲突局势、其他
    2) 情感维度：正面 / 中立 / 负面（关键词规则打分，与原项目一致）
    3) 关键词维度：中文 jieba 分词、英文正则切词，各取 TopN
    4) 时间维度：每日讨论量（国内外分开）、爆发期/高峰期/降温期分段
外加平台维度交叉（微博 / 知乎 / YouTube 各维度占比）。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import jieba
import pandas as pd
from langchain_core.tools import tool

from . import ALL_MERGED, ANALYSIS_JSON, OUTPUT_DIR, read_csv_auto

CN_PLATFORMS = {"weibo.com", "zhihu.com"}

# ---------------------------------------------------------------- 词表

TOPIC_RULES: list[tuple[str, set[str]]] = [
    ("巴勒斯坦/哈马斯", {"哈马斯", "hamas", "palestinian", "palestine", "gaza", "加沙"}),
    ("以色列", {"以色列", "以军", "israel", "idf", "israeli", "zionist", "zionists"}),
    ("伊朗", {"伊朗", "革命卫队", "iran", "irgc", "iranian", "德黑兰", "tehran"}),
    ("美国", {"美国", "特朗普", "america", "usa", "trump", "united states", "uss"}),
    ("冲突局势", {"冲突", "战争", "袭击", "空袭", "导弹", "conflict", "war", "attack", "missile"}),
]

POS_WORDS_CN = ["支持", "赞同", "正义", "和平", "合理", "理解", "希望", "祝福", "感谢", "赞", "同情", "克制"]
NEG_WORDS_CN = ["反对", "暴行", "残酷", "侵略", "不满", "谴责", "愤怒", "可怕", "恶心", "讨厌", "活该", "谎言", "霸权"]
POS_WORDS_EN = ["support", "agree", "justice", "peace", "reasonable", "understand", "hope", "bless", "pray", "brave"]
NEG_WORDS_EN = ["oppose", "atrocity", "cruel", "invasion", "condemn", "angry", "horrible", "terrible", "evil", "lie"]

STOPWORDS_CN = {
    "的", "了", "是", "就", "在", "和", "也", "都", "有", "我", "你", "他", "她", "它", "们", "这", "那",
    "一个", "什么", "自己", "可以", "不是", "没有", "还是", "所以", "因为", "但是", "如果", "已经",
    "现在", "真的", "这个", "那个", "我们", "你们", "他们", "觉得", "知道", "应该", "可能", "就是",
}
STOPWORDS_EN = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "need", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "again", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "but", "if", "or", "because", "until", "while",
    "this", "that", "these", "those", "what", "which", "who", "whom", "whose", "my", "your", "his", "her",
    "its", "our", "their", "me", "him", "us", "them", "and", "it", "we", "you", "they", "i", "he", "she",
    "like", "get", "got", "go", "going", "know", "think", "want", "make", "made", "one", "people", "now",
    "video", "youtube", "comment", "comments", "still", "even", "much", "many", "way", "time",
}

PERIODS = ["爆发期", "高峰期", "降温期"]
_LAST_RESULT: dict[str, Any] | None = None


# ---------------------------------------------------------------- 读取


def _parse_datetime(value: Any) -> pd.Timestamp | None:
    """兼容微博/知乎的 'Fri May 29 21:58:08 +0800 2026' 与 YouTube 的 '2026-05-30'。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "nat"):
        return None
    try:
        dt = pd.to_datetime(text, format="%a %b %d %H:%M:%S %z %Y")
        return dt.tz_localize(None) if dt.tzinfo is not None else dt
    except (ValueError, TypeError):
        pass
    try:
        dt = pd.to_datetime(text)
        return dt.tz_localize(None) if getattr(dt, "tzinfo", None) is not None else dt
    except (ValueError, TypeError):
        return None


def load_data(path: str | Path = ALL_MERGED) -> pd.DataFrame:
    df = read_csv_auto(path)
    required = {"website", "text_raw", "id", "screen_name", "like_counts", "created_at"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"数据缺少必需列：{sorted(missing)}")

    df["text_raw"] = df["text_raw"].astype(str).str.replace(r"<.*?>", "", regex=True).str.strip()
    df = df[df["text_raw"].str.len() > 0].copy()

    df["date"] = df["created_at"].apply(_parse_datetime)
    df = df[df["date"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["like_counts"] = pd.to_numeric(df["like_counts"], errors="coerce").fillna(0).astype(int)
    df["platform"] = df["website"].astype(str)
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------- 四维计算


def classify_topic(text: str) -> str:
    lowered = text.lower()
    for label, words in TOPIC_RULES:
        if any(w in text or w in lowered for w in words):
            return label
    return "其他"


def classify_sentiment(text: str) -> str:
    """关键词计数法（可解释、无模型依赖），正负持平记中立。"""
    lowered = text.lower()
    pos = sum(1 for w in POS_WORDS_CN if w in text) + sum(1 for w in POS_WORDS_EN if w in lowered)
    neg = sum(1 for w in NEG_WORDS_CN if w in text) + sum(1 for w in NEG_WORDS_EN if w in lowered)
    if pos > neg:
        return "正面"
    if neg > pos:
        return "负面"
    return "中立"


def extract_keywords(text: str, is_chinese: bool) -> list[str]:
    if is_chinese:
        words = [w.strip() for w in jieba.cut(text)]
        return [w for w in words if len(w) >= 2 and w not in STOPWORDS_CN and not re.fullmatch(r"\d+", w)]
    words = re.findall(r"[a-zA-Z]{2,}", text.lower())
    return [w for w in words if w not in STOPWORDS_EN]


def _ratio(counter: dict[str, int], total: int) -> dict[str, dict[str, Any]]:
    return {
        k: {"count": int(v), "ratio": round(v / total * 100, 1) if total else 0.0}
        for k, v in sorted(counter.items(), key=lambda kv: -kv[1])
    }


def _period_stats(df: pd.DataFrame) -> dict[str, Any]:
    """按时间顺序切三段：前 25% 爆发期、25%-60% 高峰期、其余降温期（与原项目一致）。"""
    if df.empty:
        return {}
    ordered = df.sort_values("date").reset_index(drop=True)
    n = len(ordered)
    burst_end, peak_end = int(n * 0.25), int(n * 0.6)

    labels = ["爆发期"] * burst_end + ["高峰期"] * (peak_end - burst_end) + ["降温期"] * (n - peak_end)
    ordered["period"] = labels

    result: dict[str, Any] = {}
    for period in PERIODS:
        chunk = ordered[ordered["period"] == period]
        if chunk.empty:
            result[period] = {"count": 0}
            continue
        senti = chunk["sentiment"].value_counts().to_dict()
        topic = chunk["topic"].value_counts().to_dict()
        result[period] = {
            "count": int(len(chunk)),
            "date_range": [chunk["date"].min().strftime("%Y-%m-%d"), chunk["date"].max().strftime("%Y-%m-%d")],
            "sentiment": {k: int(senti.get(k, 0)) for k in ("正面", "中立", "负面")},
            "negative_ratio": round(senti.get("负面", 0) / len(chunk) * 100, 1),
            "top_topic": max(topic.items(), key=lambda kv: kv[1])[0] if topic else "无",
            "topic": {k: int(v) for k, v in topic.items()},
        }
    return result


def analyze(df: pd.DataFrame) -> dict[str, Any]:
    df = df.copy()
    df["topic"] = df["text_raw"].apply(classify_topic)
    df["sentiment"] = df["text_raw"].apply(classify_sentiment)
    df["is_cn"] = df["platform"].isin(CN_PLATFORMS)
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    total = len(df)
    platform_counts = df["platform"].value_counts().to_dict()

    # —— 维度 1：话题
    topic_counts = df["topic"].value_counts().to_dict()
    topic_by_platform = {
        str(platform): {k: int(v) for k, v in chunk["topic"].value_counts().items()}
        for platform, chunk in df.groupby("platform")
    }

    # —— 维度 2：情感
    senti_counts = df["sentiment"].value_counts().to_dict()
    senti_by_platform = {
        str(platform): {
            "正面": int((chunk["sentiment"] == "正面").sum()),
            "中立": int((chunk["sentiment"] == "中立").sum()),
            "负面": int((chunk["sentiment"] == "负面").sum()),
            "负面占比": round((chunk["sentiment"] == "负面").mean() * 100, 1),
        }
        for platform, chunk in df.groupby("platform")
    }
    senti_x_topic = {
        str(topic): {k: int(v) for k, v in chunk["sentiment"].value_counts().items()}
        for topic, chunk in df.groupby("topic")
    }

    # —— 维度 3：关键词
    cn_df, en_df = df[df["is_cn"]], df[~df["is_cn"]]
    cn_words, en_words = Counter(), Counter()
    for text in cn_df["text_raw"]:
        cn_words.update(extract_keywords(text, True))
    for text in en_df["text_raw"]:
        en_words.update(extract_keywords(text, False))
    keywords_by_platform = {}
    for platform, chunk in df.groupby("platform"):
        is_cn = platform in CN_PLATFORMS
        counter = Counter()
        for text in chunk["text_raw"]:
            counter.update(extract_keywords(text, is_cn))
        keywords_by_platform[str(platform)] = [[w, int(c)] for w, c in counter.most_common(30)]

    # —— 维度 4：时间
    cn_daily = cn_df.groupby("date_str").size().to_dict()
    en_daily = en_df.groupby("date_str").size().to_dict()
    all_dates = sorted(set(cn_daily) | set(en_daily))
    daily = [
        {"date": d, "cn": int(cn_daily.get(d, 0)), "en": int(en_daily.get(d, 0)),
         "total": int(cn_daily.get(d, 0) + en_daily.get(d, 0))}
        for d in all_dates
    ]
    peak = max(daily, key=lambda x: x["total"]) if daily else None

    # —— 附加：高赞样本（给 LLM 写结论用）
    samples = []
    for platform, chunk in df.groupby("platform"):
        top = chunk.nlargest(3, "like_counts")
        for _, row in top.iterrows():
            samples.append(
                {
                    "platform": str(platform),
                    "likes": int(row["like_counts"]),
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "topic": row["topic"],
                    "sentiment": row["sentiment"],
                    "text": row["text_raw"][:160],
                }
            )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": ALL_MERGED.name,
        "overview": {
            "total": int(total),
            "platforms": {str(k): int(v) for k, v in platform_counts.items()},
            "date_range": [df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")],
            "days": int(df["date_str"].nunique()),
            "avg_daily": round(total / max(df["date_str"].nunique(), 1), 1),
            "total_likes": int(df["like_counts"].sum()),
            "avg_text_len": round(df["text_raw"].str.len().mean(), 1),
        },
        "topic": {"overall": _ratio(topic_counts, total), "by_platform": topic_by_platform},
        "sentiment": {
            "overall": _ratio(senti_counts, total),
            "by_platform": senti_by_platform,
            "by_topic": senti_x_topic,
        },
        "keywords": {
            "cn_top50": [[w, int(c)] for w, c in cn_words.most_common(50)],
            "en_top50": [[w, int(c)] for w, c in en_words.most_common(50)],
            "by_platform": keywords_by_platform,
        },
        "time": {"daily": daily, "peak_day": peak, "periods": _period_stats(df)},
        "samples": samples,
    }


# ---------------------------------------------------------------- 摘要文本


def build_summary(result: dict[str, Any], top_n: int = 5) -> str:
    """把结果压成 LLM 友好、人类也读得懂的文字摘要。"""
    ov = result["overview"]
    lines = [
        "【数据概览】",
        f"总评论 {ov['total']:,} 条，时间跨度 {ov['date_range'][0]} ~ {ov['date_range'][1]}（{ov['days']} 天，"
        f"日均 {ov['avg_daily']} 条），总点赞 {ov['total_likes']:,}，平均文本长度 {ov['avg_text_len']} 字。",
        "平台分布：" + "、".join(f"{k} {v:,} 条" for k, v in ov["platforms"].items()),
        "",
        "【维度一：话题分布】",
    ]
    for topic, info in result["topic"]["overall"].items():
        lines.append(f"  - {topic}: {info['count']:,} 条（{info['ratio']}%）")

    lines += ["", "【维度二：情感分布】"]
    for senti, info in result["sentiment"]["overall"].items():
        lines.append(f"  - {senti}: {info['count']:,} 条（{info['ratio']}%）")
    lines.append("  分平台负面占比：" + "、".join(
        f"{p} {v['负面占比']}%" for p, v in result["sentiment"]["by_platform"].items()))

    lines += ["", "【维度三：高频关键词】"]
    cn = result["keywords"]["cn_top50"][:top_n]
    en = result["keywords"]["en_top50"][:top_n]
    lines.append("  中文平台：" + "、".join(f"{w}({c})" for w, c in cn))
    lines.append("  海外平台：" + "、".join(f"{w}({c})" for w, c in en))

    lines += ["", "【维度四：时间演化】"]
    peak = result["time"]["peak_day"]
    if peak:
        lines.append(f"  讨论峰值日：{peak['date']}，共 {peak['total']:,} 条（中文 {peak['cn']} / 海外 {peak['en']}）")
    for period, info in result["time"]["periods"].items():
        if info.get("count"):
            lines.append(
                f"  - {period}（{info['date_range'][0]}~{info['date_range'][1]}）：{info['count']:,} 条，"
                f"负面占比 {info['negative_ratio']}%，最热话题 {info['top_topic']}"
            )

    if result.get("samples"):
        lines += ["", "【高赞样本（节选）】"]
        for s in result["samples"][:6]:
            lines.append(f"  - [{s['platform']}|{s['likes']}赞|{s['topic']}|{s['sentiment']}] {s['text']}")
    return "\n".join(lines)


@tool
def analyze_data(detail: str = "full") -> str:
    """对三平台合并数据做四维分析（话题 / 情感 / 关键词 / 时间），并把结构化结果写入 outputs。

    前置条件：Comments/merged/三个平台数据合并结果.csv 已存在（merge_all_platforms 的产物）。

    Args:
        detail: "full" 返回完整摘要（默认）；"brief" 只返回核心数字。

    Returns:
        分析摘要文本（同时落盘 outputs/analysis_result.json）。
    """
    global _LAST_RESULT
    try:
        df = load_data()
    except FileNotFoundError:
        return f"分析未执行：未找到 {ALL_MERGED}，请先执行 merge_all_platforms。"
    except Exception as exc:
        return f"分析失败（{type(exc).__name__}: {exc}）"

    if df.empty:
        return "分析未执行：清洗后没有可用数据（日期或正文全部为空）。"

    result = analyze(df)
    _LAST_RESULT = result
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = build_summary(result, top_n=5 if detail != "brief" else 3)
    tail = f"\n\n结构化结果已保存：{ANALYSIS_JSON}"
    return (summary + tail) if detail != "brief" else ("\n".join(summary.splitlines()[:10]) + tail)


@tool
def save_report(content: str, filename: str = "") -> str:
    """把分析结论保存成 Markdown 报告（outputs/ 目录）。

    Args:
        content: 报告正文（Markdown）。
        filename: 文件名，默认按时间戳生成，如 舆情分析报告_20261116_1700.md。

    Returns:
        保存路径。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = filename.strip() or f"舆情分析报告_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    if not name.lower().endswith((".md", ".txt")):
        name += ".md"
    path = OUTPUT_DIR / name
    path.write_text(content, encoding="utf-8")
    return f"报告已保存：{path}（{len(content)} 字）"


def get_last_result() -> dict[str, Any] | None:
    """给 chart_tool 用：优先内存缓存，其次读取落盘 JSON，最后重新计算。"""
    global _LAST_RESULT
    if _LAST_RESULT is not None:
        return _LAST_RESULT
    if ANALYSIS_JSON.exists():
        try:
            _LAST_RESULT = json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))
            return _LAST_RESULT
        except (json.JSONDecodeError, OSError):
            pass
    try:
        df = load_data()
        if df.empty:
            return None
        _LAST_RESULT = analyze(df)
        return _LAST_RESULT
    except Exception:
        return None


ANALYSIS_TOOLS = [analyze_data, save_report]
