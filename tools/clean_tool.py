"""clean_tool.py —— 清洗工具

整合原项目的 weibo_clean.py / zhihu_clean.py / youtube_clean.py：
    1) 按正文去重
    2) 丢掉空文本
    3) 语言比例过滤（中文平台保留中文占比 ≥30%，YouTube 保留英文占比 ≥30%）
    4) 关键词命中筛选（美伊冲突相关词表）

输入：Comments/merged/*_合并版.csv
输出：Comments/merged/微博话题_关键字筛选.csv / 知乎话题_关键字筛选.csv / YouTube清洗分析版.csv
"""

from __future__ import annotations

import re

import pandas as pd
from langchain_core.tools import tool

from . import (
    WEIBO_CLEAN,
    WEIBO_MERGED,
    YOUTUBE_CLEAN,
    YOUTUBE_MERGED,
    YOUTUBE_RAW,
    ZHIHU_CLEAN,
    ZHIHU_MERGED,
    format_table,
    read_csv_auto,
    write_csv,
)

# 中文平台关键词（沿用原 weibo_clean.py 的词表）
KEYWORDS_CN = [
    '中东局势', '美以伊冲突', '美伊局势', '伊朗袭击以色列',
    '以色列空袭伊朗', '美伊谈判', '伊',
    '中东', '伊朗', '以色列', '美国', '伊以', '美伊',
    '冲突', '局势', '战争', '谈判', '核', '核问题',
    '袭击', '空袭', '轰炸', '哈马斯', '真主党', '巴以',
    '制裁', '石油', '霍尔木兹', '革命卫队', '内塔尼亚胡', '哈梅内伊',
    '支持', '反对', '赞成', '抵制', '讨厌', '喜欢',
    '正义', '邪恶', '侵略', '反人类', '恐怖', '霸权',
    '和平', '反战', '战争贩子', '霸权主义', '活该', '离谱',
    '可怕', '愤怒', '失望', '应该', '必须', '肯定',
    '呼吁', '要求', '讨论', '追踪', '最新', '看法', '对抗', '绝了', '稳定', '动荡',
    '仁慈', '善良', '承诺', '谎言', '虚伪', '乐观', '中立', '站队', '吐槽', '解读',
    '观点', '态度', '失控', '客观', '包容', '残忍', '非正义', '合法', '违约', '欺骗', '辟谣', '表态',
]

# 英文平台关键词（沿用原 youtube_clean.py 的词表）
KEYWORDS_EN = [
    'middle east', 'iran', 'israel', 'us', 'america', 'conflict',
    'war', 'attack', 'missile', 'strike', 'nuclear', 'sanction',
    'peace', 'negotiation', 'tension', 'bombing', 'air strike',
    'hamas', 'hezbollah', 'palestine', 'gaza', 'tehran', 'tel aviv',
    'iron dome', 'military', 'defense', 'retaliation', 'diplomacy',
    'support', 'oppose', 'condemn', 'justice', 'atrocity', 'cruel',
    'invasion', 'aggression', 'peaceful', 'anti-war', 'hegemony',
]

MIN_LANGUAGE_RATIO = 0.3


def _clean_frame(df: pd.DataFrame, text_col: str, lang: str, keywords: list[str]) -> tuple[pd.DataFrame, dict]:
    """通用清洗流程，返回 (清洗后数据, 各步骤计数)。"""
    stats = {"原始": len(df)}

    df = df.drop_duplicates(subset=[text_col], keep="first")
    stats["去重后"] = len(df)

    df = df[df[text_col].astype(str).str.strip() != ""]
    stats["去空后"] = len(df)

    def lang_ratio_ok(text: str) -> bool:
        text = str(text)
        if lang == "zh":
            hits = re.findall(r"[\u4e00-\u9fa5]", text)
        else:
            hits = re.findall(r"[a-zA-Z]", text)
        total = len(text.replace(" ", ""))
        if total == 0:
            return False
        return len(hits) / total >= MIN_LANGUAGE_RATIO

    df = df[df[text_col].apply(lang_ratio_ok)]
    stats["语言过滤后"] = len(df)

    if lang == "zh":
        df = df[df[text_col].astype(str).apply(lambda t: any(k in t for k in keywords))]
    else:
        df = df[df[text_col].astype(str).str.lower().apply(lambda t: any(k in t for k in keywords))]
    stats["关键词筛选后"] = len(df)

    return df, stats


def clean_weibo() -> dict:
    df = read_csv_auto(WEIBO_MERGED)
    cleaned, stats = _clean_frame(df, "text_raw", "zh", KEYWORDS_CN)
    write_csv(cleaned, WEIBO_CLEAN)
    return {"平台": "微博", "输出": WEIBO_CLEAN.name, **stats}


def clean_zhihu() -> dict:
    df = read_csv_auto(ZHIHU_MERGED)
    cleaned, stats = _clean_frame(df, "text_raw", "zh", KEYWORDS_CN)
    write_csv(cleaned, ZHIHU_CLEAN)
    return {"平台": "知乎", "输出": ZHIHU_CLEAN.name, **stats}


def clean_youtube() -> dict:
    source = YOUTUBE_MERGED if YOUTUBE_MERGED.exists() else YOUTUBE_RAW
    df = read_csv_auto(source)
    cleaned, stats = _clean_frame(df, "content", "en", KEYWORDS_EN)
    write_csv(cleaned, YOUTUBE_CLEAN)
    return {"平台": "YouTube", "输出": YOUTUBE_CLEAN.name, **stats}


@tool
def clean_data(platform: str = "all") -> str:
    """清洗并筛选评论数据（去重 + 语言过滤 + 关键词筛选）。

    前置条件：对应平台已完成合并（Comments/merged/xxx_合并版.csv 存在）。

    Args:
        platform: "weibo" / "zhihu" / "youtube" / "all"（默认 all，依次清洗三个平台）。

    Returns:
        各步骤剩余条数汇总表。
    """
    platform = (platform or "all").lower().strip()
    jobs = {"weibo": clean_weibo, "zhihu": clean_zhihu, "youtube": clean_youtube}

    if platform in ("all", "全部"):
        targets = ["weibo", "zhihu", "youtube"]
    elif platform in jobs:
        targets = [platform]
    else:
        return f"不支持的平台：{platform}，可选 weibo / zhihu / youtube / all。"

    results, errors = [], []
    for name in targets:
        try:
            results.append(jobs[name]())
        except FileNotFoundError as exc:
            errors.append(f"{name}: 缺少输入文件（{exc}），请先执行 merge_platform")
        except Exception as exc:
            errors.append(f"{name}: 清洗失败（{type(exc).__name__}: {exc}）")

    if not results:
        return "清洗未执行：\n" + "\n".join(errors)

    header = ["平台", "原始", "去重后", "去空后", "语言过滤后", "关键词筛选后", "输出"]
    rows = [[r["平台"], r["原始"], r["去重后"], r["去空后"], r["语言过滤后"], r["关键词筛选后"], r["输出"]]
            for r in results]
    text = "数据清洗完成：\n" + format_table(rows, header)
    if errors:
        text += "\n\n未完成项：\n" + "\n".join(errors)
    return text


CLEAN_TOOLS = [clean_data]
