"""merge_tool.py —— 合并工具

整合原项目的 combine_weibo.py / combine_zhihu.py / combine_youtube.py / combineAll.py：

    merge_platform("weibo")   Comments/weibo/*.csv   -> Comments/merged/微博话题_合并版.csv
    merge_platform("zhihu")   Comments/zhihu/*.csv   -> Comments/merged/知乎话题_合并版.csv
    merge_platform("youtube") Comments/youtube/youtube.csv -> Comments/merged/Youtube话题_合并版.csv
    merge_all_platforms()     三个平台的清洗结果 -> Comments/merged/三个平台数据合并结果.csv

三平台合并会把 YouTube 的字段名映射成统一 schema：
    platform->website, content->text_raw, source_id->id,
    user_name->screen_name, like_count->like_counts, published_at->created_at
"""

from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool

from . import (
    ALL_MERGED,
    WEIBO_CLEAN,
    WEIBO_DIR,
    WEIBO_MERGED,
    YOUTUBE_CLEAN,
    YOUTUBE_MERGED,
    YOUTUBE_RAW,
    ZHIHU_CLEAN,
    ZHIHU_DIR,
    ZHIHU_MERGED,
    ensure_dirs,
    format_table,
    read_csv_auto,
    write_csv,
)

UNIFIED_COLUMNS = ["website", "text_raw", "id", "screen_name", "like_counts", "created_at"]

YOUTUBE_COLUMN_MAPPING = {
    "platform": "website",
    "content": "text_raw",
    "source_id": "id",
    "user_name": "screen_name",
    "like_count": "like_counts",
    "published_at": "created_at",
}


def _concat_folder(folder, dedupe_key: str | None = None) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    frames = []
    for file in files:
        try:
            frames.append(read_csv_auto(file))
        except Exception:
            continue
    if not frames:
        raise FileNotFoundError(f"{folder} 下没有可读取的 CSV")
    df = pd.concat(frames, ignore_index=True)
    if dedupe_key and dedupe_key in df.columns:
        df = df.drop_duplicates(subset=[dedupe_key], keep="first")
    return df


def merge_weibo() -> dict:
    ensure_dirs()
    df = _concat_folder(WEIBO_DIR, dedupe_key="id")
    write_csv(df, WEIBO_MERGED)
    return {"平台": "微博", "文件数": len(list(WEIBO_DIR.glob("*.csv"))), "输出": WEIBO_MERGED.name, "条数": len(df)}


def merge_zhihu() -> dict:
    ensure_dirs()
    df = _concat_folder(ZHIHU_DIR, dedupe_key="id")
    write_csv(df, ZHIHU_MERGED)
    return {"平台": "知乎", "文件数": len(list(ZHIHU_DIR.glob("*.csv"))), "输出": ZHIHU_MERGED.name, "条数": len(df)}


def merge_youtube() -> dict:
    ensure_dirs()
    if not YOUTUBE_RAW.exists():
        raise FileNotFoundError(f"未找到 {YOUTUBE_RAW.name}，请先运行 crawl_youtube")
    df = read_csv_auto(YOUTUBE_RAW)
    if "source_id" in df.columns:
        df = df.drop_duplicates(subset=["source_id"], keep="first")
    write_csv(df, YOUTUBE_MERGED)
    return {"平台": "YouTube", "文件数": 1, "输出": YOUTUBE_MERGED.name, "条数": len(df)}


def merge_all() -> dict:
    """三个平台的清洗结果 -> 统一 schema 的大表。"""
    ensure_dirs()
    missing = [p.name for p in (WEIBO_CLEAN, ZHIHU_CLEAN, YOUTUBE_CLEAN) if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少清洗结果：" + "、".join(missing) + "，请先执行 clean_data")

    df_weibo = read_csv_auto(WEIBO_CLEAN)
    df_zhihu = read_csv_auto(ZHIHU_CLEAN)
    df_youtube = read_csv_auto(YOUTUBE_CLEAN)

    available = [c for c in YOUTUBE_COLUMN_MAPPING if c in df_youtube.columns]
    df_youtube = df_youtube[available].rename(columns=YOUTUBE_COLUMN_MAPPING)

    df = pd.concat([df_weibo, df_zhihu, df_youtube], ignore_index=True)
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[UNIFIED_COLUMNS]
    write_csv(df, ALL_MERGED)

    dist = df["website"].value_counts().to_dict()
    return {"输出": ALL_MERGED.name, "总条数": len(df), "分布": dist}


@tool
def merge_platform(platform: str = "all") -> str:
    """把分话题的原始 CSV 合并成单个平台的合并表。

    Args:
        platform: "weibo" / "zhihu" / "youtube" / "all"（默认 all）。

    Returns:
        合并结果摘要（文件数、条数、输出文件）。
    """
    platform = (platform or "all").lower().strip()
    jobs = {"weibo": merge_weibo, "zhihu": merge_zhihu, "youtube": merge_youtube}
    targets = ["weibo", "zhihu", "youtube"] if platform in ("all", "全部") else [platform]
    if any(t not in jobs for t in targets):
        return f"不支持的平台：{platform}，可选 weibo / zhihu / youtube / all。"

    rows, errors = [], []
    for name in targets:
        try:
            r = jobs[name]()
            rows.append([r["平台"], r["文件数"], f"{r['条数']:,}", r["输出"]])
        except FileNotFoundError as exc:
            errors.append(f"{name}: {exc}")
        except Exception as exc:
            errors.append(f"{name}: 合并失败（{type(exc).__name__}: {exc}）")

    if not rows:
        return "合并未执行：\n" + "\n".join(errors)
    text = "平台内合并完成：\n" + format_table(rows, ["平台", "输入文件数", "合并后条数", "输出文件"])
    if errors:
        text += "\n\n未完成项：\n" + "\n".join(errors)
    return text


@tool
def merge_all_platforms() -> str:
    """把三个平台（微博/知乎/YouTube）的清洗结果合并成统一 schema 的总表，供分析使用。

    Returns:
        合并结果摘要（总条数 + 各平台分布）。
    """
    try:
        result = merge_all()
    except FileNotFoundError as exc:
        return f"三平台合并未执行：{exc}"
    except Exception as exc:
        return f"三平台合并失败（{type(exc).__name__}: {exc}）"

    lines = [f"三平台合并完成：共 {result['总条数']:,} 条 -> {result['输出']}"]
    for site, n in sorted(result["分布"].items(), key=lambda kv: -kv[1]):
        share = n / result["总条数"] * 100 if result["总条数"] else 0
        lines.append(f"  - {site}: {n:,} 条（{share:.1f}%）")
    return "\n".join(lines)


MERGE_TOOLS = [merge_platform, merge_all_platforms]
