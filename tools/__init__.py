"""tools 包：把原 US-Iran Conflict 项目的脚本，重构成 Agent 可调用的工具集。

目录约定（所有读写都限制在 BASE_DIR 内部，绝不写到项目外）：
    Comments/            原始 CSV 数据与中间产物（原项目 Comments 目录的数据，只读写不改结构）
        weibo/           微博分话题原始 CSV
        zhihu/           知乎分话题原始 CSV
        youtube/         YouTube 原始 CSV
        merged/          合并 / 清洗后的中间 CSV
    outputs/             看板 HTML、分析结果 JSON、报告等交付物

五个工具模块：
    crawl_tool      采集：微博 / 知乎 / YouTube
    clean_tool      清洗：去重、语言比例过滤、关键词筛选
    merge_tool      合并：分话题合并 + 三平台合并
    analysis_tool   四维分析：话题 / 情感 / 关键词 / 时间 + 报告落盘
    chart_tool      绘图：生成可视化看板 HTML
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------- 路径常量

BASE_DIR = Path(__file__).resolve().parents[1]
COMMENTS_DIR = BASE_DIR / "Comments"
OUTPUT_DIR = BASE_DIR / "outputs"

WEIBO_DIR = COMMENTS_DIR / "weibo"
ZHIHU_DIR = COMMENTS_DIR / "zhihu"
YOUTUBE_DIR = COMMENTS_DIR / "youtube"
MERGED_DIR = COMMENTS_DIR / "merged"

# 中间产物文件名（沿用原项目命名，方便和旧代码对照）
WEIBO_MERGED = MERGED_DIR / "微博话题_合并版.csv"
WEIBO_CLEAN = MERGED_DIR / "微博话题_关键字筛选.csv"
ZHIHU_MERGED = MERGED_DIR / "知乎话题_合并版.csv"
ZHIHU_CLEAN = MERGED_DIR / "知乎话题_关键字筛选.csv"
YOUTUBE_RAW = YOUTUBE_DIR / "youtube.csv"
YOUTUBE_MERGED = MERGED_DIR / "Youtube话题_合并版.csv"
YOUTUBE_CLEAN = MERGED_DIR / "YouTube清洗分析版.csv"
ALL_MERGED = MERGED_DIR / "三个平台数据合并结果.csv"

ANALYSIS_JSON = OUTPUT_DIR / "analysis_result.json"
DASHBOARD_HTML = OUTPUT_DIR / "评论分析看板.html"

ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb18030")


def ensure_dirs() -> None:
    """确保数据目录与输出目录存在。"""
    for d in (COMMENTS_DIR, OUTPUT_DIR, WEIBO_DIR, ZHIHU_DIR, YOUTUBE_DIR, MERGED_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- CSV 读写


def read_csv_auto(path: str | Path, **kwargs) -> pd.DataFrame:
    """读取 CSV，自动尝试常见编码（原项目里微博数据是 GBK、其它是 UTF-8-BOM）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    last_error: Exception | None = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
            continue
        # 列名里出现替换字符说明编码选错了，换下一个继续试
        if any("\ufffd" in str(c) for c in df.columns):
            last_error = UnicodeDecodeError("utf-8", b"", 0, 1, "列名乱码")
            continue
        return df
    raise RuntimeError(f"无法解码 CSV：{path}（最后一次错误：{last_error}）")


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """统一以 utf-8-sig 写出，Excel 打开不乱码。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def count_rows(path: str | Path) -> int:
    """快速统计 CSV 行数（不含表头），文件不存在返回 0。"""
    path = Path(path)
    if not path.exists():
        return 0
    try:
        with open(path, encoding="utf-8-sig", errors="ignore") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return 0


def human_size(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return "-"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def format_table(rows: Iterable[Iterable[object]], headers: Iterable[str]) -> str:
    """把二维数据排成纯文本表格，方便塞进 LLM 的上下文。"""
    headers = [str(h) for h in headers]
    data = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)) for row in data)
    return f"{line}\n{sep}\n{body}" if body else f"{line}\n{sep}"


# ---------------------------------------------------------------- 工具注册


def register_all_tools() -> list:
    """返回交给 LangChain Agent 的全部工具（按 采集 -> 清洗 -> 合并 -> 分析 -> 绘图 排序）。"""
    from .crawl_tool import CRAWL_TOOLS
    from .clean_tool import CLEAN_TOOLS
    from .merge_tool import MERGE_TOOLS
    from .analysis_tool import ANALYSIS_TOOLS
    from .chart_tool import CHART_TOOLS

    return [*CRAWL_TOOLS, *CLEAN_TOOLS, *MERGE_TOOLS, *ANALYSIS_TOOLS, *CHART_TOOLS]


ensure_dirs()
