"""crawl_tool.py —— 采集工具

把原项目里的三个爬虫（crawler_weibo.py / crawler_zhihu.py / crawler_youtube.py）
整理成三个 Agent 工具：

    crawl_weibo   微博：话题搜索页 -> 每条微博的评论接口
    crawl_zhihu   知乎：搜索接口 -> 回答的 root_comments 接口
    crawl_youtube YouTube：搜索结果 -> watch 页 + innertube 评论续页

凭证全部从 .env 读取（WEIBO_COOKIE / ZHIHU_COOKIE / YOUTUBE_PROXY），
没有配置时工具返回提示文案而不是抛异常，保证 Agent 不会因为缺 key 直接崩。

落盘位置：Comments/weibo/*.csv、Comments/zhihu/*.csv、Comments/youtube/youtube.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

from . import BASE_DIR, WEIBO_DIR, YOUTUBE_RAW, ZHIHU_DIR, ensure_dirs

load_dotenv(BASE_DIR / ".env")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)

# 原项目里写死的话题 / 搜索词，作为默认值保留
DEFAULT_WEIBO_TOPICS = ["#美以伊冲突#", "#中东局势#", "#美伊战争#", "#美伊谈判#", "#伊朗核问题#", "#以色列空袭伊朗#"]
DEFAULT_ZHIHU_TOPICS = ["中东局势", "美以伊冲突", "美伊局势", "伊朗袭击以色列", "以色列空袭伊朗", "美伊谈判"]
DEFAULT_YOUTUBE_QUERIES = [
    "Iran Israel war",
    "US Israel Iran conflict",
    "Iran missile attack Israel",
    "Iran Israel ceasefire",
]

CSV_HEADER = ["website", "text_raw", "id", "screen_name", "like_counts", "created_at"]


def _split_csv_arg(value: str, default: list[str]) -> list[str]:
    """把 Agent 传来的 "a,b,c" 拆成列表；空值用默认话题。"""
    if not value or not value.strip():
        return list(default)
    parts = [p.strip() for p in re.split(r"[,，;；\n]", value) if p.strip()]
    return parts or list(default)


def _safe_name(topic: str) -> str:
    return re.sub(r'[\\/:*?"<>|#\s]+', "_", topic).strip("_") or "topic"


# ======================================================================
# 微博
# ======================================================================


class WeiboCrawler:
    """微博评论采集：搜索页拿微博链接 -> statuses/buildComments 逐页取评论。"""

    platform = "weibo.com"

    def __init__(self, cookie: str, xsrf: str, uid: str = "3266943013", delay: float = 0.8):
        self.cookie = cookie
        self.xsrf = xsrf
        self.uid = uid
        self.delay = delay
        self.page_headers = {
            "Cookie": cookie,
            "user-agent": USER_AGENT,
        }
        self.api_headers = {
            "user-agent": USER_AGENT,
            "x-requested-with": "XMLHttpRequest",
            "referer": "https://weibo.com/",
            "cookie": cookie,
            "x-xsrf-token": xsrf,
        }
        self.total = 0

    # -- 内部请求 ------------------------------------------------------
    def _get_article_id(self, mid: str) -> int:
        try:
            resp = requests.get(
                "https://weibo.com/ajax/statuses/show",
                headers=self.api_headers,
                params={"id": mid},
                timeout=15,
            )
            if resp.status_code != 200:
                return 0
            return int(resp.json().get("id") or 0)
        except (requests.RequestException, ValueError):
            return 0

    def _fetch_comment_page(self, params: dict[str, Any], writer: csv.DictWriter) -> Optional[str]:
        try:
            resp = requests.get(
                "https://weibo.com/ajax/statuses/buildComments",
                headers=self.api_headers,
                params=params,
                timeout=20,
            )
            if resp.status_code != 200:
                return None
            payload = resp.json()
        except (requests.RequestException, ValueError):
            return None

        data_list = payload.get("data") or []
        if not data_list:
            return None

        for item in data_list:
            try:
                writer.writerow(
                    {
                        "website": self.platform,
                        "text_raw": item.get("text_raw", ""),
                        "id": item.get("id", ""),
                        "screen_name": (item.get("user") or {}).get("screen_name", ""),
                        "like_counts": item.get("like_counts", 0),
                        "created_at": item.get("created_at", ""),
                    }
                )
                self.total += 1
            except (csv.Error, AttributeError):
                continue

        max_id = payload.get("max_id")
        return str(max_id) if max_id else None

    def _fetch_post_comments(self, mid: str, writer: csv.DictWriter, max_pages: int) -> None:
        params = {
            "is_reload": 1,
            "id": mid,
            "is_show_bulletin": 2,
            "is_mix": 0,
            "count": 20,
            "uid": int(self.uid) if str(self.uid).isdigit() else 0,
        }
        empty_rounds = 0
        for _ in range(max_pages):
            max_id = self._fetch_comment_page(params, writer)
            if not max_id:
                empty_rounds += 1
                if empty_rounds >= 2:
                    break
                continue
            empty_rounds = 0
            params["max_id"] = max_id
            time.sleep(self.delay)

    # -- 对外入口 ------------------------------------------------------
    def crawl_topic(self, topic: str, max_pages: int = 3, posts_per_page: int = 10) -> Path:
        """爬一个话题，返回写入的文件路径。"""
        ensure_dirs()
        out_file = WEIBO_DIR / f"{_safe_name(topic)}.csv"
        new_file = not out_file.exists()

        with open(out_file, "a", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, CSV_HEADER)
            if new_file:
                writer.writeheader()

            for page in range(1, max_pages + 1):
                url = f"https://s.weibo.com/weibo?q={quote(topic)}&Refer=index&page={page}"
                try:
                    resp = requests.get(url, headers=self.page_headers, timeout=20)
                except requests.RequestException:
                    break
                if resp.status_code != 200:
                    break

                # 搜索页里的微博链接：weibo.com/<uid>/<mid>?refer_flag=...
                mids = re.findall(r"weibo\.com/\d+/([A-Za-z0-9]+)\?refer_flag", resp.text)
                if not mids:
                    break

                for mid in dict.fromkeys(mids)[:posts_per_page]:
                    self.api_headers["referer"] = f"https://weibo.com/2656274875/{mid}"
                    article_id = self._get_article_id(mid) or mid
                    self._fetch_post_comments(article_id, writer, max_pages=2)
                time.sleep(self.delay)

        return out_file


# ======================================================================
# 知乎
# ======================================================================


class ZhihuCrawler:
    """知乎评论采集：search_v3 找回答 -> answers/{id}/root_comments 翻页。"""

    platform = "zhihu.com"

    def __init__(self, cookie: str, max_comment_page: int = 30, delay: float = 1.2):
        self.headers = {
            "user-agent": USER_AGENT,
            "referer": "https://www.zhihu.com/",
            "Cookie": cookie,
        }
        self.max_comment_page = max_comment_page
        self.delay = delay
        self.total = 0

    def _fetch_comment_page(self, answer_id: str, writer: csv.DictWriter) -> Optional[str]:
        try:
            resp = requests.get(
                f"https://www.zhihu.com/api/v4/answers/{answer_id}/root_comments",
                headers=self.headers,
                params={"order": "normal", "limit": 20},
                timeout=15,
            )
            if resp.status_code != 200:
                return None
            payload = resp.json()
        except (requests.RequestException, ValueError):
            return None

        data_list = payload.get("data") or []
        if not data_list:
            return None

        for item in data_list:
            try:
                member = (item.get("author") or {}).get("member") or {}
                writer.writerow(
                    {
                        "website": self.platform,
                        "text_raw": str(item.get("content", "")).replace("\n", " ").replace(",", "，"),
                        "id": item.get("id", ""),
                        "screen_name": member.get("name", ""),
                        "like_counts": item.get("vote_count", 0),
                        "created_at": time.strftime(
                            "%a %b %d %H:%M:%S +0800 %Y", time.localtime(item.get("created_time") or 0)
                        ),
                    }
                )
                self.total += 1
            except (csv.Error, KeyError, ValueError, OSError):
                continue

        paging = payload.get("paging") or {}
        if paging.get("is_end", True):
            return None
        return paging.get("next")

    def crawl_topic(self, topic: str, max_answers: int = 5, max_comments: int = 600) -> Path:
        ensure_dirs()
        out_file = ZHIHU_DIR / f"{_safe_name(topic)}.csv"
        new_file = not out_file.exists()
        before = self.total

        with open(out_file, "a", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, CSV_HEADER)
            if new_file:
                writer.writeheader()

            try:
                resp = requests.get(
                    "https://www.zhihu.com/api/v4/search_v3",
                    headers=self.headers,
                    params={"q": topic, "t": "general", "limit": 20},
                    timeout=15,
                )
                items = resp.json().get("data", []) if resp.status_code == 200 else []
            except (requests.RequestException, ValueError):
                items = []

            answer_ids = [i["object"]["id"] for i in items if (i.get("object") or {}).get("type") == "answer"]
            for answer_id in answer_ids[:max_answers]:
                if self.total - before >= max_comments:
                    break
                for _ in range(self.max_comment_page):
                    nxt = self._fetch_comment_page(str(answer_id), writer)
                    if not nxt or self.total - before >= max_comments:
                        break
                    time.sleep(self.delay)

        return out_file


# ======================================================================
# YouTube
# ======================================================================


def _clean_content(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _stable_id(video_id: str, content: str, author: str) -> str:
    raw = f"{video_id}|{author}|{content[:200]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class YouTubeCrawler:
    """YouTube 评论采集：requests + innertube（不需要浏览器 / API key）。"""

    platform = "YouTube"

    def __init__(self, proxy: str = "", timeout: int = 30, delay: float = 1.2, retries: int = 3):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self.proxies = {"http": proxy, "https": proxy} if proxy else {}
        self.timeout = timeout
        self.delay = delay
        self.retries = retries
        self.comments: list[dict[str, str]] = []
        self._seen: set[str] = set()

    # -- 基础请求 ------------------------------------------------------
    def _get_text(self, url: str, params: dict[str, Any]) -> str:
        for _ in range(self.retries):
            try:
                resp = self.session.get(url, params=params, proxies=self.proxies, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.text
            except requests.RequestException:
                pass
            time.sleep(self.delay)
        return ""

    def _post_youtubei(
        self, endpoint: str, key: str, context: dict, payload: dict, referer: str | None = None
    ) -> Optional[dict]:
        body = {"context": context, **payload}
        client = (context or {}).get("client", {})
        headers = {
            "Content-Type": "application/json",
            "Origin": "https://www.youtube.com",
            "Referer": referer or "https://www.youtube.com/",
            "X-YouTube-Client-Name": "1",
            "X-YouTube-Client-Version": client.get("clientVersion", ""),
        }
        for _ in range(self.retries):
            try:
                resp = self.session.post(
                    f"https://www.youtube.com/youtubei/v1/{endpoint}",
                    params={"key": key},
                    headers=headers,
                    json=body,
                    proxies=self.proxies,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    return resp.json()
            except (requests.RequestException, ValueError):
                pass
            time.sleep(self.delay)
        return None

    # -- JSON 解析工具 -------------------------------------------------
    @staticmethod
    def _balanced_json(text: str, start: int) -> Optional[str]:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        return None

    def _extract_json(self, text: str, marker: str) -> Optional[dict]:
        idx = text.find(marker)
        if idx < 0:
            return None
        start = text.find("{", idx)
        if start < 0:
            return None
        raw = self._balanced_json(text, start)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _extract_context_and_key(self, html: str) -> tuple[Optional[dict], Optional[str]]:
        match = re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', html)
        key = match.group(1) if match else None
        cfg = self._extract_json(html, "ytcfg.set({")
        context = cfg.get("INNERTUBE_CONTEXT") if isinstance(cfg, dict) else None
        if not key and isinstance(cfg, dict):
            key = cfg.get("INNERTUBE_API_KEY")
        return context, key

    @staticmethod
    def _find_values(data: Any, name: str) -> list[Any]:
        found: list[Any] = []
        if isinstance(data, dict):
            for k, v in data.items():
                if k == name:
                    found.append(v)
                found.extend(YouTubeCrawler._find_values(v, name))
        elif isinstance(data, list):
            for item in data:
                found.extend(YouTubeCrawler._find_values(item, name))
        return found

    @staticmethod
    def _runs_text(value: Any) -> str:
        if isinstance(value, str):
            return _clean_content(value)
        if not isinstance(value, dict):
            return ""
        if "simpleText" in value:
            return _clean_content(value["simpleText"])
        return _clean_content("".join(str(r.get("text", "")) for r in value.get("runs", []) if isinstance(r, dict)))

    @staticmethod
    def _parse_like_count(text: str) -> int:
        value = (text or "").lower().replace(",", "").strip()
        match = re.search(r"(\d+(?:\.\d+)?)\s*(k|m)?", value)
        if not match:
            return 0
        number = float(match.group(1))
        unit = match.group(2)
        if unit == "k":
            number *= 1000
        elif unit == "m":
            number *= 1000000
        return int(number)

    # -- 评论解析 ------------------------------------------------------
    def _comment_dict(self, video_id: str, watch_url: str, cid: str, content: str, author: str,
                      published: str, like_text: Any) -> dict[str, str]:
        published = (published or "").strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", published):
            published = date.today().isoformat()
        return {
            "platform": self.platform,
            "source_id": f"youtube:{cid}",
            "user_name": author,
            "content": content,
            "published_at": published,
            "like_count": str(self._parse_like_count(self._runs_text(like_text) or str(like_text or ""))),
            "url": f"{watch_url}&lc={cid}",
        }

    def _parse_renderers(self, data: Any, video_id: str, watch_url: str) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []

        # 旧版 commentRenderer / commentThreadRenderer
        for key in ("commentRenderer", "commentThreadRenderer"):
            for renderer in self._find_values(data, key):
                if not isinstance(renderer, dict):
                    continue
                if key == "commentThreadRenderer":
                    renderer = (renderer.get("comment") or {}).get("commentRenderer") or {}
                content = _clean_content(self._runs_text(renderer.get("contentText")))
                if not content:
                    continue
                author = self._runs_text(renderer.get("authorText"))
                cid = renderer.get("commentId") or _stable_id(video_id, content, author)
                result.append(
                    self._comment_dict(
                        video_id, watch_url, cid, content, author,
                        self._runs_text(renderer.get("publishedTimeText")),
                        renderer.get("voteCount"),
                    )
                )

        # 新版 commentEntityPayload
        for payload in self._find_values(data, "commentEntityPayload"):
            if not isinstance(payload, dict):
                continue
            props = payload.get("properties") or {}
            content = _clean_content((props.get("content") or {}).get("content", ""))
            if not content:
                continue
            author = (payload.get("author") or {}).get("displayName", "")
            cid = props.get("commentId") or _stable_id(video_id, content, author)
            result.append(
                self._comment_dict(
                    video_id, watch_url, cid, content, author,
                    props.get("publishedTime", ""),
                    (payload.get("toolbar") or {}).get("likeButtonA11y", ""),
                )
            )
        return result

    # -- 主流程 --------------------------------------------------------
    def search_videos(self, query: str, limit: int = 5) -> list[str]:
        html = self._get_text("https://www.youtube.com/results", {"search_query": query})
        if not html:
            return []
        ids = list(dict.fromkeys(re.findall(r'"videoId":"([^"]+)"', html)))
        return ids[:limit]

    def fetch_comments(self, video_id: str, page_limit: int = 3) -> list[dict[str, str]]:
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        html = self._get_text("https://www.youtube.com/watch", {"v": video_id})
        if not html:
            return []

        context, key = self._extract_context_and_key(html)
        initial = self._extract_json(html, "ytInitialData =") or self._extract_json(html, "var ytInitialData =")
        comments = self._parse_renderers(initial, video_id, watch_url) if initial else []

        if not context or not key:
            return comments

        seen = {c["source_id"] for c in comments}
        cursors = [c.get("token") for c in self._find_values(initial, "continuationCommand") if isinstance(c, dict)]
        for cursor in [c for c in dict.fromkeys(cursors) if c][:1]:
            for _ in range(page_limit):
                data = self._post_youtubei("next", key, context, {"continuation": cursor}, referer=watch_url)
                if not data:
                    break
                for comment in self._parse_renderers(data, video_id, watch_url):
                    if comment["source_id"] in seen:
                        continue
                    seen.add(comment["source_id"])
                    comments.append(comment)
                next_cursor = next((v for v in self._find_values(data, "continuation") if isinstance(v, str)), None)
                if not next_cursor:
                    break
                cursor = next_cursor
                time.sleep(self.delay)
        return comments

    def crawl(self, queries: list[str], target: int = 200, videos_per_query: int = 5,
              comment_pages: int = 3) -> Path:
        ensure_dirs()
        for query in queries:
            for video_id in self.search_videos(query, limit=videos_per_query):
                for comment in self.fetch_comments(video_id, page_limit=comment_pages):
                    if comment["source_id"] in self._seen:
                        continue
                    self._seen.add(comment["source_id"])
                    self.comments.append(comment)
                    if len(self.comments) >= target:
                        break
                if len(self.comments) >= target:
                    break
            if len(self.comments) >= target:
                break

        header = ["platform", "source_id", "user_name", "content", "published_at", "like_count", "url"]
        with open(YOUTUBE_RAW, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for c in self.comments:
                writer.writerow([c[h] for h in header])
        return YOUTUBE_RAW


# ======================================================================
# LangChain 工具封装
# ======================================================================


@tool
def crawl_weibo(topics: str = "", max_pages: int = 3) -> str:
    """采集微博评论。

    Args:
        topics: 话题列表，逗号分隔，例如 "#美以伊冲突#,#中东局势#"；留空则使用默认 6 个话题。
        max_pages: 每个话题最多翻多少页搜索结果（1-10）。

    Returns:
        采集结果摘要（每个话题的落盘文件与新增条数）。
    """
    cookie = os.getenv("WEIBO_COOKIE", "").strip()
    xsrf = os.getenv("WEIBO_XSRF_TOKEN", "").strip()
    if not cookie:
        return "未配置 WEIBO_COOKIE，无法采集微博。请在 .env 中填入微博 Cookie 后重试（或跳过采集直接分析已有 CSV）。"

    topic_list = _split_csv_arg(topics, DEFAULT_WEIBO_TOPICS)
    max_pages = max(1, min(int(max_pages), 10))
    crawler = WeiboCrawler(cookie=cookie, xsrf=xsrf)

    lines = [f"微博采集：{len(topic_list)} 个话题，每话题最多 {max_pages} 页搜索结果"]
    for topic in topic_list:
        before = crawler.total
        try:
            path = crawler.crawl_topic(topic, max_pages=max_pages)
            lines.append(f"  - {topic}: +{crawler.total - before} 条 -> {path.name}")
        except Exception as exc:  # 单个话题失败不影响整批
            lines.append(f"  - {topic}: 失败（{type(exc).__name__}: {exc}）")
    lines.append(f"微博本次新增评论总数：{crawler.total} 条")
    return "\n".join(lines)


@tool
def crawl_zhihu(topics: str = "", max_answers: int = 5, max_comments: int = 600) -> str:
    """采集知乎评论。

    Args:
        topics: 搜索关键词，逗号分隔；留空使用默认 6 个话题。
        max_answers: 每个话题最多取多少条回答。
        max_comments: 每个话题最多采集多少条评论。

    Returns:
        采集结果摘要。
    """
    cookie = os.getenv("ZHIHU_COOKIE", "").strip()
    if not cookie:
        return "未配置 ZHIHU_COOKIE，无法采集知乎。请在 .env 中填入知乎 Cookie 后重试。"

    topic_list = _split_csv_arg(topics, DEFAULT_ZHIHU_TOPICS)
    crawler = ZhihuCrawler(cookie=cookie)

    lines = [f"知乎采集：{len(topic_list)} 个话题，回答数/话题≤{max_answers}，评论数/话题≤{max_comments}"]
    for topic in topic_list:
        before = crawler.total
        try:
            path = crawler.crawl_topic(topic, max_answers=max_answers, max_comments=max_comments)
            lines.append(f"  - {topic}: +{crawler.total - before} 条 -> {path.name}")
        except Exception as exc:
            lines.append(f"  - {topic}: 失败（{type(exc).__name__}: {exc}）")
    lines.append(f"知乎本次新增评论总数：{crawler.total} 条")
    return "\n".join(lines)


@tool
def crawl_youtube(query: str = "", target: int = 200, videos_per_query: int = 5, comment_pages: int = 3) -> str:
    """采集 YouTube 评论（需要本地代理时在 .env 配置 YOUTUBE_PROXY）。

    Args:
        query: 英文搜索词，逗号分隔；留空使用默认 4 个搜索词。
        target: 目标评论条数上限。
        videos_per_query: 每个搜索词取多少个视频。
        comment_pages: 每个视频翻多少页评论。

    Returns:
        采集结果摘要。
    """
    query_list = _split_csv_arg(query, DEFAULT_YOUTUBE_QUERIES)
    target = max(10, min(int(target), 5000))
    crawler = YouTubeCrawler(proxy=os.getenv("YOUTUBE_PROXY", "").strip())
    try:
        path = crawler.crawl(query_list, target=target, videos_per_query=videos_per_query,
                             comment_pages=comment_pages)
    except Exception as exc:
        return f"YouTube 采集失败（{type(exc).__name__}: {exc}）。请检查网络/代理后重试。"
    return (
        f"YouTube 采集完成：{len(query_list)} 个搜索词，实际入库 {len(crawler.comments)} 条"
        f"（目标 {target} 条）-> {path}"
    )


CRAWL_TOOLS = [crawl_weibo, crawl_zhihu, crawl_youtube]
