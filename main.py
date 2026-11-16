"""main.py —— US-Iran 舆情分析 Agent 入口

用 LangChain 1.x 的 `create_agent` 把 tools/ 下的 9 个工具注册给大模型，
由模型自己决定「先采集还是直接分析」「先清洗还是先合并」，本文件只做：
    1. 读取 .env 里的模型配置，构建 ChatOpenAI（兼容 DeepSeek / 通义 / OpenAI）
    2. 注册全部工具 + 系统提示词（含目录红线与标准作业顺序）
    3. 命令行交互：多轮对话、工具调用过程可视化
    4. 无 LLM 时的确定性兜底：--pipeline 直接按固定顺序跑完整链路

用法：
    python main.py                       # 进入交互模式（推荐）
    python main.py --task "分析现有数据并出看板"
    python main.py --pipeline            # 不调 LLM，直接跑「合并->清洗->合并->分析->看板」
    python main.py --pipeline --crawl    # 额外先跑三个平台的采集
    python main.py --list-tools          # 只打印已注册的工具
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Windows 控制台偶发 GBK 编码报错时，宁可丢字符也不要让程序崩
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from tools import OUTPUT_DIR, ensure_dirs, register_all_tools  # noqa: E402
from tools.analysis_tool import analyze_data  # noqa: E402
from tools.chart_tool import generate_dashboard  # noqa: E402
from tools.clean_tool import clean_data  # noqa: E402
from tools.merge_tool import merge_all_platforms, merge_platform  # noqa: E402

SYSTEM_PROMPT = """你是「美伊冲突多平台舆情分析 Agent」，服务于一名数据分析/产品岗求职者的作品集项目。
你的能力来自 9 个工具，覆盖一条完整的数据流水线：

  【采集】crawl_weibo / crawl_zhihu / crawl_youtube
  【合并】merge_platform（分话题合并）/ merge_all_platforms（三平台统一 schema）
  【清洗】clean_data（去重 + 语言过滤 + 关键词筛选）
  【分析】analyze_data（话题/情感/关键词/时间 四维分析）
  【交付】generate_dashboard（可视化看板 HTML）/ save_report（Markdown 结论报告）

标准作业顺序（数据已存在时可跳过前面的步骤）：
  crawl_* -> merge_platform -> clean_data -> merge_all_platforms -> analyze_data -> generate_dashboard -> save_report

工作规则：
1. 目录红线：只能读写本项目目录（D:/pycharm_file/US-Iran_Agent01）。Comments/ 下是原始数据，只读取不改写；
   所有产物只能落在 outputs/。绝不触碰任何其它项目的文件夹。
2. 先规划再动手：接到任务先用一句话说明你打算调用哪些工具、为什么，然后逐个调用。
3. 忠于数据：所有数字必须来自工具返回结果，不允许估算或编造；工具报错时如实说明并给出补救建议。
4. 分析要落到结论：拿到四维分析结果后，至少给出
   「核心发现（3-5 条，带数字）」「中外舆论差异」「风险/情绪拐点」「可执行建议」四部分。
5. 需要出报告时，把结论写成 Markdown 交给 save_report 落盘（不要只在对话里输出）。
6. 回答用简体中文，简洁、有信息密度，避免空话套话。"""

BANNER = r"""
============================================================
  US-Iran 多平台舆情分析 Agent
  微博 / 知乎 / YouTube -> 清洗 -> 四维分析 -> 可视化看板
============================================================
  /tools   查看已注册工具        /pipeline  跑确定性全流程
  /reset   清空对话上下文        /quit      退出
============================================================
"""


# ---------------------------------------------------------------- 模型


def build_llm():
    """按 .env 构建 ChatOpenAI（OpenAI 兼容协议，DeepSeek/通义/月之暗面都能用）。"""
    from langchain_openai import ChatOpenAI

    api_key = (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None, "未在 .env 中找到 LLM_API_KEY / OPENAI_API_KEY"

    base_url = (os.getenv("LLM_BASE_URL") or "").strip() or None
    model = (os.getenv("LLM_MODEL") or "gpt-4o-mini").strip()
    try:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    except ValueError:
        temperature = 0.3

    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=temperature, timeout=120)
    return llm, f"模型：{model}｜base_url：{base_url or 'OpenAI 官方'}"


def build_agent(llm, tools):
    """优先用 LangChain 1.x 的 create_agent；老版本回退到 langgraph 的 react agent。"""
    try:
        from langchain.agents import create_agent

        return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    except ImportError:
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------- 执行


def _short(text: str, limit: int = 260) -> str:
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + " ..."


def run_agent_turn(agent, messages: list, verbose: bool = True) -> tuple[str, list]:
    """跑一轮 Agent 对话，边流式打印工具调用，返回 (最终回答, 累计消息列表)。"""
    from langchain_core.messages import AIMessage, ToolMessage

    cursor = len(messages)
    state_messages = messages

    try:
        for state in agent.stream({"messages": messages}, stream_mode="values"):
            if not isinstance(state, dict):
                continue
            msgs = state.get("messages") or []
            for msg in msgs[cursor:]:
                if isinstance(msg, AIMessage):
                    if getattr(msg, "tool_calls", None):
                        for call in msg.tool_calls:
                            args = ", ".join(f"{k}={v!r}" for k, v in (call.get("args") or {}).items())
                            print(f"  [调用] {call.get('name')}({args})")
                    elif msg.content:
                        print(f"  [思考] {_short(msg.content, 200)}")
                elif isinstance(msg, ToolMessage) and verbose:
                    print(f"  [返回] {_short(msg.content)}")
            cursor = len(msgs)
            state_messages = msgs
    except Exception as exc:
        print(f"  [警告] 流式执行中断（{type(exc).__name__}: {exc}），改为一次性调用 ...")
        result = agent.invoke({"messages": messages})
        return result["messages"][-1].content, result["messages"]

    final_text = ""
    for msg in reversed(state_messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            final_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    return final_text or "(本轮没有文本输出)", state_messages


def interactive(agent, tools) -> None:
    from langchain_core.messages import HumanMessage

    print(BANNER)
    messages: list = []
    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return
        if not user_input:
            continue
        if user_input in ("/quit", "/exit", "quit", "exit", "0"):
            print("已退出。")
            return
        if user_input == "/reset":
            messages = []
            print("上下文已清空。")
            continue
        if user_input == "/tools":
            for i, tool in enumerate(tools, 1):
                desc = (tool.description or "").splitlines()[0]
                print(f"  {i}. {tool.name:<22} {desc}")
            continue
        if user_input == "/pipeline":
            run_pipeline(with_crawl=False)
            continue

        messages.append(HumanMessage(content=user_input))
        print()
        answer, messages = run_agent_turn(agent, messages)
        print(f"\nAgent > {answer}")


# ---------------------------------------------------------------- 确定性兜底


def run_pipeline(with_crawl: bool = False) -> str:
    """不依赖 LLM，按固定顺序跑完整链路；返回分析摘要。"""
    ensure_dirs()
    print("\n[管道] 开始执行确定性流程（不调用大模型）")

    if with_crawl:
        from tools.crawl_tool import crawl_weibo, crawl_youtube, crawl_zhihu

        for name, tool in (("微博", crawl_weibo), ("知乎", crawl_zhihu), ("YouTube", crawl_youtube)):
            print(f"[管道] 采集 {name} ...")
            try:
                print(tool.invoke({"target": 200} if tool.name == "crawl_youtube" else {}))
            except Exception as exc:
                print(f"[管道] {name} 采集跳过（{type(exc).__name__}: {exc}）")

    print("[管道] 1/4 平台内合并 ...")
    print(merge_platform.invoke({"platform": "all"}))
    print("[管道] 2/4 清洗筛选 ...")
    print(clean_data.invoke({"platform": "all"}))
    print("[管道] 3/4 三平台合并 ...")
    print(merge_all_platforms.invoke({}))
    print("[管道] 4/4 四维分析 ...")
    summary = analyze_data.invoke({"detail": "full"})
    print(summary)
    print("[管道] 生成看板 ...")
    print(generate_dashboard.invoke({}))

    # 落一份确定性报告（不依赖 LLM，保证 pipeline 模式也有交付物）
    report = f"""# 美伊冲突多平台舆情分析报告

> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 数据来源：Comments/merged/三个平台数据合并结果.csv
> 生成方式：确定性流水线（爬取 -> 合并 -> 清洗 -> 四维分析 -> 看板）

## 一、四维分析结果

```
{summary}
```

## 二、可视化看板

见 outputs/评论分析看板.html（话题分布、情感分布、双语词云、时间趋势、分段聚合、话题演化）。

## 三、说明

本文由脚本自动生成，数字均来自工具计算结果。如需带洞察的结论（核心发现 / 中外差异 / 情绪拐点 / 建议），
请运行 `python main.py` 进入 Agent 模式，让大模型基于同一份分析结果撰写。
"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"舆情分析报告_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    path.write_text(report, encoding="utf-8")
    print(f"[管道] 报告已保存：{path}")
    return summary


# ---------------------------------------------------------------- 入口


def main() -> None:
    parser = argparse.ArgumentParser(description="US-Iran 多平台舆情分析 Agent")
    parser.add_argument("--task", type=str, default="", help="一次性任务（不进入交互）")
    parser.add_argument("--pipeline", action="store_true", help="不调用 LLM，跑确定性流水线")
    parser.add_argument("--crawl", action="store_true", help="配合 --pipeline，先执行采集")
    parser.add_argument("--list-tools", action="store_true", help="打印已注册工具")
    parser.add_argument("--quiet", action="store_true", help="不打印工具调用细节")
    args = parser.parse_args()

    tools = register_all_tools()

    if args.list_tools:
        print(f"已注册 {len(tools)} 个工具：")
        for t in tools:
            print(f"  - {t.name}: {(t.description or '').splitlines()[0]}")
        return

    if args.pipeline:
        run_pipeline(with_crawl=args.crawl)
        return

    llm, info = build_llm()
    if llm is None:
        print(f"[提示] {info}，无法启动大模型 Agent。")
        print("       可选：1) 在 .env 填入 LLM_API_KEY 后重跑；2) 先跑 `python main.py --pipeline` 走固定流程。")
        return

    print(f"[配置] {info}")
    agent = build_agent(llm, tools)

    if args.task:
        from langchain_core.messages import HumanMessage

        print(f"\n任务：{args.task}\n")
        answer, _ = run_agent_turn(agent, [HumanMessage(content=args.task)], verbose=not args.quiet)
        print(f"\nAgent > {answer}")
        return

    interactive(agent, tools)


if __name__ == "__main__":
    main()
