"""chart_tool.py —— 绘图工具

读取 analysis_tool 产出的四维分析结果（没有就现算），用 pyecharts 生成 ECharts option，
再拼装成一个单文件看板：outputs/评论分析看板.html

看板内容（对应四维分析）：
    KPI 概览 -> 话题饼图 / 情感饼图 -> 双语词云对比 -> 时间趋势折线
    -> 分段情感与话题堆叠柱 -> 话题演化动态词云（可播放）
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie
from pyecharts.globals import ThemeType

from . import DASHBOARD_HTML, OUTPUT_DIR
from .analysis_tool import CN_PLATFORMS, extract_keywords, get_last_result, load_data

PALETTE = ["#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de", "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc"]
PERIODS = ["爆发期", "高峰期", "降温期"]


# ---------------------------------------------------------------- 单图构建


def pie_option(title: str, data: list[tuple[str, int]]) -> dict:
    pie = (
        Pie(init_opts=opts.InitOpts(width="600px", height="350px", theme=ThemeType.LIGHT))
        .add("", data, radius=["35%", "70%"], center=["40%", "50%"])
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title, title_textstyle_opts=opts.TextStyleOpts(font_size=16)),
            legend_opts=opts.LegendOpts(pos_right="5%", pos_top="center", orient="vertical",
                                        item_width=20, item_height=12, item_gap=8),
        )
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%", margin=15))
    )
    return json.loads(pie.dump_options())


def wordcloud_option(title: str, data: list[list], font: str, animation: bool = False) -> dict:
    return {
        # 词云布局较重，关掉动画可保证首帧渲染完成（避免多图并存时抢帧导致空白）
        "animation": animation,
        "title": {"text": title, "left": "center", "textStyle": {"fontSize": 16}},
        "tooltip": {"show": True},
        "series": [
            {
                "type": "wordCloud",
                "shape": "circle",
                "gridSize": 6,
                "sizeRange": [16, 78],
                "rotationRange": [-90, 90],
                "rotationStep": 45,
                "drawOutOfBound": False,
                "textStyle": {"fontFamily": font, "color": "#2c3e50"},
                "emphasis": {"focus": "self", "textStyle": {"shadowBlur": 10}},
                # 逐词上色（echarts-wordcloud 2.x 里 textStyle.color 不接受颜色数组）
                "data": [
                    {"name": w, "value": c, "textStyle": {"color": PALETTE[i % len(PALETTE)]}}
                    for i, (w, c) in enumerate(data)
                ] or [{"name": "no data", "value": 1}],
            }
        ],
    }


def line_option(daily: list[dict]) -> dict:
    dates = [d["date"] for d in daily]
    line = (
        Line(init_opts=opts.InitOpts(width="1200px", height="450px", theme=ThemeType.LIGHT))
        .add_xaxis(dates)
        .add_yaxis("中国平台", [d["cn"] for d in daily], is_smooth=True,
                   linestyle_opts=opts.LineStyleOpts(width=3))
        .add_yaxis("海外平台", [d["en"] for d in daily], is_smooth=True,
                   linestyle_opts=opts.LineStyleOpts(width=3))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="国内外舆论热度时间趋势", title_textstyle_opts=opts.TextStyleOpts(font_size=16)),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45, font_size=10)),
            yaxis_opts=opts.AxisOpts(name="评论数量"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            legend_opts=opts.LegendOpts(pos_top="5%"),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
        )
    )
    return json.loads(line.dump_options())


def period_sentiment_option(periods: dict[str, Any]) -> dict:
    bar = (
        Bar(init_opts=opts.InitOpts(width="600px", height="400px", theme=ThemeType.LIGHT))
        .add_xaxis(PERIODS)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="各时段情感分布", title_textstyle_opts=opts.TextStyleOpts(font_size=16)),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow"),
            legend_opts=opts.LegendOpts(pos_top="5%"),
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )
    colors = {"正面": "#91cc75", "中立": "#fac858", "负面": "#ee6666"}
    for senti in ("正面", "中立", "负面"):
        values = [int(periods.get(p, {}).get("sentiment", {}).get(senti, 0)) for p in PERIODS]
        bar.add_yaxis(senti, values, stack="total", itemstyle_opts=opts.ItemStyleOpts(color=colors[senti]))
    return json.loads(bar.dump_options())


def period_topic_option(periods: dict[str, Any]) -> dict:
    topics = set()
    for info in periods.values():
        topics.update((info.get("topic") or {}).keys())
    bar = (
        Bar(init_opts=opts.InitOpts(width="600px", height="400px", theme=ThemeType.LIGHT))
        .add_xaxis(PERIODS)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="各时段话题热度", title_textstyle_opts=opts.TextStyleOpts(font_size=16)),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow"),
            legend_opts=opts.LegendOpts(pos_top="5%", orient="vertical", pos_right="2%"),
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )
    for i, topic in enumerate(sorted(topics)):
        values = [int(periods.get(p, {}).get("topic", {}).get(topic, 0)) for p in PERIODS]
        bar.add_yaxis(topic, values, itemstyle_opts=opts.ItemStyleOpts(color=PALETTE[i % len(PALETTE)]))
    return json.loads(bar.dump_options())


def dynamic_topic_data(df: pd.DataFrame, max_days: int = 30, top_words: int = 25) -> list[dict]:
    """按天切词，得到「话题演化」动态词云的数据序列。"""
    df = df.copy()
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    timeline = []
    for day, chunk in df.groupby("date_str"):
        counter = Counter()
        for platform, text in zip(chunk["platform"], chunk["text_raw"]):
            counter.update(extract_keywords(text, str(platform) in CN_PLATFORMS))
        words = counter.most_common(top_words) or [("no data", 1)]
        timeline.append({"date": day, "words": [[w, int(c)] for w, c in words]})
    return timeline[-max_days:]


# ---------------------------------------------------------------- 看板拼装

HTML_HEAD = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>美伊冲突多平台舆情分析看板</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/echarts-wordcloud@2.1.0/dist/echarts-wordcloud.min.js"></script>
<style>
    * {margin:0;padding:0;box-sizing:border-box;}
    body {background:#f5f7fa;font-family:'Microsoft YaHei','PingFang SC',sans-serif;min-height:100vh;color:#2c3e50;}
    .header {background:linear-gradient(135deg,#2c3e50 0%,#34495e 100%);color:#fff;padding:26px;text-align:center;}
    .header h1 {font-size:26px;font-weight:500;margin-bottom:8px;}
    .header p {font-size:13px;opacity:.85;}
    .container {max-width:1320px;margin:24px auto;padding:0 20px;}
    .kpis {display:flex;gap:16px;flex-wrap:wrap;margin-bottom:26px;}
    .kpi {flex:1;min-width:180px;background:#fff;border-radius:12px;padding:18px 20px;box-shadow:0 2px 15px rgba(0,0,0,.06);}
    .kpi .label {font-size:13px;color:#7f8c8d;margin-bottom:8px;}
    .kpi .value {font-size:26px;font-weight:600;color:#2c3e50;}
    .kpi .sub {font-size:12px;color:#95a5a6;margin-top:6px;}
    .section {margin-bottom:28px;}
    .section-title {font-size:17px;font-weight:600;margin-bottom:16px;padding-left:14px;border-left:4px solid #3498db;}
    .chart-row {display:flex;gap:18px;justify-content:center;flex-wrap:wrap;}
    .chart-box {background:#fff;padding:16px;border-radius:12px;box-shadow:0 2px 15px rgba(0,0,0,.06);}
    .full-width {width:100%;min-width:820px;}
    .half-width {width:calc(50% - 9px);min-width:560px;}
    .ctrl {display:flex;gap:10px;align-items:center;margin-bottom:8px;}
    .ctrl button {padding:7px 15px;border:none;border-radius:6px;background:#3498db;color:#fff;cursor:pointer;font-size:13px;}
    .ctrl button:hover {background:#2980b9;}
    .ctrl span {font-weight:600;color:#2c3e50;}
    .footer {text-align:center;color:#95a5a6;font-size:12px;padding:18px 0 30px;}
    @media (max-width:1200px) {.half-width{width:100%;min-width:300px;} .full-width{min-width:300px;}}
</style>
</head>
<body>
<div class="header">
    <h1>美伊冲突多平台舆情分析看板</h1>
    <p>微博 · 知乎 · YouTube 三平台评论，四维分析（话题 / 情感 / 关键词 / 时间）</p>
</div>
<div class="container">
"""


def _kpi_block(overview: dict, sentiment: dict, peak: dict | None) -> str:
    platforms = " · ".join(f"{k} {v:,}" for k, v in overview["platforms"].items())
    negative = sentiment.get("负面", {}).get("ratio", 0)
    peak_text = f"{peak['date']}" if peak else "-"
    peak_sub = f"{peak['total']:,} 条（中 {peak['cn']} / 外 {peak['en']}）" if peak else ""
    cards = [
        ("评论总量", f"{overview['total']:,}", f"{overview['date_range'][0]} ~ {overview['date_range'][1]}，共 {overview['days']} 天"),
        ("平台分布", f"{len(overview['platforms'])} 个", platforms),
        ("讨论峰值日", peak_text, peak_sub),
        ("负面情绪占比", f"{negative}%", f"日均 {overview['avg_daily']} 条"),
    ]
    html = ['<div class="kpis">']
    for label, value, sub in cards:
        html.append(
            f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
        )
    html.append("</div>")
    return "\n".join(html)


HTML_SCRIPT = """
<script>
var CTRL = null;
function mount(id, opt){ var c = echarts.init(document.getElementById(id)); c.setOption(opt); return c; }
mount('pie_topic', PIE_TOPIC);
mount('pie_emo', PIE_EMO);
mount('wc_cn', WC_CN);
mount('wc_en', WC_EN);
mount('time_trend', TIME_TREND);
mount('sentiment_bar', SENTIMENT_BAR);
mount('topic_bar', TOPIC_BAR);

var dynamic_chart = echarts.init(document.getElementById('dynamic_topic'));
var DYNAMIC_DATA = __DYNAMIC_DATA__;
var idx = 0, timer = null;

function renderTopic(i){
    if (i < 0 || i >= DYNAMIC_DATA.length) return;
    idx = i;
    var item = DYNAMIC_DATA[i];
    document.getElementById('current_date').innerHTML = item.date;
    dynamic_chart.setOption({
        title: {text: '话题演化 - ' + item.date, left: 'center', textStyle: {fontSize: 15}},
        tooltip: {show: true},
        series: [{
            type: 'wordCloud', shape: 'circle', left: 'center', top: 'center',
            width: '92%', height: '88%', gridSize: 9, sizeRange: [12, 52],
            rotationRange: [-90, 90], rotationStep: 45, drawOutOfBound: false,
            textStyle: {fontFamily: "'Microsoft YaHei', Arial, sans-serif",
                color: ['#5470c6','#91cc75','#fac858','#ee6666','#73c0de','#3ba272','#fc8452','#9a60b4','#ea7ccc']},
            emphasis: {focus: 'self', textStyle: {shadowBlur: 10, shadowColor: 'rgba(0,0,0,.4)'}},
            data: item.words.map(function(w){ return {name: w[0], value: w[1]}; })
        }]
    });
}
function prevDate(){ renderTopic(idx - 1); }
function nextDate(){ renderTopic(idx + 1); }
function togglePlay(){
    if (timer) { clearInterval(timer); timer = null; }
    else { timer = setInterval(function(){ renderTopic((idx + 1) % DYNAMIC_DATA.length); }, 2000); }
}
if (DYNAMIC_DATA.length > 0) {
    renderTopic(0);
    setTimeout(function(){
        if (!timer) timer = setInterval(function(){ renderTopic((idx + 1) % DYNAMIC_DATA.length); }, 2500);
    }, 2500);
}
document.addEventListener('visibilitychange', function(){
    if (document.hidden && timer) { clearInterval(timer); timer = null; }
});
window.addEventListener('resize', function(){
    document.querySelectorAll('.chart-box div').forEach(function(el){
        var inst = echarts.getInstanceByDom(el); if (inst) inst.resize();
    });
});
</script>
</body>
</html>
"""


def build_dashboard(result: dict[str, Any] | None = None) -> Path:
    """根据分析结果生成看板 HTML，返回文件路径。"""
    result = result or get_last_result()
    if not result:
        raise RuntimeError("没有可用的分析结果，请先运行 analyze_data")

    overview = result["overview"]
    sentiment = result["sentiment"]
    topic = result["topic"]["overall"]
    periods = result["time"]["periods"]

    pie_topic = pie_option("评论话题分类", [(k, v["count"]) for k, v in topic.items()])
    pie_emo = pie_option(
        "评论情感分布",
        [(k, v["count"]) for k, v in sentiment["overall"].items()],
    )
    wc_cn = wordcloud_option("中国网友高频词汇", result["keywords"]["cn_top50"], "Microsoft YaHei")
    wc_en = wordcloud_option("海外网友高频词汇", result["keywords"]["en_top50"], "Arial")

    try:
        df = load_data()
        dynamic = dynamic_topic_data(df)
    except Exception:
        dynamic = []

    body = f"""
{_kpi_block(overview, sentiment["overall"], result["time"].get("peak_day"))}
<div class="section">
  <div class="section-title">基础数据概览 · 话题与情感</div>
  <div class="chart-row">
    <div class="chart-box half-width"><div id="pie_topic" style="width:100%;height:350px;"></div></div>
    <div class="chart-box half-width"><div id="pie_emo" style="width:100%;height:350px;"></div></div>
  </div>
</div>
<div class="section">
  <div class="section-title">双语并列词云 · 国内外关注点对比</div>
  <div class="chart-row">
    <div class="chart-box half-width"><div id="wc_cn" style="width:100%;height:480px;"></div></div>
    <div class="chart-box half-width"><div id="wc_en" style="width:100%;height:480px;"></div></div>
  </div>
</div>
<div class="section">
  <div class="section-title">时间趋势 · 国内外舆论热度对比</div>
  <div class="chart-row">
    <div class="chart-box full-width"><div id="time_trend" style="width:100%;height:450px;"></div></div>
  </div>
</div>
<div class="section">
  <div class="section-title">时段聚合 · 情感与话题动态变化</div>
  <div class="chart-row">
    <div class="chart-box half-width"><div id="sentiment_bar" style="width:100%;height:400px;"></div></div>
    <div class="chart-box half-width"><div id="topic_bar" style="width:100%;height:400px;"></div></div>
  </div>
</div>
<div class="section">
  <div class="section-title">演化型话题模型 · 话题随时间演变</div>
  <div class="chart-row">
    <div class="chart-box full-width">
      <div class="ctrl">
        <button onclick="prevDate()">上一天</button>
        <span id="current_date">-</span>
        <button onclick="nextDate()">下一天</button>
        <button onclick="togglePlay()" style="margin-left:auto;">播放 / 暂停</button>
      </div>
      <div id="dynamic_topic" style="width:100%;height:450px;"></div>
    </div>
  </div>
</div>
<div class="footer">数据来源：{result.get('source_file', '')} · 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · 由 US-Iran Agent 自动生成</div>
</div>
"""

    options = (
        "var PIE_TOPIC = " + json.dumps(pie_topic, ensure_ascii=False) + ";\n"
        "var PIE_EMO = " + json.dumps(pie_emo, ensure_ascii=False) + ";\n"
        "var WC_CN = " + json.dumps(wc_cn, ensure_ascii=False) + ";\n"
        "var WC_EN = " + json.dumps(wc_en, ensure_ascii=False) + ";\n"
        "var TIME_TREND = " + json.dumps(line_option(result["time"]["daily"]), ensure_ascii=False) + ";\n"
        "var SENTIMENT_BAR = " + json.dumps(period_sentiment_option(periods), ensure_ascii=False) + ";\n"
        "var TOPIC_BAR = " + json.dumps(period_topic_option(periods), ensure_ascii=False) + ";\n"
    )
    script = HTML_SCRIPT.replace("__DYNAMIC_DATA__", json.dumps(dynamic, ensure_ascii=False))

    html = HTML_HEAD + body + "<script>\n" + options + "</script>\n" + script
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_HTML.write_text(html, encoding="utf-8")
    return DASHBOARD_HTML


@tool
def generate_dashboard() -> str:
    """生成可视化看板 HTML（outputs/评论分析看板.html）。

    会自动复用最近一次 analyze_data 的结果；若还没分析过，则现场计算一次。

    Returns:
        看板路径 + 图表清单。
    """
    try:
        path = build_dashboard()
    except Exception as exc:
        return f"看板生成失败（{type(exc).__name__}: {exc}）。请先执行 analyze_data。"

    return (
        f"看板已生成：{path}\n"
        "包含 7 个图表：话题分布饼图、情感分布饼图、中文词云、海外词云、"
        "国内外时间趋势、分段情感/话题堆叠柱、话题演化动态词云（可播放）。"
    )


CHART_TOOLS = [generate_dashboard]
