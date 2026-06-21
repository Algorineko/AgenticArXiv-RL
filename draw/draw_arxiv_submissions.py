#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 arXiv 月度/年度提交量统计图
- 左侧 Y 轴：年度总量柱状图
- 右侧 Y 轴：月度数据折线图
- 标题显示截止月份和累积总数，全部使用中文标注
- 自动适配系统中文字体
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from datetime import datetime
import numpy as np

# ----------------------------- 中文字体自动查找 -----------------------------
def find_chinese_font():
    """返回第一个可用的中文字体名称，若没有则返回 None"""
    candidates = [
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'Noto Sans CJK SC',
        'Noto Sans CJK TC',
        'SimHei',
        'Microsoft YaHei',
        'Arial Unicode MS',
        'PingFang SC',
        'Heiti SC',
        'STHeiti',
    ]
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available_fonts:
            return font_name
    return None

chinese_font = find_chinese_font()
if chinese_font is None:
    print("警告：未找到任何中文字体，图表中的中文将无法正常显示。")
    print("请在终端执行以下命令安装中文字体（例如文泉驿微米黑）：")
    print("  sudo apt update && sudo apt install fonts-wqy-microhei")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
else:
    print(f"使用中文字体: {chinese_font}")
    plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False

# ----------------------------- 配置 -----------------------------
DATA_PATH = "./data/get_monthly_submissions.csv"
OUTPUT_DIR = "./draw/images"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "arxiv_submissions.png")

# ----------------------------- 数据处理 --------------s---------------
df = pd.read_csv(DATA_PATH)
df['month'] = pd.to_datetime(df['month'])
df = df.sort_values('month').reset_index(drop=True)

total_submissions = df['submissions'].sum()
last_month = df['month'].iloc[-1]
last_month_str = last_month.strftime('%Y-%m')

df['year'] = df['month'].dt.year
yearly = df.groupby('year')['submissions'].sum().reset_index()
yearly['date_mid'] = pd.to_datetime(yearly['year'].astype(str) + '-07-01')

# ----------------------------- 绘图 -----------------------------
fig, ax = plt.subplots(figsize=(16, 7),dpi=300)

# 创建右侧 Y 轴用于月度数据
ax2 = ax.twinx()

# 1) 年度柱状图（左侧 Y 轴）
bar_width = 360
bars = ax.bar(yearly['date_mid'], yearly['submissions'],
              width=bar_width, color='steelblue', alpha=0.5,
              label='年度总提交量', zorder=2)

# 2) 月度折线图（右侧 Y 轴）
line, = ax2.plot(df['month'], df['submissions'],
                 color='darkorange', linewidth=1.2, alpha=0.9,
                 label='月度提交量', zorder=3)

# 设置 X 轴格式
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlabel('年份', fontsize=14)
ax.set_xlim(df['month'].min(), df['month'].max())

# 左侧 Y 轴标签
ax.set_ylabel('年度提交量', fontsize=14)
ax.tick_params(axis='y')

# 右侧 Y 轴标签
ax2.set_ylabel('月度提交量', fontsize=14)
ax2.tick_params(axis='y')

# 网格（仅使用左侧轴网格，避免重叠）
ax.grid(True, linestyle='--', alpha=0.3)
ax2.grid(False)

# 标题
title = (f'ArXiv 月度及年度提交量\n'
         f'截止日期：{last_month_str}  |  历史累积总数：{total_submissions:,}')
ax.set_title(title, fontsize=18, fontweight='bold')

# 合并图例
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

fig.tight_layout()

# 保存
os.makedirs(OUTPUT_DIR, exist_ok=True)
fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"图片已保存至：{OUTPUT_FILE}")