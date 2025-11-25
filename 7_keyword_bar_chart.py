#!/usr/bin/env python3
# ============================================
# 📊 EU Feedback Keyword Frequency Visualization Script
# Created by Plop (@plopnl)
# Last updated: 2025-11-21
# Description: Visualizes keyword frequencies in consultation feedback using bar charts.
#              Uses same layout and footer style as other EU plots for consistency.
# Usage: python 7_keyword_barchart.py
# ============================================

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 🔧 SETTINGS
# =========================
DATA_DIR = "data"
FIGURES_DIR = "figures"
INPUT_FILE = os.path.join(DATA_DIR, "keyword_counts.csv")

TOP_X = 20   # number of top keywords to show

os.makedirs(FIGURES_DIR, exist_ok=True)

# =========================
# 🔧 Helper: Plot with footer
# =========================
def plot_with_footer(figsize=(10, 12)):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[20, 1])
    ax = fig.add_subplot(gs[0])
    ax_footer = fig.add_subplot(gs[1])
    ax_footer.axis("off")
    return fig, ax, ax_footer

# =========================
# 📥 Load keyword counts
# =========================
df = pd.read_csv(INPUT_FILE)
df = df.sort_values("count", ascending=False).reset_index(drop=True)

# =========================
# 🎨 Function to plot chart
# =========================
def plot_keyword_bars(data, title, filename):
    sns.set(style="whitegrid")
    fig, ax, ax_footer = plot_with_footer(figsize=(10, 12))

    sns.barplot(
        data=data,
        y="keyword",
        x="count",
        color="skyblue",
        ax=ax
    )

    ax.set_xlim(0, data["count"].max() * 1.15)
    ax.set_xlabel("Keyword Occurrence Count")
    ax.set_ylabel("Keyword")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.tick_params(axis="y", labelsize=8)

    for i, row in data.iterrows():
        ax.text(
            row["count"] + data["count"].max() * 0.01,
            i,
            f'{row["count"]}',
            va="center",
            fontsize=7,
            color="black"
        )

    ax_footer.text(
        0.5, 0.5,
        "Data source: Official consultation feedback (03 September 2025 - 31 October 2025)\n"
        "URL: https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/\n"
        "12645-Tobacco-taxation-excise-duties-for-manufactured-tobacco-products-updated-rules-_en",
        ha="center", va="center", fontsize=8, color="gray"
    )

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📊 Saved plot: {out_path}")

# =========================
# 📊 Generate both plots
# =========================
plot_keyword_bars(df, "Keyword Frequencies in Feedback\n(Full List)", "keyword_frequencies_full.png")

df_top = df.head(TOP_X)
plot_keyword_bars(df_top, f"Top {TOP_X} Keywords in Feedback\n(Count Only)", f"keyword_frequencies_top{TOP_X}.png")
