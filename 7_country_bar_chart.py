#!/usr/bin/env python3
# ============================================
# 📊 EU Contributions by Country Visualization Script
# Created by Plop (@plopnl)
# Last updated: 2025-11-21
# Description: Visualizes consultation contributions by country from the EU "Have Your Say" portal.
#              Shows both raw counts and percentage shares as overlayed bars.
#              Includes footer metadata for consistency with other plots.
# Usage: python 8_country_barchart.py
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
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
INPUT_FILE = os.path.join(DATA_DIR, "feedback_totals_by_country_named.csv")

TOP_X = 10   # number of top countries to show in second plot

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
# 📥 Load and prepare data
# =========================
df = pd.read_csv(INPUT_FILE)
df = df.sort_values("feedback_count", ascending=False).reset_index(drop=True)

# Scale percentage to match count axis
max_count = df["feedback_count"].max()
df["percentage_scaled"] = df["percentage"] / 100 * max_count

# =========================
# 🎨 Function to plot chart
# =========================
def plot_country_bars(data, title, filename):
    sns.set(style="whitegrid")
    fig, ax, ax_footer = plot_with_footer(figsize=(10, 12))

    # Feedback count bars
    sns.barplot(
        data=data,
        y="country_name",
        x="feedback_count",
        color="skyblue",
        label="Feedback Count",
        ax=ax
    )

    # Overlay percentage bars
    sns.barplot(
        data=data,
        y="country_name",
        x="percentage_scaled",
        color="orange",
        alpha=0.6,
        label="Percentage (scaled)",
        ax=ax
    )

    # Extend x-axis for labels
    ax.set_xlim(0, data["feedback_count"].max() * 1.15)

    # Titles and labels
    ax.set_xlabel("Number of Contributions")
    ax.set_ylabel("Country")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.legend()

    # Reduce font size of country labels
    ax.tick_params(axis="y", labelsize=8)

    # Annotate each bar with count + percentage
    for i, row in data.iterrows():
        ax.text(
            row["feedback_count"] + data["feedback_count"].max() * 0.01,
            i,
            f'{row["feedback_count"]} ({row["percentage"]:.1f}%)',
            va="center",
            fontsize=7,
            color="black"
        )

    # Footer text
    ax_footer.text(
        0.5, 0.5,
        "Data source: Official consultation contributions (03 September 2025 - 31 October 2025)\n"
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
plot_country_bars(df, "Contributions by Country\n(Count + Percentage Overlay)", "country_contributions_full.png")

df_top = df.head(TOP_X)
plot_country_bars(df_top, f"Top {TOP_X} Countries by Contributions\n(Count + Percentage Overlay)", f"country_contributions_top{TOP_X}.png")
