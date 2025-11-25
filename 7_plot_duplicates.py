#!/usr/bin/env python3
# ============================================
# 📄 EU Contributions Duplicate Visualization Script (using similarity values)
# Created by Plop (@plopnl)
# Last updated: 2025-11-21
# Description: Visualizes duplicate contributions using similarity values and submission time differences.
#              Generates gradient scatter plots, grouped plots, and group-colored plots.
# Usage: python 7_plot_duplicates.py
# ============================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap

# =========================
# 🔧 SETTINGS
# =========================
DATA_DIR = "data"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

DUPLICATE_THRESHOLD = 0.95
ZOOM_MARGIN = 0.02
SHOW_STARS = False
TOP_GROUPS = 10

plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 300

TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 7
FOOTER_FONTSIZE = 7

# =========================
# Helper: Bot window legend entry
# =========================
def bot_window_entry(df):
    bot_count = (df["time_diff_days"] < (5/1440)).sum()
    bot_pct = bot_count / len(df) * 100 if len(df) > 0 else 0
    return mlines.Line2D([], [], color="blue", marker="s", linestyle="None",
                         markersize=8,
                         label=f"Bot-like window (<5 min): {bot_count} items ({bot_pct:.1f}%)")

# =========================
# Helper: Footer text
# =========================
def add_footer(ax, location, extra_lines):
    footer_text = (
        "Data source: Official consultation contributions (03 September 2025 - 31 October 2025)\n"
        "URL: https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/\n"
        "12645-Tobacco-taxation-excise-duties-for-manufactured-tobacco-products-updated-rules-_en\n"
        + "\n".join(extra_lines)
    )
    ax.text(
        0.5, -location,
        footer_text,
        ha="center", va="center", fontsize=FOOTER_FONTSIZE, color="gray",
        transform=ax.transAxes
    )

# =========================
# Helper: Gradient Scatter Plot
# =========================
def gradient_scatter(df, title, filename_suffix):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    cmap = LinearSegmentedColormap.from_list("green_orange", ["green", "orange"])

    scatter = ax.scatter(
        df["max_similarity_in_group"],
        df["time_diff_days"],
        c=df["max_similarity_in_group"],
        cmap=cmap,
        alpha=0.6
    )

    if SHOW_STARS:
        exact_dupes = df[df["max_similarity_in_group"] == 1.0]
        ax.scatter(
            exact_dupes["max_similarity_in_group"],
            exact_dupes["time_diff_days"],
            color="red",
            marker="*",
            s=120,
            label="Exact Duplicates"
        )

    sns.kdeplot(
        data=df,
        x="max_similarity_in_group",
        y="time_diff_days",
        levels=5,
        color="black",
        linewidths=1,
        alpha=0.5,
        ax=ax,
        warn_singular=False
    )

    density_proxy = mlines.Line2D([], [], color="black", linestyle="-", label="Density Contours")
    ax.axvline(DUPLICATE_THRESHOLD, color="gray", linestyle="--", label=f"Threshold {int(DUPLICATE_THRESHOLD*100)}%")
    ax.axhline(y=59, color="black", linestyle="--", label="Submission Window Limit (59 days)")

    bot_count = (df["time_diff_days"] < (5/1440)).sum()
    bot_pct = bot_count / len(df) * 100 if len(df) > 0 else 0
    ax.axhspan(0, 5/1440, color="blue", alpha=0.2, label=f"Bot-like window (<5 min): {bot_count} items ({bot_pct:.1f}%)")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x*100)}%"))
    ax.set_xlim(DUPLICATE_THRESHOLD - ZOOM_MARGIN, 1.0 + ZOOM_MARGIN)

    fig.colorbar(scatter, ax=ax, label="Similarity Score", shrink=0.8, pad=0.01, aspect=30)

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Similarity Score (%). Higher values mean more overlap in wording", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Time Difference (days)\nGap between contribution submission dates", fontsize=LABEL_FONTSIZE)

    handles = [*ax.get_legend_handles_labels()[0], density_proxy]
    ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE, loc="upper right", bbox_to_anchor=(1, 0.95))

    fig.tight_layout(rect=[0, 0.08, 1.05, 1])
    add_footer(ax, location=0.16, extra_lines=[
        "Legend includes count and percentage of items falling into the bot-like submission window (<5 min).",
        "Method: TF-IDF vectorization + cosine similarity, time differences computed from submission dates"
    ])

    fig_path = os.path.join(FIGURES_DIR, f"{filename_suffix}_gradient_scatter.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✅ Saved figure: {fig_path}")

# =========================
# Helper: Scatter by Group Coloring
# =========================
def scatter_by_group(df, title, filename_suffix):
    fig, ax = plt.subplots(figsize=(8.5, 6))

    group_sizes = df["group_id"].value_counts()
    top_groups = group_sizes.head(TOP_GROUPS)

    df["group_label"] = df["group_id"].apply(lambda g: g if g in top_groups.index else "Other")
    codes = df["group_label"].astype("category").cat.codes
    palette = sns.color_palette("tab20", n_colors=len(top_groups) + 1)

    ax.scatter(df["max_similarity_in_group"], df["time_diff_days"], c=codes, cmap="tab20", alpha=0.6)

    handles = []
    ordered_labels = list(top_groups.index) + (["Other"] if "Other" in df["group_label"].unique() else [])
    for i, label in enumerate(ordered_labels):
        size = len(df[df["group_label"] == label])
        legend_label = f"Group {label} size {size}" if label != "Other" else f"Other size {size}"
        handles.append(mlines.Line2D([], [], color=palette[i], marker="o", linestyle="None", markersize=8, label=legend_label))

    handles.append(bot_window_entry(df))
    ax.legend(handles=handles, title=f"Top {TOP_GROUPS} Groups (others lumped)", fontsize=LEGEND_FONTSIZE, loc="upper left", bbox_to_anchor=(0.005, 0.95))

    ax.axvline(DUPLICATE_THRESHOLD, color="gray", linestyle="--")
    ax.axhline(y=59, color="black", linestyle="--")
    ax.axhspan(0, 5/1440, color="blue", alpha=0.2)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x*100)}%"))
    ax.set_xlim(DUPLICATE_THRESHOLD - ZOOM_MARGIN, 1.0 + ZOOM_MARGIN)

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Similarity Score (%)\nHigher values mean more overlap in wording", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Time Difference (days)\nGap between contribution submission dates", fontsize=LABEL_FONTSIZE)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    add_footer(ax, location=0.20, extra_lines=[
        "Legend ordered by group size (largest first), includes group sizes and bot-window counts/percentages (<5 min).",
        "Method: TF-IDF vectorization + cosine similarity, time differences computed from submission dates"
    ])

    fig_path = os.path.join(FIGURES_DIR, f"{filename_suffix}_scatter_by_group.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✅ Saved figure: {fig_path}")

# =========================
# Load and Plot Each Dataset
# =========================
files = {
    "feedback_duplicates.csv": "Original Duplicate Contributions",
    "feedback_duplicates_translated.csv": "Translated Duplicate Contributions",
    "feedback_short_duplicates.csv": "Short Duplicate Contributions"
}

for file, title in files.items():
    df = pd.read_csv(os.path.join(DATA_DIR, file))
    base_name = os.path.splitext(file)[0]

    # Strip "feedback_" prefix if present
    if base_name.startswith("feedback_"):
        base_name = base_name[len("feedback_"):]

    if "time_to_closest_match_sec" in df.columns:
        df["time_diff_days"] = df["time_to_closest_match_sec"] / (3600 * 24)
    else:
        df["time_diff_days"] = 0

    if "max_similarity_in_group" not in df.columns:
        print(f"⚠️ {file} missing max_similarity_in_group column")
        continue

    # --- Plot individuals ---
    gradient_scatter(
        df,
        f"{title} (Individual Submissions, Green→Orange Gradient)",
        f"{base_name}_individual"
    )

    # --- Plot grouped by group_id ---
    if "group_id" in df.columns:
        df_grouped = (
            df.groupby("group_id", as_index=False)
              .agg({"max_similarity_in_group": "max", "time_diff_days": "min"})
        )
        gradient_scatter(
            df_grouped,
            f"{title} (Grouped by group_id, Green→Orange Gradient)",
            f"{base_name}_grouped"
        )

        # --- Plot individuals colored by group_id ---
        scatter_by_group(
            df,
            f"{title} (Individuals colored by group_id,\nTop {TOP_GROUPS} Groups distinct)",
            f"{base_name}_group_colored"
        )
