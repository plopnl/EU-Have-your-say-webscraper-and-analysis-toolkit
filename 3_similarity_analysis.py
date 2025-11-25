# ============================================
# 📄 EU Feedback Similarity Visualization Script
# Created by Plop (Twitter/GitHub: @plopnl)
# Last updated: 2025-11-20
# Description: Visualizes consultation feedback data from the EU "Have Your Say" portal.
#              Produces multiple plots (scatter, histogram + KDE, gradient scatter)
#              to explore relationships between text similarity scores and submission timing.
#              Highlights exact duplicates, suspiciously short submission windows,
#              and distribution of similarity values.
# Usage: python 3_similarity_analysis.py
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
# ============================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap

# =========================
# Settings
# =========================
SHOW_STARS = False   # toggle exact duplicate stars

# 🎨 Matplotlib Settings
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 300

# Font sizes (centralized for easy adjustment)
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 7
FOOTER_FONTSIZE = 7

# Output path figures
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# =========================
# 📥 LOAD DATA
# =========================
similar_df = pd.read_csv(os.path.join("data", "feedback_similarity_topn.csv"))
feedback_df = pd.read_csv(os.path.join("data", "feedback_details_with_lengths.csv"))
feedback_df["date"] = pd.to_datetime(feedback_df["date"])
groups_df = pd.read_csv(os.path.join("data", "feedback_duplicates.csv"))[["feedbackId", "group_id"]]
feedback_df = feedback_df.merge(groups_df, on="feedbackId", how="left")

similar_df = similar_df.merge(
    feedback_df[["feedbackId", "date", "group_id"]],
    left_on="feedbackId_i",
    right_on="feedbackId",
    how="left"
).rename(columns={"date": "date_i", "group_id": "group_id_i"}).drop("feedbackId", axis=1)

similar_df = similar_df.merge(
    feedback_df[["feedbackId", "date", "group_id"]],
    left_on="feedbackId_j",
    right_on="feedbackId",
    how="left"
).rename(columns={"date": "date_j", "group_id": "group_id_j"}).drop("feedbackId", axis=1)

similar_df["time_diff_sec"] = (similar_df["date_i"] - similar_df["date_j"]).abs().dt.total_seconds()
similar_df["time_diff_days"] = similar_df["time_diff_sec"] / (3600 * 24)

# =========================
# Helpers
# =========================
def add_centered_footer(ax, location=0.06, extra_lines=None):
    base_lines = [
        "Data source: Official consultation feedback (03 September 2025 - 31 October 2025)",
        "URL: https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/",
        "12645-Tobacco-taxation-excise-duties-for-manufactured-tobacco-products-updated-rules-_en"
    ]
    if extra_lines:
        base_lines.extend(extra_lines)

    footer_text = "\n".join(base_lines)
    ax.text(
        0.5, -location,
        footer_text,
        ha="center", va="center",
        fontsize=FOOTER_FONTSIZE,
        color="gray",
        transform=ax.transAxes
    )

# =========================
# Plot 1: Scatter plot
# =========================
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(similar_df["similarity"], similar_df["time_diff_days"], alpha=0.3, color="slateblue")
ax.set_title("Similarity vs Time Difference (All Matched Pairs)", fontsize=TITLE_FONTSIZE)
ax.set_xlabel("Similarity Score (%). Higher values mean more overlap in wording", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("Time Difference (days)\nGap between feedback submission dates", fontsize=LABEL_FONTSIZE)
ax.axhline(y=59, color="black", linestyle="--", label="Submission Window Limit (59 days)")
bot_count = (similar_df["time_diff_days"] < (5/1440)).sum()
bot_pct = bot_count / len(similar_df) * 100 if len(similar_df) > 0 else 0
ax.axhspan(0, 5/1440, color="blue", alpha=0.2, label=f"Bot-like window (<5 min): {bot_count} items ({bot_pct:.1f}%)")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x*100)}%"))
ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right", bbox_to_anchor=(1, 0.95))
fig.tight_layout(rect=[0, 0.09, 1, 1])

add_centered_footer(
    ax,
    location=0.17,
    extra_lines=[
        "Legend includes count and percentage of items falling into the bot-like submission window (<5 min).",
        "Method: TF-IDF vectorization + cosine similarity, time differences computed from submission dates"
    ]
)

fig_path = os.path.join(FIGURES_DIR, "similarity_vs_time_scatter.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"✅ Saved figure: {fig_path}")

# =========================
# Plot 2: Histogram + KDE
# =========================
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(similar_df["similarity"], bins=50, kde=True, color="blue", ax=ax)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x*100)}%"))
ax.set_title("Distribution of Similarity Scores", fontsize=TITLE_FONTSIZE)
ax.set_xlabel("Similarity Score (%). Higher values mean more overlap in wording", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("Frequency", fontsize=LABEL_FONTSIZE)

fig.tight_layout(rect=[0, 0.08, 1, 1])

add_centered_footer(
    ax,
    location=0.18,
    extra_lines=[
        "Method: TF-IDF vectorization + cosine similarity, time differences computed from submission dates"
    ]
)

fig_path = os.path.join(FIGURES_DIR, "similarity_distribution_kde.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"✅ Saved figure: {fig_path}")

# =========================
# Plot 3: Gradient Scatter (Green→Orange)
# =========================
fig, ax = plt.subplots(figsize=(8.5, 6))
green_orange_cmap = LinearSegmentedColormap.from_list("green_orange", ["green", "orange"])

scatter = ax.scatter(
    similar_df["similarity"],
    similar_df["time_diff_days"],
    c=similar_df["similarity"],
    cmap=green_orange_cmap,
    alpha=0.6
)

if SHOW_STARS:
    exact_dupes = similar_df[similar_df["similarity"] == 1.0]
    ax.scatter(
        exact_dupes["similarity"],
        exact_dupes["time_diff_days"],
        color="red",
        marker="*",
        s=120,
        label="Exact Duplicates"
    )

sns.kdeplot(
    data=similar_df,
    x="similarity",
    y="time_diff_days",
    levels=5,
    color="black",
    linewidths=1,
    alpha=0.5,
    ax=ax
)

density_proxy = mlines.Line2D([], [], color="black", linestyle="-", label="Density Contours")
ax.axhline(y=59, color="black", linestyle="--", label="Submission Window Limit (59 days)")
bot_count = (similar_df["time_diff_days"] < (5/1440)).sum()
bot_pct = bot_count / len(similar_df) * 100 if len(similar_df) > 0 else 0
ax.axhspan(0, 5/1440, color="blue", alpha=0.2,
           label=f"Bot-like window (<5 min): {bot_count} items ({bot_pct:.1f}%)")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x*100)}%"))

fig.colorbar(scatter, ax=ax, label="Similarity Score", shrink=0.8, pad=0.01, aspect=30)

ax.set_title("Similarity vs Time Difference (Green→Orange Gradient)", fontsize=TITLE_FONTSIZE)
ax.set_xlabel("Similarity Score (%). Higher values mean more overlap in wording", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("Time Difference (days)\nGap between feedback submission dates", fontsize=LABEL_FONTSIZE)

handles = [*ax.get_legend_handles_labels()[0], density_proxy]
ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right", bbox_to_anchor=(1, 0.95))

fig.tight_layout(rect=[0, 0.08, 1.05, 1])

add_centered_footer(
    ax,
    location=0.16,
    extra_lines=[
        "Legend includes count and percentage of items falling into the bot-like submission window (<5 min).",
        "Method: TF-IDF vectorization + cosine similarity, time differences computed from submission dates"
    ]
)

fig_path = os.path.join(FIGURES_DIR, "similarity_vs_time_gradient.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"✅ Saved figure: {fig_path}")
