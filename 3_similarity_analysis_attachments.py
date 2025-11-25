#!/usr/bin/env python3
# ============================================
# 📄 EU Feedback Similarity Visualization Script
# Created by Plop (Twitter/GitHub: @plopnl)
# Description: Visualizes consultation feedback data from the EU "Have Your Say" portal.
#              Combines four plots to explore textual similarity across submitted attachments:
#              - Scatter plot: similarity vs submission timing (clustered by anchor)
#              - Bar plot: attachment cluster sizes with match percentages
#              - Gradient scatter: all pairwise similarity scores with density contours
#              - Histogram: distribution of similarity scores across all document pairs
#              Highlights coordinated responses, exact duplicates, and suspiciously short submission windows.
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
# ============================================


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

PAIR_FILE = "data/similarity_with_attachments.csv"
ATTACH_FILE = "data/attachment_duplicates.csv"
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

SCATTER_FILE = os.path.join(FIGDIR, "scatter_by_group.png")
BAR_FILE = os.path.join(FIGDIR, "cluster_sizes_barplot.png")
GRADIENT_FILE = os.path.join(FIGDIR, "similarity_gradient_plot.png")
HISTOGRAM_FILE = os.path.join(FIGDIR, "similarity_score_distribution.png")

# =========================
# Font settings
# =========================
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 9
TICK_FONTSIZE = 7
LEGEND_FONTSIZE = 7
FOOTER_FONTSIZE = 7

SUBMISSION_LIMIT_DAYS = 59
BOT_WINDOW_SEC = 300

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
# Load attachment metadata
# =========================
att = pd.read_csv(ATTACH_FILE)
required_cols = {"feedbackId", "filename", "clean_filename", "group_id", "group_size"}
missing = required_cols - set(att.columns)
if missing:
    raise ValueError(f"{ATTACH_FILE} missing columns: {sorted(missing)}")

fid_to_gid = dict(zip(att["feedbackId"], att["group_id"]))
anchors = att.groupby("group_id").first()[["clean_filename", "group_size"]]
group_map = anchors.to_dict(orient="index")

# =========================
# Load pairwise similarity
# =========================
df = pd.read_csv(PAIR_FILE)
df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce")
df["time_diff_sec"] = pd.to_numeric(df["time_diff_sec"], errors="coerce")
df["time_diff_days"] = pd.to_numeric(df["time_diff_days"], errors="coerce")
df = df.dropna(subset=["similarity", "time_diff_days", "time_diff_sec"])
df = df[df["time_diff_days"] >= 0]

# =========================
# Assign group and color
# =========================
df["group_id"] = df["feedbackId_i"].map(fid_to_gid)

def assign_color_group(gid):
    if pd.isna(gid):
        return "unclustered"
    return group_map.get(gid, {"clean_filename": "unclustered"})["clean_filename"]

df["color_group"] = df["group_id"].map(assign_color_group)

# =========================
# Legend labels sorted by group size
# =========================
legend_labels = {
    info["clean_filename"]: f"{info['group_size']}x {info['clean_filename']}"
    for gid, info in group_map.items()
}
sorted_legend = dict(sorted(
    legend_labels.items(),
    key=lambda x: int(x[1].split("x")[0]),
    reverse=True
))

# =========================
# Palette
# =========================
labels_in_plot = sorted(set(df["color_group"]))
palette = sns.color_palette("husl", len(labels_in_plot))
group_palette = dict(zip(labels_in_plot, palette))

# =========================
# Average similarity score per cluster
# =========================
def parse_scores(s):
    try:
        return [float(x) for x in s.split(";") if x.strip()]
    except:
        return []

att["scores_list"] = att["similarity_scores"].apply(parse_scores)
att["avg_similarity"] = att["scores_list"].apply(lambda x: sum(x)/len(x) if x else None)
att["match_pct"] = att["avg_similarity"] * 100

cluster_stats = att.groupby("group_id").agg({
    "clean_filename": "first",
    "group_size": "first",
    "match_pct": "mean"
}).reset_index()

cluster_stats["label"] = cluster_stats["group_size"].astype(str) + "x " + cluster_stats["clean_filename"]
cluster_stats = cluster_stats.sort_values("group_size", ascending=False)

# =========================
# Plot 1: Scatter by group
# =========================
fig, ax = plt.subplots(figsize=(10, 6))
scatter = sns.scatterplot(
    data=df,
    x="similarity",
    y="time_diff_days",
    hue="color_group",
    palette=group_palette,
    ax=ax,
    s=40,
    edgecolor="none",
    legend="full"
)

ax.axhline(SUBMISSION_LIMIT_DAYS, ls="--", color="black", label=f"Submission Window Limit ({SUBMISSION_LIMIT_DAYS} days)")
ax.axhline(0, color="blue", lw=2, label="Bot-like window (<5 min)")

ax.text(
    0.5, 1.08,
    "Pairwise Attachment Similarity ≥95% vs Submission Timing (Clustered by Anchor)",
    ha="center",
    va="center",
    fontsize=TITLE_FONTSIZE,
    transform=ax.transAxes
)
summary_line = f"{len(df)} high-similarity pairs across {len(sorted_legend)} anchor clusters"
ax.text(
	0.5, 1.02, 
	summary_line, 
	ha="center", 
	fontsize=LABEL_FONTSIZE, 
	transform=ax.transAxes
)
ax.set_xlabel("Similarity Score (%). Higher values mean more overlap in wording", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("Time Difference (days) Gap between feedback submission dates", fontsize=LABEL_FONTSIZE)
ax.tick_params(labelsize=TICK_FONTSIZE)

handles, labels = scatter.get_legend_handles_labels()
handle_map = dict(zip(labels, handles))
final_labels = list(sorted_legend.keys()) + [lbl for lbl in labels if lbl not in sorted_legend]
final_handles = [handle_map[lbl] for lbl in final_labels]
final_texts = [sorted_legend.get(lbl, lbl) for lbl in final_labels]

ax.legend(
	final_handles, 
	final_texts, 
	loc="upper center",
	title="Group_size x Anchor filename", 
	fontsize=LEGEND_FONTSIZE,
    bbox_to_anchor=(0.5, 0.95)
)
add_centered_footer(
	ax,
	location=0.2,
	extra_lines=[
        "Method: TF-IDF vectorization + cosine similarity, time differences computed from submission dates",
        "Each dot represents a pair of submissions with ≥95% similar attachments,",
        "plotted by the time gap between their submission dates. Colors indicate anchor clusters.",
        "Each pair is plotted once, using the earlier submission as reference."
    ]
)
plt.tight_layout(rect=[0, -0.02, 1, 1])
#plt.tight_layout()
plt.savefig(SCATTER_FILE, dpi=300)
plt.close()
print(f"✅ Saved {SCATTER_FILE}")

# =========================
# Plot 2: Bar plot of cluster sizes with match percentages
# =========================
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, cluster_stats["group_size"].max() + 0.4)

sns.barplot(data=cluster_stats, x="group_size", y="label", ax=ax, palette="viridis")

for patch, (_, row) in zip(ax.patches, cluster_stats.iterrows()):
    x = patch.get_width()                          # actual bar end
    y = patch.get_y() + patch.get_height() / 2     # bar center

    ax.annotate(
        f'{row["match_pct"]:.1f}%',
        xy=(x, y),
        xytext=(36, 0),            # 4 points left from the tip
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=LABEL_FONTSIZE,
        color="black",
        clip_on=True
    )

# Hide default y-axis labels
ax.set_yticklabels([])

# Place filenames at x=0
for i, row in cluster_stats.iterrows():
    ax.text(
        0.05, i,
        row["clean_filename"],
        ha="left",
        va="center",
        fontsize=LABEL_FONTSIZE
    )

# Title + subtitle
ax.text(
    0.5, 1.08,
    "Attachment Cluster Sizes",
    ha="center",
    va="center",
    fontsize=TITLE_FONTSIZE,
    transform=ax.transAxes
)
ax.text(
    0.5, 1.02,
    f"{cluster_stats.shape[0]} anchor clusters",
    ha="center",
    fontsize=LABEL_FONTSIZE,
    transform=ax.transAxes
)

# Axis labels
ax.set_xlabel("Group Size", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("Filename", fontsize=LABEL_FONTSIZE)
ax.tick_params(labelsize=TICK_FONTSIZE)

# Footer
add_centered_footer(
    ax,
    location=0.18,
    extra_lines=[
        "Each bar shows the number of attachments clustered under a shared anchor filename.",
        "Match percentage indicates average similarity across all attachments in the cluster."
    ]
)

plt.tight_layout(rect=[0, -0.02, 1, 1])
plt.savefig(BAR_FILE, dpi=300)
plt.close()
print(f"✅ Saved {BAR_FILE}")

# =========================
# Plot 3: Gradient + contour plot
# =========================
fig, ax = plt.subplots(figsize=(10, 6))
scatter = sns.scatterplot(
    data=df,
    x="similarity",
    y="time_diff_days",
    hue="similarity",
    palette="YlOrBr",
    ax=ax,
    s=40,
    edgecolor="none",
    legend=False
)

sns.kdeplot(
    data=df,
    x="similarity",
    y="time_diff_days",
    levels=5,
    color="gray",
    linewidths=1,
    ax=ax
)

ax.axhline(SUBMISSION_LIMIT_DAYS, ls="--", color="black", label=f"Submission Window Limit ({SUBMISSION_LIMIT_DAYS} days)")
ax.axhline(0, color="blue", lw=2, label="Bot-like window (<5 min)")

ax.text(
    0.5, 1.08,
    "Similarity vs Time Difference (Green→Orange Gradient)",
    ha="center",
    va="center",
    fontsize=TITLE_FONTSIZE,
    transform=ax.transAxes
)
summary_line = f"{len(df)} high-similarity pairs with attachment metadata"
ax.text(
    0.5, 1.02,
    summary_line,
    ha="center",
    fontsize=LABEL_FONTSIZE,
    transform=ax.transAxes
)

ax.set_xlabel("Similarity Score (%). Higher values mean more overlap in wording", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("Time Difference (days) Gap between feedback submission dates", fontsize=LABEL_FONTSIZE)
ax.tick_params(labelsize=TICK_FONTSIZE)

# Colorbar
norm = plt.Normalize(df["similarity"].min(), df["similarity"].max())
sm = plt.cm.ScalarMappable(cmap="YlOrBr", norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Similarity Score", fontsize=LABEL_FONTSIZE)

add_centered_footer(
    ax,
    location=0.2,
    extra_lines=[
        "Method: TF-IDF vectorization + cosine similarity, time differences computed from submission dates",
        f"Each dot represents a pair of submissions with ≥95% similar attachments,",
        f"plotted by the time gap between their submission dates. Gradient indicates similarity score.",
        f"Each pair is plotted once, using the earlier submission as reference."
    ]
)

# Layout margins
plt.tight_layout(rect=[0, -0.02, 1, 1])
plt.savefig(GRADIENT_FILE, dpi=300)
plt.close()
print(f"✅ Saved {GRADIENT_FILE}")


# =========================
# Plot 4: Histogram of similarity scores
# =========================
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df["similarity"], bins=30, kde=True, ax=ax, color="steelblue")

ax.set_title("Distribution of Similarity Scores", fontsize=TITLE_FONTSIZE)
ax.set_xlabel("Similarity Score (%). Higher values mean more overlap in wording", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("Frequency", fontsize=LABEL_FONTSIZE)
ax.tick_params(labelsize=TICK_FONTSIZE)

add_centered_footer(ax, extra_lines=[f"Pairs with attachments: {len(df)}"])
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(HISTOGRAM_FILE, dpi=300)
plt.close()
print(f"✅ Saved {HISTOGRAM_FILE}")
