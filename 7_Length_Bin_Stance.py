# ============================================
# 📊 Feedback Length Bin & Stance Plot Script
# Created by Plop (@plopnl)
# Description: Reads feedback CSV + stance labels, bins submissions by length,
#              outputs enriched CSV, and generates bar graphs
#              of stance vs length bins. Graphs saved in figures/.
# ============================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 🔧 SETTINGS
# =========================
INPUT_CSV = "data/feedback_details_with_lengths.csv"
STANCE_CSV = "data/stance_detection/stance_supervised.csv"
ATTACH_UNIQUE = "data/attachment_unique.csv"
ATTACH_DUPLICATES = "data/attachment_duplicates.csv"
OUTPUT_CSV = "data/feedback_binned.csv"
FIGURE_DIR = "figures"
SUBDIR = os.path.join(FIGURE_DIR, "stance_length_bins")
os.makedirs(SUBDIR, exist_ok=True)

# === Font sizes ===
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 9
FOOTER_FONTSIZE = 7

# === Bin setup ===
BIN_LABELS = ["Ultra-short (0–10)", "Short (11–50)", "Medium (51–200)", "Long (>200)"]
BIN_EDGES = [0, 10, 50, 200, np.inf]
BIN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue, orange, green, red

# === Stance categories ===
STANCE_ORDER = ["Against", "For", "Unclear"]

# === Normalize labels ===
def normalize_group_label(label):
    if label == "full":
        return "All submissions"
    elif label.upper() == "NGO":
        return "NGO submissions"
    else:
        return f"{label.replace('_', ' ').title()} submissions"

# =========================
# 📈 PLOTTING FUNCTION
# =========================
def plot_stance_length_bins(df, label="full"):
    # === Clean + bin ===
    df = df.dropna(subset=["stance_supervised", "word_count_original"])
    df["stance_supervised"] = df["stance_supervised"].astype(str).str.strip().str.title()
    df["stance_supervised"] = pd.Categorical(df["stance_supervised"], categories=STANCE_ORDER, ordered=True)
    df["word_count_original"] = pd.to_numeric(df["word_count_original"], errors="coerce")
    df["length_bin"] = pd.cut(df["word_count_original"], bins=BIN_EDGES, labels=BIN_LABELS)

    # === Group and reindex ===
    stance_bins = df.groupby(["stance_supervised", "length_bin"]).size().unstack(fill_value=0)
    stance_bins = stance_bins.reindex(index=STANCE_ORDER)
    present_bins = [lab for lab in BIN_LABELS if lab in stance_bins.columns]
    stance_bins = stance_bins[present_bins]

    # === Plot ===
    fig, ax = plt.subplots(figsize=(10, 6))
    stance_bins.plot(kind="bar", stacked=False, ax=ax, color=[BIN_COLORS[BIN_LABELS.index(l)] for l in present_bins])

    title_group = normalize_group_label(label)
    ax.set_title(f"{title_group}: Submission length bins by stance", fontsize=TITLE_FONTSIZE)
    ax.set_ylabel("Submission count", fontsize=LABEL_FONTSIZE)
    ax.set_xlabel("Stance", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", rotation=0, labelsize=LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=LABEL_FONTSIZE)
    ax.legend(title="Length bin", fontsize=LEGEND_FONTSIZE)

    for container in ax.containers:
        ax.bar_label(container, fontsize=LEGEND_FONTSIZE, label_type="edge")

    fig.text(
        0.5, 0.09,
        f"{title_group} shown by stance and document length bin.\n"
        "Shorter texts are mostly citizen comments; longer ones tend to come from NGOs and academics.\n"
        "Data source: EU consultation feedback (03 September 2025 – 31 October 2025)\n"
        "URL: https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/\n"
        "12645-Tobacco-taxation-excise-duties-for-manufactured-tobacco-products-updated-rules-_en",
        ha="center", va="center", fontsize=FOOTER_FONTSIZE, color="gray", linespacing=1.5
    )

    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig_path = os.path.join(SUBDIR, f"{label}.png")
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"📊 Saved: {fig_path}")


# =========================
# 🚀 MAIN EXECUTION
# =========================
df_lengths = pd.read_csv(INPUT_CSV)
df_stance = pd.read_csv(STANCE_CSV)

df = pd.merge(df_lengths, df_stance[["feedbackId", "stance_supervised"]], on="feedbackId", how="left")

# === Attachments integration ===
df_unique = pd.read_csv(ATTACH_UNIQUE)
df_dupes = pd.read_csv(ATTACH_DUPLICATES)
df_attach = pd.concat([df_unique, df_dupes], ignore_index=True)
df_attach_sum = df_attach.groupby("feedbackId")[["word_count","char_count"]].sum().reset_index()
df = pd.merge(df, df_attach_sum, on="feedbackId", how="left")
df["word_count_original"] = df["word_count_original"] + df["word_count"].fillna(0)
df["char_count_original"] = df["char_count_original"] + df["char_count"].fillna(0)
df = df.drop(columns=["word_count","char_count"])
del df_unique, df_dupes, df_attach, df_attach_sum

df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Enriched CSV saved to {OUTPUT_CSV} ({len(df)} rows)")

# Full dataset plot
plot_stance_length_bins(df, label="full")

# Per userType plots
for ut in sorted(df["userType"].dropna().unique()):
    df_ut = df[df["userType"] == ut]
    plot_stance_length_bins(df_ut, label=ut)

# =========================
# 📋 DEBUG: List feedback IDs by group + stance
# =========================
def list_feedback_ids(df, group, stance):
    df_filtered = df[
        (df["userType"].astype(str).str.strip().str.lower() == group.lower()) &
        (df["stance_supervised"].astype(str).str.strip().str.title() == stance)
    ]
    ids = df_filtered["feedbackId"].dropna().unique()
    print(f"🧾 {group} submissions with stance '{stance}': {len(ids)} found")
    print("Feedback IDs:", ", ".join(map(str, ids)))
list_feedback_ids(df, group="CONSUMER_ORGANISATION", stance="For")
