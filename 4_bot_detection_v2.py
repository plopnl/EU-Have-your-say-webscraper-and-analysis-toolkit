#!/usr/bin/env python3
# ============================================
# 📄 EU Feedback Burst & Similarity Analysis — Version 2
# Created by Plop (Twitter/GitHub: @plopnl)
# Last updated: 2025-11-20
# Description: Detects suspicious submission patterns in EU "Have Your Say" feedback.
#              Combines timing, similarity, entropy, and rhythm signals to flag coordinated or bot-like entries.
#              Outputs row-level flags, cluster summaries, suspicious clusters, threshold sweep results, and plots.
# Usage: python 4_bot_detection_v2.py
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
# ============================================

import os
import pandas as pd
import numpy as np
import re
import math
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from decimal import Decimal

# =========================
# 🔧 SETTINGS
# =========================
BURST_TIME_THRESHOLD = 60
BURST_MIN_SIZE = 5
SIMILARITY_BASELINE = 0.7
ADAPTIVE_SIM_PERCENTILE = 90
REVIEW_MODE = True
TOP_N_CLUSTERS_FOR_REVIEW = 25
THRESHOLD_SWEEP = True

TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 7
FOOTER_FONTSIZE = 7

DATA_DIR = "data"
FIGURES_DIR = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# =========================
# 📥 LOAD & CLEAN DATA
# =========================
df = pd.read_csv(os.path.join(DATA_DIR, "feedback_all_similarity_scores.csv"))

def clean_text(text):
    return re.sub(r"[^\w\s]", "", str(text).lower().strip())

df["feedback_original_clean"] = df["feedback_original"].fillna("").apply(clean_text)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# =========================
# ⏱️ BURST CLUSTERING
# =========================
df["time_diff_sec"] = df["date"].diff().dt.total_seconds().fillna(0)

burst_cluster_id = 0
cluster_ids = []
for diff in df["time_diff_sec"]:
    if diff > BURST_TIME_THRESHOLD:
        burst_cluster_id += 1
    cluster_ids.append(burst_cluster_id)
df["burst_cluster"] = cluster_ids

# =========================
# 🔤 TEXT ENTROPY
# =========================
def shannon_entropy(text):
    if not text:
        return 0.0
    freq = Counter(text)
    probs = [f / len(text) for f in freq.values()]
    return -sum(p * math.log2(p) for p in probs)

df["char_entropy"] = df["feedback_original_clean"].apply(shannon_entropy)

# =========================
# 📐 CLUSTER METRICS
# =========================
vectorizer = TfidfVectorizer(stop_words="english")

cluster_metrics = []
for cid, group in df.groupby("burst_cluster"):
    texts = group["feedback_original_clean"].tolist()
    times = group["date"].tolist()
    n = len(group)

    if n > 1:
        gaps = group["date"].diff().dt.total_seconds().fillna(0).tolist()[1:]
        avg_gap = float(np.mean(gaps)) if gaps else 0.0
        std_gap = float(np.std(gaps)) if gaps else 0.0
        duration_sec = (times[-1] - times[0]).total_seconds()
    else:
        avg_gap = std_gap = duration_sec = 0.0

    if n >= 2:
        tfidf = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)
        np.fill_diagonal(sim_matrix, np.nan)
        sim_sum = np.nansum(sim_matrix)
        sim_avg = sim_sum / (n * (n - 1))
        sim_max = np.nanmax(sim_matrix)
    else:
        sim_avg = sim_max = 0.0

    avg_entropy = float(group["char_entropy"].mean())

    cluster_metrics.append({
        "burst_cluster": cid,
        "cluster_size": n,
        "avg_similarity": sim_avg,
        "max_similarity_in_group": sim_max,
        "avg_gap_sec": avg_gap,
        "std_gap_sec": std_gap,
        "duration_sec": duration_sec,
        "avg_entropy": avg_entropy
    })

clusters = pd.DataFrame(cluster_metrics)

# =========================
# 🎚️ ADAPTIVE SIMILARITY THRESHOLD
# =========================
eligible = clusters[clusters["cluster_size"] >= BURST_MIN_SIZE]
adaptive_threshold = (
    np.percentile(eligible["avg_similarity"], ADAPTIVE_SIM_PERCENTILE)
    if len(eligible) > 0 else SIMILARITY_BASELINE
)

# =========================
# 🧮 SUSPICION SCORING
# =========================
def score_cluster(row, thresh):
    score = 0
    size_signal = row["cluster_size"] >= BURST_MIN_SIZE
    similarity_signal = row["avg_similarity"] >= thresh
    rhythm_signal = row["std_gap_sec"] <= 10 and row["avg_gap_sec"] > 0
    entropy_signal = row["avg_entropy"] <= 3.5
    duration_signal = (row["duration_sec"] <= (row["cluster_size"] * 45))
    score += size_signal + similarity_signal + rhythm_signal + entropy_signal + duration_signal
    return score, size_signal, similarity_signal, rhythm_signal, entropy_signal, duration_signal

scores, signals = [], []
fixed_thresh = 0.95
for _, r in clusters.iterrows():
    s, a, b, c, d, e = score_cluster(r, fixed_thresh)
    scores.append(s)
    signals.append((a, b, c, d, e))

clusters["suspicion_score"] = scores
clusters["size_signal"] = [t[0] for t in signals]
clusters["similarity_signal"] = [t[1] for t in signals]
clusters["rhythm_signal"] = [t[2] for t in signals]
clusters["entropy_signal"] = [t[3] for t in signals]
clusters["duration_signal"] = [t[4] for t in signals]
clusters["cluster_suspicious"] = clusters["suspicion_score"] >= 3

df = df.merge(clusters, on="burst_cluster", how="left")
df["timing_suspicious"] = df["cluster_size"] >= BURST_MIN_SIZE
df["similarity_suspicious"] = df["avg_similarity"] >= max(SIMILARITY_BASELINE, adaptive_threshold)
df["suspicious_burst"] = df["timing_suspicious"] & df["similarity_suspicious"]

# =========================
# 💾 SAVE OUTPUTS
# =========================
df.to_csv(os.path.join(DATA_DIR, "feedback_with_bot_flags_v2.csv"), index=False)
clusters.to_csv(os.path.join(DATA_DIR, "cluster_summary_v2.csv"), index=False)
clusters[clusters["cluster_suspicious"]].to_csv(os.path.join(DATA_DIR, "suspicious_clusters_v2.csv"), index=False)

if REVIEW_MODE and len(clusters[clusters["cluster_suspicious"]]) > 0:
    top_cids = clusters[clusters["cluster_suspicious"]]["burst_cluster"].head(TOP_N_CLUSTERS_FOR_REVIEW).tolist()
    review_rows = df[df["burst_cluster"].isin(top_cids)].sort_values(["burst_cluster", "date"])
    review_rows.to_csv(os.path.join(DATA_DIR, "review_top_clusters_v2.csv"), index=False)

# =========================
# 📉 Threshold Sweep: Rows vs Clusters
# =========================
THRESHOLD_SWEEP = True
if THRESHOLD_SWEEP:
    from decimal import Decimal

    # Generate thresholds (100% down to 0% by 5%)
    thresholds = [float(Decimal("1.0") - Decimal("0.05") * i) for i in range(21)]
    results = []

    for thresh in thresholds:
        # Row-level counts (submissions)
        similarity_rows = (df["similarity_original"] >= thresh).sum()
        timing_rows = 0
        combined_rows = 0
        score_rows = 0

        # Cluster-level counts (bursts)
        similarity_clusters = 0
        timing_clusters = 0
        combined_clusters = 0
        score_clusters = 0

        for _, row in clusters.iterrows():
            # Signals
            size_signal = row["cluster_size"] >= BURST_MIN_SIZE
            similarity_signal = row["avg_similarity"] >= thresh
            rhythm_signal = row["std_gap_sec"] <= 10 and row["avg_gap_sec"] > 0
            entropy_signal = row["avg_entropy"] <= 3.5
            duration_signal = row["duration_sec"] <= (row["cluster_size"] * 45)

            # Flags
            timing_flag = size_signal
            similarity_flag = similarity_signal
            combined_flag = timing_flag and similarity_flag
            score = int(size_signal) + int(similarity_signal) + int(rhythm_signal) + int(entropy_signal) + int(duration_signal)
            score_flag = score >= 3

            # Row-level (submissions)
            if timing_flag:
                timing_rows += row["cluster_size"]
            if combined_flag:
                combined_rows += row["cluster_size"]
            if score_flag:
                score_rows += row["cluster_size"]

            # Cluster-level (bursts)
            if timing_flag:
                timing_clusters += 1
            if similarity_flag:
                similarity_clusters += 1
            if combined_flag:
                combined_clusters += 1
            if score_flag:
                score_clusters += 1

        print(f"Threshold {thresh:.2f} → Similarity only (rows): {similarity_rows}")

        results.append({
            "threshold": thresh,
            "timing_rows": timing_rows,
            "similarity_rows": similarity_rows,
            "combined_rows": combined_rows,
            "score_rows": score_rows,
            "timing_clusters": timing_clusters,
            "similarity_clusters": similarity_clusters,
            "combined_clusters": combined_clusters,
            "score_clusters": score_clusters,
        })

    # Final sanity check
    print("\n✅ Final check:")
    print(f"Total submissions in dataset: {len(df)}")
    print(f"Sum of cluster_size across clusters: {clusters['cluster_size'].sum()}")

    # Build DataFrame
    df_thresholds = pd.DataFrame(results)
    df_thresholds.to_csv(os.path.join(DATA_DIR, "suspicious_counts_by_threshold.csv"), index=False)
    print("📉 Saved sweep results to data/suspicious_counts_by_threshold.csv")

# =========================
# 📉 Sweep Plot: Suspicious Counts vs Threshold
# =========================
percent_labels = [f"{int(round(t * 100))}%" for t in df_thresholds["threshold"]]

fig, ax = plt.subplots(figsize=(9, 6.5))

# === Main plot ===
ax.plot(df_thresholds["threshold"], df_thresholds["similarity_rows"], marker="s", label="Similar individual entries")
ax.plot(df_thresholds["threshold"], df_thresholds["combined_rows"], marker="^", label="Fast & similar individual entries")
ax.plot(df_thresholds["threshold"], df_thresholds["score_rows"], marker="d", label="Individual entries in bursts with score ≥ 3")
ax.plot(df_thresholds["threshold"], df_thresholds["combined_clusters"], marker="^", linestyle="--", label="Fast & similar bursts")
ax.plot(df_thresholds["threshold"], df_thresholds["similarity_clusters"], marker="x", linestyle="--", label="Bursts with similar entries")
ax.plot(df_thresholds["threshold"], df_thresholds["score_clusters"], marker="*", linestyle="--", label="Suspicious bursts (score ≥ 3)")

# Adaptive threshold line
ax.axvline(x=SIMILARITY_BASELINE, color="gray", linestyle="--", linewidth=1,
           label=f"Adaptive threshold = {SIMILARITY_BASELINE}")

# Parameter notes in legend
ax.plot([], [], ' ', label="--- Parameters ---")
ax.plot([], [], ' ', label=f"Burst time threshold = {BURST_TIME_THRESHOLD}s")
ax.plot([], [], ' ', label=f"Burst minimum size = {BURST_MIN_SIZE}")
ax.plot([], [], ' ', label=f"Similarity baseline = {SIMILARITY_BASELINE}")
ax.plot([], [], ' ', label=f"Adaptive sim percentile = {ADAPTIVE_SIM_PERCENTILE}")

# Axis settings
ax.set_title("Suspicious counts vs similarity threshold", fontsize=TITLE_FONTSIZE)
ax.set_xlabel("Similarity threshold (%)", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("Suspicious count", fontsize=LABEL_FONTSIZE)
ax.grid(True)
ax.legend(loc="upper left", fontsize=LEGEND_FONTSIZE)
ax.invert_xaxis()
ax.set_xticks(df_thresholds["threshold"][::2])
ax.set_xticklabels(percent_labels[::2], rotation=0)

# Footer inline
fig.text(
    0.5, 0.06,
    "Data source: EU consultation feedback (03 September 2025 - 31 October 2025)\n"
    "URL: https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/\n"
    "12645-Tobacco-taxation-excise-duties-for-manufactured-tobacco-products-updated-rules-_en\n"
    "Chart shows how many entries and bursts are flagged as suspicious across similarity thresholds.\n"
    "Method: TF-IDF vectorization + cosine similarity; timing from submission date gaps",
    ha="center", va="center", fontsize=FOOTER_FONTSIZE, color="gray", linespacing=1.5
)

fig.tight_layout(rect=[0, 0.10, 1, 1])
fig_path = os.path.join(FIGURES_DIR, "suspicious_counts_vs_threshold.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"✅ Saved figure: {fig_path}")
