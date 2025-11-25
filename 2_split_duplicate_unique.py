# ============================================
# 📄 EU Feedback Deduplication & Grouping Script
# Created by Plop (Twitter/GitHub: @plopnl)
# Last updated: 2025-11-20
# Description: Detects near-duplicate consultation feedback entries from the EU "Have Your Say" portal
#              using TF-IDF vectorization + cosine similarity. Groups similar entries into clusters,
#              assigns group IDs, and calculates time differences to closest matches.
#              Separates short entries, translated duplicates, and unique feedback into
#              dedicated CSV outputs for further analysis.
# Usage: python 2_split_duplicate_unique.py
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
# ============================================

import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import os

# =========================
# 🔧 SETTINGS
# =========================
INPUT_FILE = os.path.join("data", "feedback_details_with_lengths.csv")
OUTPUT_DIR = "data"
DUPLICATE_THRESHOLD = 0.95
SHORT_WORD_LIMIT = 10

# =========================
# 🧼 TEXT CLEANING
# =========================
def remove_punctuation(text):
    return re.sub(r"[^\w\s']", "", text)

EXCLUDE_STRINGS = [
    "see attached file", "no comment", "na", "refer to document",
    "please see attachment", "attached document", "see file", "see annex",
    "see attachment", "please see the attached document",
    "see attached document", "please see the attached file", ""
]
EXCLUDE_STRINGS = [remove_punctuation(s.lower().strip()) for s in EXCLUDE_STRINGS]

# =========================
# 📥 LOAD & PREP DATA
# =========================
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])  # ensure datetime

df["feedback_original_clean"] = (
    df["feedback_original"].fillna("").str.strip().str.lower().apply(remove_punctuation)
)
df["feedback_translated_clean"] = (
    df["feedback_translated"].fillna("").str.strip().str.lower().apply(remove_punctuation)
)

# Filter out meaningless entries
meaningful_mask = ~(
    df["feedback_original_clean"].isin(EXCLUDE_STRINGS) &
    df["feedback_translated_clean"].isin(EXCLUDE_STRINGS)
)
print(f"🗑️ Dropped {(~meaningful_mask).sum()} meaningless entries")
df = df[meaningful_mask].reset_index(drop=True)

# Word counts
df["combined_clean_text"] = df["feedback_original_clean"] + " " + df["feedback_translated_clean"]
df["word_count_combined"] = df["combined_clean_text"].str.split().apply(len)
df["word_count_original"] = (
    df["feedback_original"].fillna("").str.replace(r"\s+", " ", regex=True)
    .str.strip().str.split().apply(len)
)

# =========================
# ✂️ SPLIT SHORT ENTRIES
# =========================
df_short = df[df["word_count_original"] <= SHORT_WORD_LIMIT].reset_index(drop=True)
df_main = df[df["word_count_original"] > SHORT_WORD_LIMIT].reset_index(drop=True)

# =========================
# 📐 Compute similarity matrix for originals
# =========================
print("🔍 Computing similarity matrix for original texts...")
vectorizer_orig = TfidfVectorizer(stop_words="english")
tfidf_orig = vectorizer_orig.fit_transform(df["feedback_original_clean"].tolist())
sim_matrix_orig = cosine_similarity(tfidf_orig)
np.fill_diagonal(sim_matrix_orig, np.nan)

df["similarity_original"] = np.nanmax(sim_matrix_orig, axis=1)

# =========================
# 📐 Compute similarity matrix for translations
# =========================
print("🔍 Computing similarity matrix for translated texts...")
vectorizer_trans = TfidfVectorizer(stop_words="english")
tfidf_trans = vectorizer_trans.fit_transform(df["feedback_translated_clean"].tolist())
sim_matrix_trans = cosine_similarity(tfidf_trans)
np.fill_diagonal(sim_matrix_trans, np.nan)

df["similarity_translated"] = np.nanmax(sim_matrix_trans, axis=1)

# =========================
# 🔁 GROUP DETECTION FUNCTION
# =========================
def detect_similarity_groups(df_subset, sim_matrix, similarity_threshold):
    idx = df_subset.index.to_list()
    graph = defaultdict(set)
    sim_scores = defaultdict(list)

    for i_pos, i in enumerate(idx):
        for j_pos, j in enumerate(idx):
            if i != j and sim_matrix[i, j] > similarity_threshold:
                id_i = df_subset.loc[i, "feedbackId"]
                id_j = df_subset.loc[j, "feedbackId"]
                graph[id_i].add(id_j)
                graph[id_j].add(id_i)
                sim_scores[id_i].append(sim_matrix[i, j])
                sim_scores[id_j].append(sim_matrix[i, j])

    visited = set()
    groups = []
    for node in graph:
        if node not in visited:
            stack = [node]
            group = set()
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    group.add(current)
                    stack.extend(graph[current] - visited)
            groups.append(group)

    rows = []
    for group_id, group in enumerate(groups):
        for fid in group:
            row = df_subset[df_subset["feedbackId"] == fid].iloc[0].to_dict()
            scores = sim_scores.get(fid, [])
            max_sim = max(scores, default=None)
            row.update({
                "group_id": group_id,
                "group_size": len(group),
                "matched_feedbackIds": "; ".join(str(i) for i in sorted(group)),
                "max_similarity_in_group": round(max_sim, 4) if max_sim is not None else None,
                "similarity_scores": "; ".join(f"{s:.4f}" for s in scores)
            })
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)

# =========================
# 🔍 RUN GROUP DETECTION
# =========================
df_duplicates_original = detect_similarity_groups(df_main, sim_matrix_orig, DUPLICATE_THRESHOLD)
df_short_duplicates_original = detect_similarity_groups(df_short, sim_matrix_orig, DUPLICATE_THRESHOLD)

df_duplicates_translated = detect_similarity_groups(df_main, sim_matrix_trans, DUPLICATE_THRESHOLD)
df_short_duplicates_translated = detect_similarity_groups(df_short, sim_matrix_trans, DUPLICATE_THRESHOLD)

# =========================
# ➕ ADD TIME TO DUPLICATES
# =========================
def format_time_diff(seconds):
    if seconds is None:
        return None
    if seconds < 60:
        return f"{int(seconds)} sec"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f} hr"
    else:
        return f"{seconds / 86400:.1f} days"

def add_time_to_closest_match(df_duplicates, df_all):
    time_rows = []
    for _, row in df_duplicates.iterrows():
        fid = row["feedbackId"]
        group_ids = [int(x) for x in row["matched_feedbackIds"].split("; ")]
        group_ids = [gid for gid in group_ids if gid != fid]
        if group_ids:
            date_fid = df_all.loc[df_all["feedbackId"] == fid, "date"].iloc[0]
            diffs = []
            for gid in group_ids:
                date_gid = df_all.loc[df_all["feedbackId"] == gid, "date"].iloc[0]
                diff_sec = abs((date_fid - date_gid).total_seconds())
                diffs.append(diff_sec)
            min_diff = min(diffs)
            time_rows.append((fid, min_diff, format_time_diff(min_diff)))
        else:
            time_rows.append((fid, None, None))
    time_df = pd.DataFrame(time_rows, columns=["feedbackId", "time_to_closest_match_sec", "time_to_closest_match_fmt"])
    return df_duplicates.merge(time_df, on="feedbackId", how="left")
    
# =========================
# ➕ ADD TIME TO DUPLICATES
# =========================
df_duplicates_original = add_time_to_closest_match(df_duplicates_original, df)
df_short_duplicates_original = add_time_to_closest_match(df_short_duplicates_original, df)

df_duplicates_translated = add_time_to_closest_match(df_duplicates_translated, df)
df_short_duplicates_translated = add_time_to_closest_match(df_short_duplicates_translated, df)

# =========================
# 💾 SAVE OUTPUTS
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Original duplicates
df_duplicates_original.to_csv(os.path.join(OUTPUT_DIR, "feedback_duplicates.csv"), index=False)
print(f"✅ Saved {len(df_duplicates_original)} duplicates (original text) → feedback_duplicates.csv")

# Translated duplicates
df_duplicates_translated.to_csv(os.path.join(OUTPUT_DIR, "feedback_duplicates_translated.csv"), index=False)
print(f"📄 Saved {len(df_duplicates_translated)} duplicates (translated text) → feedback_duplicates_translated.csv")

# Short duplicates
df_short_duplicates_original.to_csv(os.path.join(OUTPUT_DIR, "feedback_short_duplicates.csv"), index=False)
print(f"📄 Saved {len(df_short_duplicates_original)} short duplicates (original text) → feedback_short_duplicates.csv")

# short translated duplicates
df_short_duplicates_translated.to_csv(os.path.join(OUTPUT_DIR, "feedback_short_duplicates_translated.csv"), index=False)
print(f"📄 Saved {len(df_short_duplicates_translated)} short duplicates (translated text) → feedback_short_duplicates_translated.csv")

# Short unique (original-based)
df_short_unique = df_short[~df_short["feedbackId"].isin(df_short_duplicates_original["feedbackId"])].reset_index(drop=True)
df_short_unique.to_csv(os.path.join(OUTPUT_DIR, "feedback_short_unique.csv"), index=False)
print(f"📄 Saved {len(df_short_unique)} short unique entries to {OUTPUT_DIR}/feedback_short_unique.csv")

# Short unique (translated-based)
df_short_unique_translated = df_short[~df_short["feedbackId"].isin(df_short_duplicates_translated["feedbackId"])].reset_index(drop=True)
df_short_unique_translated.to_csv(os.path.join(OUTPUT_DIR, "feedback_short_unique_translated.csv"), index=False)
print(f"📄 Saved {len(df_short_unique_translated)} short unique entries (translated-based) → feedback_short_unique_translated.csv")

# Unique (non-short, original-based)
duplicate_ids = set(df_duplicates_original["feedbackId"])
df_unique = df_main[~df_main["feedbackId"].isin(duplicate_ids)].reset_index(drop=True)
df_unique.to_csv(os.path.join(OUTPUT_DIR, "feedback_unique.csv"), index=False)
print(f"✅ Saved {len(df_unique)} unique entries to {OUTPUT_DIR}/feedback_unique.csv")

# Unique (non-short, translated-based)
duplicate_ids_translated = set(df_duplicates_translated["feedbackId"])
df_unique_translated = df_main[~df_main["feedbackId"].isin(duplicate_ids_translated)].reset_index(drop=True)
df_unique_translated.to_csv(os.path.join(OUTPUT_DIR, "feedback_unique_translated.csv"), index=False)
print(f"✅ Saved {len(df_unique_translated)} unique entries (translated-based) → feedback_unique_translated.csv")


# All similarity scores
df.to_csv(os.path.join(OUTPUT_DIR, "feedback_all_similarity_scores.csv"), index=False)
print(f"📦 Saved {len(df)} total entries to {OUTPUT_DIR}/feedback_all_similarity_scores.csv (with similarity column)")
