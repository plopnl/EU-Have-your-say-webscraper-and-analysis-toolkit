# ============================================
# 📄 EU Feedback Similarity Computation Script
# Created by Plop (Twitter/GitHub: @plopnl)
# Last updated: 2025-11-20
# Description: Computes pairwise text similarity between consultation feedback entries
#              from the EU "Have Your Say" portal. Uses TF-IDF vectorization with
#              sparse dot product acceleration to efficiently calculate cosine similarity.
#              Extracts top-N matches above a chosen threshold and saves results to CSV.
# Usage: python 2_feedback_similarity_matrix.py
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
# ============================================

import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from scipy.sparse import coo_matrix
from settings import *

# =========================
# 🔧 SETTINGS
# =========================
INPUT_FILE = FEEDBACK_WITH_LENGTHS
OUTPUT_FILE = FEEDBACK_SIMILARITY_TOPN
TEXT_COLUMN = "feedback_original_clean"
RAW_COLUMN = "feedback_original"

# =========================
# 🧼 TEXT CLEANING
# =========================
def remove_punctuation(text):
    return re.sub(r"[^\w\s]", "", text)

# =========================
# 📥 LOAD & PREP DATA
# =========================
df = pd.read_csv(INPUT_FILE)

if TEXT_COLUMN not in df.columns:
    df[TEXT_COLUMN] = (
        df[RAW_COLUMN].fillna("").str.strip().str.lower().apply(remove_punctuation)
    )

# =========================
# 📊 TF-IDF VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(stop_words="english")
tfidf = vectorizer.fit_transform(df[TEXT_COLUMN])

# =========================
# ⚡ TOP-N SIMILARITY COMPUTATION
# =========================
print(f"🚀 Computing top {TOP_N} similarities above {SIMILARITY_THRESHOLD} for {len(df)} entries...")

similarity_matrix = sp_matmul_topn(
    tfidf, tfidf.T,
    top_n=TOP_N,
    threshold=SIMILARITY_THRESHOLD,
    n_threads=-1
)

# =========================
# 📤 EXTRACT NON-ZERO SIMILARITIES
# =========================
coo = similarity_matrix.tocoo()
rows = []
for i, j, score in zip(coo.row, coo.col, coo.data):
    if i < j:  # avoid duplicates and self-matches
        rows.append({
            "feedbackId_i": df.iloc[i]["feedbackId"],
            "feedbackId_j": df.iloc[j]["feedbackId"],
            "similarity": round(score, 4)
        })

# =========================
# 💾 SAVE OUTPUT
# =========================
pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved top-N similarity results to {OUTPUT_FILE} with {len(rows)} pairs")
