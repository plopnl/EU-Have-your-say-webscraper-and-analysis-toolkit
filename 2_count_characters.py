# ============================================
# 📄 EU Feedback Length Augmentation Script
# Created by Plop (Twitter/GitHub: @plopnl)
# Last updated: 2025-11-20
# Description: Enhances consultation feedback data from the EU "Have Your Say" portal
#              by adding character and word counts for both original and translated feedback text.
#              Produces an updated CSV with these additional length metrics.
# Usage: python 2_prepare_feedback_lengths.py
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
# ============================================

import pandas as pd
import os
import re

# =========================
# 🔧 SETTINGS
# =========================
INPUT_FILE = os.path.join("data", "feedback_details.csv")
OUTPUT_FILE = os.path.join("data", "feedback_details_with_lengths.csv")

# =========================
# 🧼 TEXT NORMALIZATION
# =========================
def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

# =========================
# 📥 LOAD FEEDBACK DATA
# =========================
df = pd.read_csv(INPUT_FILE)

# Normalize whitespace
df["feedback_original"] = df["feedback_original"].fillna("").apply(normalize_whitespace)
df["feedback_translated"] = df["feedback_translated"].fillna("").apply(normalize_whitespace)

# =========================
# 🔢 ADD CHARACTER COUNTS
# =========================
df["char_count_original"] = df["feedback_original"].str.len()
df["char_count_translated"] = df["feedback_translated"].str.len()
df["char_count_combined"] = (df["feedback_original"] + " " + df["feedback_translated"]).str.len()

# =========================
# 🔢 ADD WORD COUNTS
# =========================
df["word_count_original"] = df["feedback_original"].str.split().apply(len)
df["word_count_translated"] = df["feedback_translated"].str.split().apply(len)
df["word_count_combined"] = (df["feedback_original"] + " " + df["feedback_translated"]).str.split().apply(len)

# =========================
# 💾 SAVE UPDATED DATA
# =========================
os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(df)} rows with character & word counts to {OUTPUT_FILE}")
