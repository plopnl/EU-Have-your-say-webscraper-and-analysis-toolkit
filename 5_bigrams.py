#!/usr/bin/env python3
# ============================================
# 📄 EU Feedback Bigram Analysis Script
# Created by Plop (Twitter/GitHub: @plopnl)
# Last updated: 2025-11-20
# Description: Processes consultation feedback text from the EU "Have Your Say" portal.
#              Cleans text, removes stopwords, and extracts bigram frequencies
#              from both original and translated feedback fields.
#              Handy to find keyword combinations to be used with keyword_counter.py
# Usage: python 5_bigrams.py
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
# ============================================

import os
import re
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# =========================
# SETTINGS
# =========================
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "feedback_details_with_lengths.csv")
OUTPUT_FILE_ORIGINAL = os.path.join(DATA_DIR, "bigram_counts_original.csv")
OUTPUT_FILE_TRANSLATED = os.path.join(DATA_DIR, "bigram_counts_translated.csv")

# =========================
# CLEANING FUNCTION
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s']", "", text)  # remove punctuation
    return text.strip()

def extract_bigrams(series):
    bigram_counter = Counter()
    for text in series.fillna("").apply(clean_text):
        words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS and len(w) > 2]
        for i in range(len(words) - 1):
            bigram = (words[i], words[i+1])
            bigram_counter[bigram] += 1
    return pd.DataFrame(
        [(w1, w2, count) for (w1, w2), count in bigram_counter.items()],
        columns=["word1", "word2", "count"]
    ).sort_values("count", ascending=False).reset_index(drop=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

# =========================
# EXTRACT BIGRAMS
# =========================
bigram_original = extract_bigrams(df["feedback_original"])
bigram_translated = extract_bigrams(df["feedback_translated"])

# =========================
# SAVE OUTPUTS
# =========================
bigram_original.to_csv(OUTPUT_FILE_ORIGINAL, index=False)
print(f"✅ Saved {len(bigram_original)} bigrams to {OUTPUT_FILE_ORIGINAL}")
print("Top 10 original bigrams:")
print(bigram_original.head(10))

bigram_translated.to_csv(OUTPUT_FILE_TRANSLATED, index=False)
print(f"🌍 Saved {len(bigram_translated)} bigrams to {OUTPUT_FILE_TRANSLATED}")
print("Top 10 translated bigrams:")
print(bigram_translated.head(10))
