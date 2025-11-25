#!/usr/bin/env python3
# ============================================
# 📄 EU Feedback Word Frequency Script
# Created by Plop (Twitter/GitHub: @plopnl)
# Last updated: 2025-11-20
# Description: Aggregates consultation feedback from the EU "Have Your Say" portal,
#              cleans original and translated text, and computes word frequencies.
#              Always outputs two global files (original + translated) in data/,
#              and optionally produces per-country and per-language word frequency tables.
# Usage: Set SEPARATE_BY_COUNTRY = True to export one file per country into
#        data/word_frequencies_by_country/. Set SEPARATE_BY_LANGUAGE = True to export one
#        file per language into data/word_frequencies_by_language/.
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
# ============================================

import os
import re
import pandas as pd
from collections import Counter

# =========================
# 🔧 SETTINGS
# =========================
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "feedback_details_with_lengths.csv")

SEPARATE_BY_COUNTRY = True
SEPARATE_BY_LANGUAGE = True

COUNTRY_DIR = os.path.join(DATA_DIR, "word_frequencies_by_country")
LANGUAGE_DIR = os.path.join(DATA_DIR, "word_frequencies_by_language")
os.makedirs(COUNTRY_DIR, exist_ok=True)
os.makedirs(LANGUAGE_DIR, exist_ok=True)

# =========================
# 🧼 TEXT CLEANING FUNCTION
# =========================
def clean_text(text):
    return re.sub(r"[^\w\s]", "", str(text).lower().strip())

# =========================
# 📥 LOAD & CLEAN DATA
# =========================
df = pd.read_csv(INPUT_FILE)
df["feedback_original_clean"] = df["feedback_original"].fillna("").apply(clean_text)
df["feedback_translated_clean"] = df["feedback_translated"].fillna("").apply(clean_text)

# =========================
# 📊 WORD FREQUENCY COUNTER
# =========================
def count_words(series):
    all_text = " ".join(series.tolist())
    words = all_text.split()
    return pd.DataFrame(Counter(words).items(), columns=["word", "count"]).sort_values(by="count", ascending=False)

# =========================
# 📤 ALWAYS EXPORT GLOBAL ORIGINAL + TRANSLATED
# =========================
df_orig = count_words(df["feedback_original_clean"])
df_orig.to_csv(os.path.join(DATA_DIR, "word_frequencies_original.csv"), index=False)
print(f"✅ Saved word frequencies for original feedback to {os.path.join(DATA_DIR, 'word_frequencies_original.csv')}")

df_trans = count_words(df["feedback_translated_clean"])
df_trans.to_csv(os.path.join(DATA_DIR, "word_frequencies_translated.csv"), index=False)
print(f"🌍 Saved word frequencies for translated feedback to {os.path.join(DATA_DIR, 'word_frequencies_translated.csv')}")

# =========================
# 📤 OPTIONAL EXPORT PER COUNTRY
# =========================
if SEPARATE_BY_COUNTRY and "country" in df.columns:
    for country, group in df.groupby("country"):
        combined_texts = group["feedback_original_clean"].tolist() + group["feedback_translated_clean"].tolist()
        df_country = count_words(pd.Series(combined_texts))

        safe_country = re.sub(r'[\\/*?:"<>|]', "_", str(country).strip()) if pd.notna(country) else "Unknown"
        filename = os.path.join(COUNTRY_DIR, f"word_frequencies_{safe_country}.csv")
        df_country.to_csv(filename, index=False)
        print(f"📊 Saved word frequencies for country {safe_country} to {filename}")

# =========================
# 📤 OPTIONAL EXPORT PER LANGUAGE
# =========================
if SEPARATE_BY_LANGUAGE and "language" in df.columns:
    for lang, group in df.groupby("language"):
        combined_texts = group["feedback_original_clean"].tolist() + group["feedback_translated_clean"].tolist()
        df_lang = count_words(pd.Series(combined_texts))

        safe_lang = re.sub(r'[\\/*?:"<>|]', "_", str(lang).strip()) if pd.notna(lang) else "Unknown"
        filename = os.path.join(LANGUAGE_DIR, f"word_frequencies_{safe_lang}.csv")
        df_lang.to_csv(filename, index=False)
        print(f"📊 Saved word frequencies for language {safe_lang} to {filename}")
