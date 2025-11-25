#!/usr/bin/env python3
# ============================================
# 📄 EU Feedback Keyword Analysis Script
# Created by Plop (Twitter/GitHub: @plopnl)
# Last updated: 2025-11-20
# Description: Analyzes consultation feedback from the EU "Have Your Say" portal by
#              cleaning text and scanning for predefined keywords and regex patterns.
#              Counts occurrences across both original and translated feedback fields,
#              producing a ranked list of keyword frequencies for further analysis.
# Usage: Extend the KEYWORDS list with plain strings (e.g. "tax", "public health")
#        or regular expressions (e.g. r"\bharm\s*reduction\b") to capture variations.
#        Regex patterns allow flexible matching of word boundaries, spacing, or synonyms.
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
# ============================================

import os
import re
import pandas as pd

# =========================
# 📥 LOAD FEEDBACK DATA
# =========================
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "feedback_details_with_lengths.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "keyword_counts.csv")

df = pd.read_csv(INPUT_FILE)

# =========================
# 🧼 TEXT CLEANING FUNCTION
# =========================
def clean_text(text):
    return re.sub(r"[^\w\s]", "", str(text).lower().strip())

df["feedback_original_clean"] = df["feedback_original"].fillna("").apply(clean_text)
df["feedback_translated_clean"] = df["feedback_translated"].fillna("").apply(clean_text)

# =========================
# 🔍 DEFINE KEYWORDS
# =========================
KEYWORDS = [
    # Harm reduction and risk language
    "harm-reduction", "harm reduction", "harmreduction", "reduce harm", "reducing harm",
    "harm minimization", "risk reduction", "risk mitigation", "reducing risks", "reduce risks",
    "safer use", "safe use", "less harmful", "less risky", "reduced harm", "harm reducing",
    "risk reducing", "reduced risk", "lower harm", "lower risk",
    r"\bharm\s*reduction\b", r"\breduc(?:e|ing)\s+harm\b",
    r"\brisk\s+(?:reduction|minimization|mitigation)\b",

    # Safer alternatives
    "safer nicotine products", "safer products", "safer alternatives",

    # Public health and policy
    "public health", "illicit trade", "tax",

    # Market-related terms (with variations)
    "illegal market", "illicit market", "informal market", "black market", "gray market",
    "shadow market", "underground market", "bootleg market", "contraband",
    "street market", "corner market", "grey economy", "regulatory loophole", 
    "adulterated goods", "underground economy",

    # Regex to capture variations of market/marketplace/trade
    r"\billegal\s+(?:market|marketplace|trade)\b",
    r"\billicit\s+(?:market|marketplace|trade)\b",
    r"\binformal\s+(?:market|marketplace|trade)\b",
    r"\bblack\s+(?:market|marketplace|trade)\b",
    r"\bgray\s+(?:market|marketplace|trade)\b",
    r"\bshadow\s+(?:market|marketplace|trade)\b",
    r"\bunderground\s+(?:market|marketplace|trade)\b",
    r"\bbootleg\s+(?:market|marketplace|trade)\b",
    r"\bstreet\s+(?:market|marketplace|trade)\b",
    r"\bcorner\s+(?:market|marketplace|trade)\b"
]

# =========================
# 🛠️ CONVERT TO NON-CAPTURING GROUPS
# =========================
def make_non_capturing(pattern):
    return re.sub(r"\(([^?][^)]*)\)", r"(?:\1)", pattern)

KEYWORDS = [make_non_capturing(k) for k in KEYWORDS]

# =========================
# 📊 COUNT KEYWORD OCCURRENCES
# =========================
results = []
for keyword in KEYWORDS:
    count = (
        df["feedback_original_clean"].str.contains(keyword, case=False, na=False).sum() +
        df["feedback_translated_clean"].str.contains(keyword, case=False, na=False).sum()
    )
    results.append({"keyword": keyword, "count": count})

df_keywords = pd.DataFrame(results).sort_values(by="count", ascending=False)

# =========================
# 💾 SAVE RESULTS
# =========================
df_keywords.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved keyword counts to {OUTPUT_FILE}")
