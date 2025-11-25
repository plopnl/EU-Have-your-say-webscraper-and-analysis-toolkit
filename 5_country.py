#!/usr/bin/env python3
# ============================================
# 📄 EU Feedback Country Summary Script
# Created by Plop (Twitter/GitHub: @plopnl)
# Last updated: 2025-11-21
# Description: Aggregates consultation feedback submissions by country,
#              maps ISO alpha-3 codes to full names,
#              saves a single CSV with readable country names,
#              and also creates per-country CSVs.
# Usage: python 5_country.py
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
# ============================================

import os
import pandas as pd
import pycountry

DATA_DIR = "data"
PER_COUNTRY_DIR = os.path.join(DATA_DIR, "countries")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PER_COUNTRY_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, "feedback_details.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "feedback_totals_by_country_named.csv")

# =========================
# 📥 LOAD FEEDBACK DETAILS
# =========================
df = pd.read_csv(INPUT_FILE)

# =========================
# 🌍 AGGREGATE BY COUNTRY
# =========================
totals = df.groupby("country").size().reset_index(name="feedback_count")
totals["percentage"] = totals["feedback_count"] / totals["feedback_count"].sum() * 100

# =========================
# 🌍 MAP ISO CODES TO COUNTRY NAMES
# =========================
def get_country_name(code):
    try:
        return pycountry.countries.get(alpha_3=code).name
    except Exception:
        return "Unknown"

totals["country_name"] = totals["country"].apply(get_country_name)

# Reorder columns
totals = totals[["country", "country_name", "feedback_count", "percentage"]]

# =========================
# 💾 SAVE OUTPUT
# =========================
totals.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved country totals with names to {OUTPUT_FILE}")

# =========================
# 📂 SAVE PER-COUNTRY FILES
# =========================
for code in totals["country"].unique():
    country_df = df[df["country"] == code]
    country_name = get_country_name(code).replace(" ", "_")
    country_file = os.path.join(PER_COUNTRY_DIR, f"{code}_{country_name}.csv")
    country_df.to_csv(country_file, index=False)
    print(f"📄 Saved {len(country_df)} entries for {code} ({country_name}) to {country_file}")
