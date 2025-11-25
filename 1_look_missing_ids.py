# ============================================
# 📄 Missing Feedback ID Checker Script
# Created by Plop (Twitter/GitHub: @plopnl)
# Last updated: 2025-11-12
# Description: Compares full feedback ID list against WIP details to find missing entries.
# Usage: python 1_look_missing_ids.py
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
# ============================================

import pandas as pd
import os
from settings import FEEDBACK_IDS_FILE, FEEDBACK_DETAILS_FILE, DATA_DIR

# =========================
# 🔧 SETTINGS
# =========================
IDS_FILE = FEEDBACK_IDS_FILE
WIP_FILE = FEEDBACK_DETAILS_FILE  # compare against the current details file
OUTPUT_FILE = os.path.join(DATA_DIR, "missing_feedback_ids.csv")

# =========================
# 🚀 RUN SCRIPT
# =========================
if __name__ == "__main__":
    # Load full list of expected IDs
    df_all = pd.read_csv(IDS_FILE)
    all_ids = set(df_all["feedbackId"].dropna().astype(int))

    # Load WIP sheet (shorter one)
    df_wip = pd.read_csv(WIP_FILE)
    wip_ids = set(df_wip["feedbackId"].dropna().astype(int))

    # Find missing ones: in full list but not in WIP
    missing_ids = sorted(all_ids - wip_ids)

    # Save or print
    print(f"❌ Missing {len(missing_ids)} feedback IDs from WIP")
    print(missing_ids[:20])  # Show first 20 for quick inspection

    pd.DataFrame({"missing_feedbackId": missing_ids}).to_csv(OUTPUT_FILE, index=False)
    print(f"📁 Saved missing IDs to {OUTPUT_FILE}")
