# ============================================
# 📄 EU Feedback ID Fetcher Script
# Created by Plop (Twitter/GitHub: @plopnl)
# Last updated: 2025-11-12
# Description: Fetches consultation feedback IDs from the EU "Have Your Say" portal
# Usage: python get_feedback_ids.py --publication-id 20315
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
# ============================================

import requests
import pandas as pd
import time
import argparse
import os
from settings import *

# =========================
# 🔧 SETTINGS (defaults)
# =========================
OUTPUT_FILE = FEEDBACK_IDS_FILE

# =========================
# 📥 FETCH FEEDBACK IDS
# =========================
def fetch_feedback_ids(publication_id, language, page_size, sleep_seconds):
    all_ids = []
    page = 0

    print(f"📡 Starting feedback ID fetch for publication {publication_id}...")

    while True:
        url = (
            f"https://ec.europa.eu/info/law/better-regulation/api/allFeedback"
            f"?publicationId={publication_id}&language={language}"
            f"&page={page}&size={page_size}&sort=dateFeedback,DESC"
        )

        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"❌ Error on page {page}: {e}")
            break

        data = response.json()
        entries = data.get("content", [])
        if not entries:
            print("✅ No more entries found.")
            break

        for entry in entries:
            all_ids.append({
                "feedbackId": entry.get("id"),
                "title": entry.get("title"),
                "date": entry.get("dateFeedback"),
                "country": entry.get("country")
            })

        print(f"📄 Page {page} done, {len(entries)} entries.")
        page += 1
        time.sleep(sleep_seconds)

    return all_ids

# =========================
# 💾 SAVE TO CSV
# =========================
def save_feedback_ids(feedback_list, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame(feedback_list)
    before = len(df)
    df = df.drop_duplicates("feedbackId", keep="last")
    if len(df) < before:
        print(f"⚠️  Dropped {before - len(df)} duplicate feedbackIds (API pagination overlap)")
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df)} feedback IDs to {output_file}")

# =========================
# 🚀 RUN SCRIPT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch EU feedback IDs")
    parser.add_argument(
        "--publication-id",
        type=int,
        help=f"Publication ID to fetch (default: {DEFAULT_PUBLICATION_ID})"
    )
    args = parser.parse_args()

    publication_id = args.publication_id or DEFAULT_PUBLICATION_ID
    feedback_data = fetch_feedback_ids(publication_id, LANGUAGE, PAGE_SIZE, SLEEP_SECONDS)
    save_feedback_ids(feedback_data, OUTPUT_FILE)
