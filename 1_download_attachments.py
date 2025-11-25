# ============================================
# 📄 EU Feedback Attachment Downloader Script
# Created by Plop (Twitter/GitHub: @plopnl)
# Last updated: 2025-11-12
# Description: Downloads consultation feedback attachments from the EU "Have Your Say" portal
#              using metadata exported from feedback_attachments.csv. Handles filename
#              sanitization and saves files into a local attachments directory.
# Usage: python download_attachments.py
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
# ============================================

import pandas as pd
import requests
import os
import re
from settings import DATA_DIR

# =========================
# 🔧 SETTINGS
# =========================
ATTACHMENTS_CSV = os.path.join(DATA_DIR, "feedback_attachments.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_DIR), "attachments")

# =========================
# 🔧 FILENAME SANITIZATION
# =========================
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name.strip())

# =========================
# 🚀 RUN SCRIPT
# =========================
if __name__ == "__main__":
    if not os.path.exists(ATTACHMENTS_CSV):
        print(f"⏩ No attachments file found at {ATTACHMENTS_CSV} — skipping.")
        exit(0)
    df = pd.read_csv(ATTACHMENTS_CSV)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📎 Preparing to download {len(df)} feedback attachments...")

    for i, row in df.iterrows():
        urls_raw = str(row.get("attachment_urls", "")).strip()
        names_raw = str(row.get("attachment_names", "")).strip()

        if not urls_raw or not names_raw:
            continue

        urls = [u.strip() for u in urls_raw.split(";") if u.strip()]
        names = [n.strip() for n in names_raw.split(";") if n.strip()]

        for name, url in zip(names, urls):
            if url.startswith("https://ec.europa.eu/info/law/better-regulation/api/download/"):
                safe_name = sanitize_filename(name)
                filename = os.path.join(OUTPUT_DIR, f"{row['feedbackId']}_{safe_name}")
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print(f"✅ Downloaded: {filename}")
                except Exception as e:
                    print(f"❌ Failed to download {url}: {e}")
