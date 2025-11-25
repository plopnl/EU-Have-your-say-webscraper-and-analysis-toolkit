# ============================================
# 📄 Feedback Fetcher Script
# Created by Plop (Twitter/GitHub: @plopnl)
# Initiative: Tobacco Taxation (ID 12645)
# Last updated: 2025-11-12
# Description: Fetches full feedback entries and attachment metadata
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/12645
# ============================================

import requests
import pandas as pd
import time
import os
import argparse

# =========================
# 🔧 SETTINGS (defaults)
# =========================
DEFAULT_INITIATIVE_ID = 12645
LANGUAGE = "en"
SLEEP_SECONDS = 0.2
OUTPUT_DIR = "data"
DETAILS_FILE = os.path.join(OUTPUT_DIR, "feedback_details.csv")
ATTACHMENTS_FILE = os.path.join(OUTPUT_DIR, "feedback_attachments.csv")
IDS_FILE = os.path.join(OUTPUT_DIR, "feedback_ids.csv")

# =========================
# 🚀 RUN SCRIPT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch EU feedback details")
    parser.add_argument(
        "--initiative-id",
        type=int,
        help=f"Initiative ID to fetch (default: {DEFAULT_INITIATIVE_ID})"
    )
    args = parser.parse_args()
    INITIATIVE_ID = args.initiative_id or DEFAULT_INITIATIVE_ID

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 📥 Load feedback IDs
    df_ids = pd.read_csv(IDS_FILE)
    existing_ids = set()
    if os.path.exists(DETAILS_FILE):
        df_existing = pd.read_csv(DETAILS_FILE)
        existing_ids = set(df_existing["feedbackId"].dropna().astype(int))
        print(f"📁 Found existing feedback_details.csv with {len(existing_ids)} entries")

    print(f"📡 Fetching detailed feedback for {len(df_ids)} entries...")

    feedback_data = []
    attachments_data = []
    failed_ids = []

    # 🔍 Fetch details
    for i, row in df_ids.iterrows():
        fid = int(row["feedbackId"])
        if fid in existing_ids:
            continue

        url = (
            f"https://ec.europa.eu/info/law/better-regulation/api/feedbackById"
            f"?feedbackId={fid}&initiativeId={INITIATIVE_ID}&language={LANGUAGE}"
        )

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"❌ Failed to fetch ID {fid}: {e}")
            failed_ids.append(fid)
            continue

        attachments = data.get("attachments", [])
        attachment_names = "; ".join([att.get("name") for att in attachments if att.get("name")])
        attachment_urls = "; ".join([att.get("url") for att in attachments if att.get("url")])

        feedback_data.append({
            "feedbackId": fid,
            "referenceInitiative": data.get("referenceInitiative"),
            "date": data.get("dateFeedback"),
            "country": data.get("country"),
            "language": data.get("language"),
            "userType": data.get("userType"),
            "submitter": data.get("submitterType"),
            "organization": data.get("organization"),
            "type": data.get("type"),
            "publication": data.get("publication"),
            "status": data.get("status"),
            "publicationStatus": data.get("publicationStatus"),
            "firstName": data.get("firstName"),
            "surname": data.get("surname"),
            "login": data.get("login"),
            "title": data.get("title"),
            "feedback_original": data.get("feedback"),
            "feedback_translated": data.get("feedbackTextUserLanguage"),
            "attachment_names": attachment_names,
            "attachment_urls": attachment_urls
        })

        if attachments:
            attachments_data.append({
                "feedbackId": fid,
                "attachment_names": attachment_names,
                "attachment_urls": attachment_urls
            })

        if i % 50 == 0:
            print(f"🔄 Processed {i} entries...")
        time.sleep(SLEEP_SECONDS)

    # 💾 Save outputs
    if feedback_data:
        df_new = pd.DataFrame(feedback_data)
        if os.path.exists(DETAILS_FILE):
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(DETAILS_FILE, index=False)
        print(f"✅ Saved updated feedback to {DETAILS_FILE} ({len(df_combined)} total entries)")
    else:
        print("⚠️ No new feedback entries were fetched")

    if attachments_data:
        pd.DataFrame(attachments_data).to_csv(ATTACHMENTS_FILE, index=False)
        print(f"📎 Saved attachment info to {ATTACHMENTS_FILE} ({len(attachments_data)} entries)")

    # 📋 Final status
    if failed_ids:
        print(f"❌ {len(failed_ids)} entries failed to fetch:")
        print(", ".join(map(str, failed_ids)))
    else:
        print("🎉 All entries fetched successfully!")
