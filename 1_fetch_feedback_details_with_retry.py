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
from settings import *

# =========================
# 🔧 SETTINGS (defaults)
# =========================
DEFAULT_INITIATIVE_ID = INITIATIVE_ID
SLEEP_SECONDS = SLEEP_SECONDS_DETAILS
DETAILS_FILE = FEEDBACK_DETAILS_FILE
ATTACHMENTS_FILE = os.path.join(DATA_DIR, "feedback_attachments.csv")
IDS_FILE = FEEDBACK_IDS_FILE

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
    parser.add_argument(
        "--refetch-attachments",
        action="store_true",
        help="Re-fetch only entries that already exist but have no attachment URL/name data"
    )
    parser.add_argument(
        "--refetch-untranslated",
        action="store_true",
        help="Re-fetch entries tagged as non-English but missing a translation (EU lag backfill)"
    )
    parser.add_argument(
        "--refetch-all",
        action="store_true",
        help="Re-fetch every entry — full refresh, overwrites existing data"
    )
    args = parser.parse_args()
    INITIATIVE_ID = args.initiative_id or DEFAULT_INITIATIVE_ID

    os.makedirs(DATA_DIR, exist_ok=True)

    # 📥 Load feedback IDs
    df_ids = pd.read_csv(IDS_FILE)
    existing_ids = set()
    refetch_ids = set()
    if os.path.exists(DETAILS_FILE):
        df_existing = pd.read_csv(DETAILS_FILE)
        existing_ids = set(df_existing["feedbackId"].dropna().astype(int))
        print(f"📁 Found existing feedback_details.csv with {len(existing_ids)} entries")
        if args.refetch_all:
            existing_ids = set()
            print(f"🔄 Re-fetching all {len(df_existing)} entries (full refresh)...")
        elif args.refetch_attachments:
            # Find entries that have no attachment URL despite being in the attachments CSV
            if os.path.exists(ATTACHMENTS_FILE):
                df_att = pd.read_csv(ATTACHMENTS_FILE)
                refetch_ids = set(
                    df_att.loc[df_att["attachment_urls"].isna(), "feedbackId"].dropna().astype(int)
                )
                print(f"🔄 Re-fetching {len(refetch_ids)} entries with missing attachment data...")
        elif args.refetch_untranslated:
            no_trans = (
                df_existing["feedback_translated"].isna() |
                df_existing["feedback_translated"].astype(str).str.strip().isin(["", "nan"])
            )
            not_en = df_existing["language"].astype(str).str.upper() != "EN"
            refetch_ids = set(df_existing.loc[no_trans & not_en, "feedbackId"].dropna().astype(int))
            print(f"🔄 Re-fetching {len(refetch_ids)} non-EN entries with missing translations...")

    if args.refetch_attachments:
        if not refetch_ids:
            print("✅ No entries with missing attachment data — nothing to re-fetch.")
            exit(0)
        df_ids = df_existing[df_existing["feedbackId"].isin(refetch_ids)]
        existing_ids -= refetch_ids
    elif args.refetch_untranslated:
        if not refetch_ids:
            print("✅ No non-EN entries with missing translations — nothing to re-fetch.")
            exit(0)
        df_ids = df_existing[df_existing["feedbackId"].isin(refetch_ids)]
        existing_ids -= refetch_ids

    _to_fetch = [int(r["feedbackId"]) for _, r in df_ids.iterrows()
                 if int(r["feedbackId"]) not in existing_ids]
    print(f"📡 {len(_to_fetch)} to fetch  |  {len(existing_ids)} already exist  |  {len(df_ids)} total in IDs file")
    if not _to_fetch:
        print("✅ Nothing to fetch.")
        exit(0)

    feedback_data = []
    attachments_data = []
    failed_ids = []
    CHECKPOINT_EVERY = 500
    REQUEST_TIMEOUT  = 30  # seconds

    def _flush_checkpoint(batch, details_file, existing_file_existed):
        """Append a batch of rows to the details CSV, creating header only once."""
        if not batch:
            return
        df_batch = pd.DataFrame(batch)
        write_header = not os.path.exists(details_file) or os.path.getsize(details_file) == 0
        df_batch.to_csv(details_file, mode="a", index=False, header=write_header)
        print(f"💾 Checkpoint: wrote {len(batch)} rows to {details_file}")

    # On a fresh (non-refetch) run we need to preserve the existing rows first,
    # then append new ones.  Copy existing to a temp file so we can start fresh.
    _checkpoint_file = DETAILS_FILE
    if not any([args.refetch_all, args.refetch_attachments, args.refetch_untranslated]):
        if os.path.exists(DETAILS_FILE) and existing_ids:
            # Write existing rows once so appending new rows stays consistent
            df_existing.to_csv(DETAILS_FILE, index=False)

    # 🔍 Fetch details
    _fetched = 0
    for i, row in df_ids.iterrows():
        fid = int(row["feedbackId"])
        if fid in existing_ids:
            continue

        url = (
            f"https://ec.europa.eu/info/law/better-regulation/api/feedbackById"
            f"?feedbackId={fid}&initiativeId={INITIATIVE_ID}&language={LANGUAGE}"
        )

        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"❌ Failed to fetch ID {fid}: {e}")
            failed_ids.append(fid)
            continue

        attachments = data.get("attachments", [])
        _DOWNLOAD_BASE = "https://ec.europa.eu/info/law/better-regulation/api/download/"
        attachment_names = "; ".join([
            att.get("fileName") or att.get("name", "")
            for att in attachments if att.get("fileName") or att.get("name")
        ])
        attachment_urls = "; ".join([
            att["url"] if att.get("url") else _DOWNLOAD_BASE + att["documentId"]
            for att in attachments if att.get("url") or att.get("documentId")
        ])

        existing_ids.add(fid)
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

        _fetched += 1
        if _fetched % 50 == 0:
            print(f"🔄 Fetched {_fetched} / {len(_to_fetch)}...")

        if _fetched % CHECKPOINT_EVERY == 0 and not any([args.refetch_all, args.refetch_attachments, args.refetch_untranslated]):
            _flush_checkpoint(feedback_data[-CHECKPOINT_EVERY:], DETAILS_FILE, bool(existing_ids))

        time.sleep(SLEEP_SECONDS)

    # 💾 Save outputs
    if feedback_data:
        df_new = pd.DataFrame(feedback_data)
        if args.refetch_all and os.path.exists(DETAILS_FILE):
            # Replace rows we re-fetched; keep any that weren't re-fetched (e.g. new IDs added mid-run)
            df_combined = df_existing[~df_existing["feedbackId"].isin(df_new["feedbackId"])].copy()
            df_combined = pd.concat([df_combined, df_new], ignore_index=True)
        elif args.refetch_attachments and os.path.exists(DETAILS_FILE):
            # Patch attachment columns in-place for the re-fetched rows
            df_combined = df_existing.copy()
            df_combined["attachment_names"] = df_combined["attachment_names"].astype(object)
            df_combined["attachment_urls"] = df_combined["attachment_urls"].astype(object)
            for _, patch_row in df_new.iterrows():
                mask = df_combined["feedbackId"] == patch_row["feedbackId"]
                df_combined.loc[mask, "attachment_names"] = patch_row["attachment_names"]
                df_combined.loc[mask, "attachment_urls"] = patch_row["attachment_urls"]
        elif args.refetch_untranslated and os.path.exists(DETAILS_FILE):
            # Patch translation columns in-place for the re-fetched rows
            df_combined = df_existing.copy()
            df_combined["feedback_translated"] = df_combined["feedback_translated"].astype(object)
            df_combined["language"] = df_combined["language"].astype(object)
            got_trans = []
            for _, patch_row in df_new.iterrows():
                mask = df_combined["feedbackId"] == patch_row["feedbackId"]
                df_combined.loc[mask, "feedback_translated"] = patch_row["feedback_translated"]
                df_combined.loc[mask, "language"] = patch_row["language"]
                if pd.notna(patch_row["feedback_translated"]) and str(patch_row["feedback_translated"]).strip():
                    got_trans.append(int(patch_row["feedbackId"]))
            if got_trans:
                print(f"✅ {len(got_trans)} entries now have EU translations: {got_trans}")
            else:
                print("⚠️  EU still has no translations for any of the re-fetched entries")
        elif os.path.exists(DETAILS_FILE):
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(DETAILS_FILE, index=False)
        print(f"✅ Saved updated feedback to {DETAILS_FILE} ({len(df_combined)} total entries)")
    else:
        print("⚠️ No new feedback entries were fetched")

    if attachments_data:
        df_att_new = pd.DataFrame(attachments_data)
        if args.refetch_attachments and os.path.exists(ATTACHMENTS_FILE):
            df_att_existing = pd.read_csv(ATTACHMENTS_FILE).astype({"attachment_names": object, "attachment_urls": object})
            for _, patch_row in df_att_new.iterrows():
                mask = df_att_existing["feedbackId"] == patch_row["feedbackId"]
                df_att_existing.loc[mask, "attachment_names"] = patch_row["attachment_names"]
                df_att_existing.loc[mask, "attachment_urls"] = patch_row["attachment_urls"]
            df_att_existing.to_csv(ATTACHMENTS_FILE, index=False)
        else:
            df_att_new.to_csv(ATTACHMENTS_FILE, index=False)
        print(f"📎 Saved attachment info to {ATTACHMENTS_FILE} ({len(df_att_new)} entries)")

    # 📋 Final status
    if failed_ids:
        print(f"❌ {len(failed_ids)} entries failed to fetch:")
        print(", ".join(map(str, failed_ids)))
    else:
        print("🎉 All entries fetched successfully!")
