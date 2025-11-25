#!/usr/bin/env python3
# ============================================
# 🧮 EU Attachment Similarity Matrix Builder
# Created by Plop (Twitter/GitHub: @plopnl)
# Description: Processes grouped attachment similarity data from the EU "Have Your Say" portal.
#              Combines attachment clusters with submission metadata to compute pairwise similarity scores.
#              Outputs matrix-style CSVs for downstream plotting and analysis:
#              - Top-N similarity pairs (feedbackId and filename level)
#              - Combined similarity matrix with group and timing metadata
#              Enables detection of coordinated submissions and timing anomalies.
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
# ============================================

import os
import pandas as pd

DUP_FILE = "data/attachment_duplicates.csv"
FEEDBACK_FILE = "data/feedback_details_with_lengths.csv"
OUT_TOPN = "data/attachment_similarity_topn.csv"
OUT_COMBINED = "data/attachment_combined_topn.csv"
OUT_AUG = "data/similarity_with_attachments.csv"

def main():
    if not os.path.exists(DUP_FILE):
        print(f"❌ {DUP_FILE} not found")
        return

    dup = pd.read_csv(DUP_FILE, dtype=str)

    rows = []
    for _, r in dup.iterrows():
        group_id = r.get("group_id", "")
        group_size = r.get("group_size", "")
        max_sim = r.get("max_similarity_in_group", "")

        feedbackId_i = str(r.get("feedbackId", "")).strip()
        filename_i = str(r.get("filename", "")).strip()

        matched_ids = str(r.get("matched_ids", "")).split(";")
        matched_files = str(r.get("matched_files", "")).split(";")
        scores = str(r.get("similarity_scores", "")).split(";")

        matched_ids = [m.strip() for m in matched_ids if m.strip()]
        matched_files = [m.strip() for m in matched_files if m.strip()]
        scores = [s.strip() for s in scores if s.strip()]

        for fid_j, fn_j_raw, score in zip(matched_ids, matched_files, scores):
            try:
                score_val = float(score)
            except Exception:
                continue
            rows.append({
                "feedbackId_i": feedbackId_i,
                "feedbackId_j": fid_j,
                "filename_i": filename_i,
                "filename_j": fn_j_raw,
                "similarity": score_val,
                "group_id": group_id,
                "group_size": group_size,
                "max_similarity_in_group": max_sim
            })

    if not rows:
        print("⚠️ No attachment similarity pairs parsed.")
        return

    combined = pd.DataFrame(rows)
    combined = combined.sort_values("similarity", ascending=False).reset_index(drop=True)

    # Save combined
    combined.to_csv(OUT_COMBINED, index=False)
    print(f"✅ Saved {OUT_COMBINED} ({len(combined)} rows)")

    # Minimal top-N for plotting
    topn = combined[["feedbackId_i", "feedbackId_j", "similarity"]].copy()
    topn.to_csv(OUT_TOPN, index=False)
    print(f"✅ Saved {OUT_TOPN} ({len(topn)} rows)")

    # =========================
    # Merge in submission dates
    # =========================
    if os.path.exists(FEEDBACK_FILE):
        fb = pd.read_csv(FEEDBACK_FILE, dtype=str)
        if "feedbackId" in fb.columns and "date" in fb.columns:
            fb["date"] = pd.to_datetime(fb["date"], errors="coerce")
            date_map = dict(zip(fb["feedbackId"], fb["date"]))

            combined["date_i"] = combined["feedbackId_i"].map(date_map)
            combined["date_j"] = combined["feedbackId_j"].map(date_map)

            def normalize_by_id(r):
                if r["feedbackId_i"] > r["feedbackId_j"]:
                    for field in ["feedbackId", "filename", "date"]:
                        r[f"{field}_i"], r[f"{field}_j"] = r[f"{field}_j"], r[f"{field}_i"]
                return r

            combined = combined.apply(normalize_by_id, axis=1)

            combined["time_diff_sec"] = (combined["date_j"] - combined["date_i"]).dt.total_seconds()
            combined["time_diff_days"] = combined["time_diff_sec"] / 86400.0
            combined = combined[combined["time_diff_days"] >= 0]
        else:
            print("⚠️ feedback_details_with_lengths.csv missing 'feedbackId' or 'date' columns")
    else:
        print("⚠️ feedback_details_with_lengths.csv not found — time differences not computed")

    # Augmented output
    aug = combined.copy()
    aug["either_has_attachment"] = True
    aug["attachment_filenames"] = aug["filename_i"] + "; " + aug["filename_j"]

    cols = [
        "feedbackId_i", "feedbackId_j",
        "filename_i", "filename_j",
        "similarity",
        "group_id", "group_size", "max_similarity_in_group",
        "date_i", "date_j", "time_diff_sec", "time_diff_days",
        "either_has_attachment", "attachment_filenames"
    ]
    aug = aug[[c for c in cols if c in aug.columns]]

    aug.to_csv(OUT_AUG, index=False)
    print(f"✅ Saved {OUT_AUG} ({len(aug)} rows)")

if __name__ == "__main__":
    main()
