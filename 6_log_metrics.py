#!/usr/bin/env python3
# ============================================
# 📈 EU Feedback Stance Metrics Logger
# Created by Plop (@plopnl)
# Last updated: 2025-11-21
# Description: Logs supervised stance proportions and consensus disagreement
#              across runs, appends to run_metrics.csv, and generates
#              a convergence plot over time.
# Usage: python 6_log_metrics.py
# ============================================

import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# ⚙️ SETTINGS
# =========================
DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "stance_detection", "supervised_model")
REVIEW_PATH = os.path.join(OUTPUT_DIR, "manual_review.csv")
LOG_PATH = os.path.join(OUTPUT_DIR, "run_metrics.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, "run_metrics_plot.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load latest manual_review file ---
    df = pd.read_csv(REVIEW_PATH)

    # --- Compute supervised stance proportions ---
    supervised_counts = df["stance_supervised"].value_counts(normalize=True) * 100
    consensus_counts = df["consensus"].value_counts(normalize=True) * 100

    # --- File modification timestamp ---
    mod_ts = os.path.getmtime(REVIEW_PATH)
    mod_dt = datetime.datetime.fromtimestamp(mod_ts).isoformat()

    # --- Prepare row ---
    row = {
        "unclear": supervised_counts.get("unclear", 0),
        "against": supervised_counts.get("against", 0),
        "for": supervised_counts.get("for", 0),
        "disagreement": consensus_counts.get("disagreement", 0),
        "file_modified": mod_dt,
        "run_timestamp": datetime.datetime.now().isoformat()
    }

    # --- Load log or init ---
    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH)
    else:
        log_df = None

    # --- Duplicate check ---
    already_logged = (
        log_df is not None and
        not log_df.empty and
        str(log_df.iloc[-1]["file_modified"]) == row["file_modified"]
    )

    if already_logged:
        print("⚠️ No changes since last run (file_modified unchanged). Skipping append.")
    else:
        run_number = int(log_df["run"].max()) + 1 if log_df is not None and not log_df.empty else 1
        row["run"] = run_number

        new_row = pd.DataFrame([row])
        if log_df is None or log_df.empty:
            log_df = new_row
        else:
            log_df = pd.concat([log_df, new_row], ignore_index=True)

        log_df.to_csv(LOG_PATH, index=False)
        print(f"✅ Appended run {row['run']} metrics to {LOG_PATH}")

        # --- Delta vs previous run ---
        if len(log_df) > 1:
            prev = log_df.iloc[-2]
            print("Δ vs previous run:")
            for col in ["unclear", "against", "for", "disagreement"]:
                diff = row[col] - prev[col]
                print(f"  {col}: {diff:+.4f}%")

    # --- Plot evolution ---
    if log_df is not None and not log_df.empty:
        plt.figure(figsize=(8,5))
        plt.plot(log_df["run"], log_df["unclear"], label="Unclear", marker="o")
        plt.plot(log_df["run"], log_df["against"], label="Against", marker="o")
        plt.plot(log_df["run"], log_df["for"], label="For", marker="o")
        plt.plot(log_df["run"], log_df["disagreement"], label="Disagreement", marker="o")

        plt.xlabel("Run")
        plt.ylabel("Percentage")
        plt.title("Supervised stance evolution across runs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(PLOT_PATH)
        print(f"📊 Saved convergence plot: {PLOT_PATH}")
    else:
        print("ℹ️ No logged runs yet; plot not generated.")

if __name__ == "__main__":
    main()
