#!/usr/bin/env python3
# ============================================
# 🔄 EU Feedback Supervised Reset Utility
# Created by Plop (@plopnl)
# Last updated: 2025-11-21
# Description: Deletes supervised stance detection artifacts (manual_review, run_metrics, plot)
#              so the pipeline can be rerun from scratch.
# Usage: python 6_reset_supervised.py
# ============================================

import os

# =========================
# ⚙️ SETTINGS
# =========================
DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "stance_detection", "supervised_model")
FILES_TO_DELETE = [
    os.path.join(OUTPUT_DIR, "manual_review.csv"),      # optional: wipe labels
    os.path.join(OUTPUT_DIR, "run_metrics.csv"),        # metrics log
    os.path.join(OUTPUT_DIR, "run_metrics_plot.png"),   # convergence plot
]

def main():
    print("🧭 Running Step 6: Stance Detection & Evaluation (Supervised Reset)")
    print("🔄 Resetting supervised model artifacts...")
    for f in FILES_TO_DELETE:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑️ Deleted {f}")
        else:
            print(f"ℹ️ File not found, skipped: {f}")
    print("✅ Reset complete. You can now rerun your pipeline from scratch.")

if __name__ == "__main__":
    main()
