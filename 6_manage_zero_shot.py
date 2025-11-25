#!/usr/bin/env python3
# ============================================
# 🛠️ EU Feedback Zero-Shot Manager
# Created by Plop (@plopnl)
# Last updated: 2025-11-21
# Description: Manage zero-shot stance detection models by disabling,
#              archiving outputs, backing up manual_review.csv, and restoring backups.
# Usage: python 6_manage_zero_shots.py [--disable MODEL] [--restore N|MODEL] [--latest]
# ============================================

"""
manage_zero_shot.py
===================

Utility to manage zero-shot models in the supervised stance detection pipeline.

USAGE:
------

Disable a model (move its files to disabled_models/, backup manual_review.csv, drop its column):
    python manage_zero_shot.py --disable MODEL_PREFIX
    python manage_zero_shot.py --disable   # interactive list selection with performance summaries

Restore the latest backup of manual_review.csv:
    python manage_zero_shot.py --latest

Restore a specific backup by number (listed interactively):
    python manage_zero_shot.py --restore 2

Restore the most recent backup associated with a given model prefix:
    python manage_zero_shot.py --restore MODEL_PREFIX
    python manage_zero_shot.py --restore   # interactive list selection

Notes:
------
- Backups are stored in stance_detection/supervised_model/backup/
  with filenames like manual_review_backup_YYYYMMDD_HHMMSS_MODEL.csv
- Disabled model files are moved into stance_detection/supervised_model/disabled_models/
- MODEL_PREFIX must match the column name in manual_review.csv (e.g. stance_zero_shot_mDeBERTa_v3_base_xnli_multilingual_nli_2mil7)
"""

import os
import shutil
import pandas as pd
import datetime

# =========================
# ⚙️ SETTINGS
# =========================
DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "stance_detection", "supervised_model")
DISABLED_DIR = os.path.join(OUTPUT_DIR, "disabled_models")
BACKUP_DIR = os.path.join(OUTPUT_DIR, "backup")
REVIEW_PATH = os.path.join(OUTPUT_DIR, "manual_review.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DISABLED_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# =========================
# 📜 LOGGING
# =========================
def log_action(action: str, model_prefix: str, stats: dict):
    """Append disable/restore actions to a log file with timestamp and stats."""
    log_path = os.path.join(OUTPUT_DIR, "disabled_log.txt")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"{ts} | {action} | {model_prefix} | "
                f"Unclear={stats['unclear']}% Against={stats['against']}% For={stats['for']}%\n")
    print(f"📜 Action logged → {log_path}")

# =========================
# 🔒 DISABLE MODEL
# =========================
def disable_model(model_prefix: str):
    os.makedirs(DISABLED_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    print(f"🔄 Disabling {model_prefix}…")

    # Move files containing prefix
    for folder in [OUTPUT_DIR, "figures", "stance_detection"]:
        if os.path.exists(folder):
            for fname in os.listdir(folder):
                if fname == f"{model_prefix}.csv" or fname == f"{model_prefix}.png":
                    src = os.path.join(folder, fname)
                    dest = os.path.join(DISABLED_DIR, fname)
                    shutil.move(src, dest)
                    print(f"📦 Moved {src} → {dest}")

    # Backup manual_review.csv
    if os.path.exists(REVIEW_PATH):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"manual_review_backup_{timestamp}_{model_prefix}.csv"
        backup_path = os.path.join(BACKUP_DIR, backup_name)
        shutil.copy2(REVIEW_PATH, backup_path)
        print(f"💾 Backed up manual_review.csv → {backup_path}")

        # Remove column
        df = pd.read_csv(REVIEW_PATH)
        if model_prefix in df.columns:
            df.drop(columns=[model_prefix], inplace=True)
            df.to_csv(REVIEW_PATH, index=False)
            print(f"🧹 Removed column {model_prefix} from manual_review.csv")
        else:
            print(f"ℹ️ Column {model_prefix} not present in manual_review.csv")
    else:
        print("ℹ️ manual_review.csv not found, skipped backup and cleanup.")

    print("✅ Model disabled and archived.")

# =========================
# 📂 BACKUP MANAGEMENT
# =========================
def list_backups():
    if not os.path.exists(BACKUP_DIR):
        return []
    return sorted(
        [f for f in os.listdir(BACKUP_DIR) if f.startswith("manual_review_backup_")],
        reverse=True
    )

def restore_backup(choice: int = None, latest: bool = False, model_prefix: str = None):
    backups = list_backups()
    if not backups:
        print("⚠️ No backups available to restore.")
        return

    if latest:
        selected = backups[0]
    elif model_prefix:
        candidates = [f for f in backups if model_prefix in f]
        if not candidates:
            print(f"⚠️ No backups found for model prefix {model_prefix}")
            return
        selected = candidates[0]
    else:
        print("📂 Available backups:")
        for i, fname in enumerate(backups, 1):
            print(f"{i}. {fname}")
        try:
            choice = int(input("Select backup number to restore: "))
            if choice < 1 or choice > len(backups):
                print("❌ Invalid choice.")
                return
        except ValueError:
            print("❌ Please enter a valid number.")
            return
        selected = backups[choice - 1]

    src = os.path.join(BACKUP_DIR, selected)
    shutil.copy2(src, REVIEW_PATH)
    print(f"✅ Restored {selected} → {REVIEW_PATH}")

    df = pd.read_csv(REVIEW_PATH)
    print(f"🔎 manual_review.csv now has {len(df)} rows and {len(df.columns)} columns.")

# =========================
# 📊 MODEL STATS
# =========================
def load_model_stats(model_prefix: str):
    if os.path.exists(REVIEW_PATH):
        try:
            df = pd.read_csv(REVIEW_PATH)
            if model_prefix in df.columns:
                counts = df[model_prefix].value_counts(dropna=False)
                total = int(counts.sum()) if counts.sum() else 1
                def pct(label): return round(float(counts.get(label, 0)) / total * 100, 2)
                return {"for": pct("for"), "against": pct("against"), "unclear": pct("unclear")}
        except Exception as e:
            print(f"⚠️ Could not compute stats for {model_prefix}: {e}")
            return {"for": 0.0, "against": 0.0, "unclear": 0.0}
    else:
        print("ℹ️ manual_review.csv not found.")
        return {"for": 0.0, "against": 0.0, "unclear": 0.0}

def format_stats_table(stats: dict) -> str:
    return "\n".join([
        "📊 Performance summary",
        "───────────────────────────────",
        f"Unclear : {stats['unclear']:6.2f} %",
        f"Against : {stats['against']:6.2f} %",
        f"For     : {stats['for']:6.2f} %",
        "───────────────────────────────",
    ])

# =========================
# 🧩 INTERACTIVE MENU
# =========================
def confirm_action(model_prefix: str) -> bool:
    choice = input(f"⚠️ Are you sure you want to disable {model_prefix}? (Y/N): ").strip().lower()
    return choice == "y"

def preview_files(model_prefix: str):
    print("📦 Files that will be moved:")
    for folder in [OUTPUT_DIR, "figures"]:
        if os.path.exists(folder):
            for fname in os.listdir(folder):
                if model_prefix in fname:
                    print(f"  - {os.path.join(folder, fname)}")

def interactive_disable():
    if not os.path.exists(REVIEW_PATH):
        print("⚠️ manual_review.csv not found.")
        return
    df = pd.read_csv(REVIEW_PATH)
    zero_shot_cols = [c for c in df.columns if c.startswith("stance_zero_shot")]
    if not zero_shot_cols:
        print("⚠️ No zero-shot columns found in manual_review.csv.")
        return

    print("📂 Available zero-shot models (with summaries):")
    for i, col in enumerate(zero_shot_cols, 1):
        stats = load_model_stats(col)
        print(f"\n{i}. {col}")
        print(format_stats_table(stats))

    choice_str = input("\nSelect model number to disable (or press Enter/c to cancel): ").strip()
    if choice_str.lower() in ["", "c", "cancel"]:
        print("🚫 Cancelled.")
        return
    try:
        choice = int(choice_str)
        if choice < 1 or choice > len(zero_shot_cols):
            print("❌ Invalid choice.")
            return
    except ValueError:
        print("❌ Please enter a valid number.")
        return

    model_prefix = zero_shot_cols[choice - 1]
    preview_files(model_prefix)
    if confirm_action(model_prefix):
        stats = load_model_stats(model_prefix)
        disable_model(model_prefix)
        log_action("DISABLE", model_prefix, stats)
    else:
        print("🚫 Action cancelled.")

def main_menu():
    print("\n📋 Zero-Shot Manager Menu")
    print("1. Disable a zero-shot model")
    print("2. Restore a backup")
    print("3. Exit")

    try:
        choice = int(input("Select an option: "))
    except ValueError:
        print("❌ Invalid input.")
        return

    if choice == 1:
        interactive_disable()
    elif choice == 2:
        restore_backup()
    elif choice == 3:
        print("👋 Exiting.")
    else:
        print("❌ Invalid choice.")

# =========================
# 🚀 ENTRY POINT
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manage zero-shot models and manual_review backups")
    parser.add_argument("--disable", nargs="?", const="", help="Disable a zero-shot model by prefix name or interactive if omitted")
    parser.add_argument("--restore", help="Restore backup by number or model prefix (interactive if omitted)")
    parser.add_argument("--latest", action="store_true", help="Restore the latest backup")
    args = parser.parse_args()

    if args.disable is not None:
        if args.disable == "":
            interactive_disable()
        else:
            stats = load_model_stats(args.disable)
            print(format_stats_table(stats))
            preview_files(args.disable)
            if confirm_action(args.disable):
                disable_model(args.disable)
                log_action("DISABLE", args.disable, stats)
            else:
                print("🚫 Action cancelled.")
    elif args.latest:
        restore_backup(latest=True)
    elif args.restore is not None:
        if args.restore == "":
            restore_backup()
        elif args.restore.isdigit():
            restore_backup(choice=int(args.restore))
        else:
            restore_backup(model_prefix=args.restore)
    else:
        # No arguments → show menu
        main_menu()
