#!/usr/bin/env python3
# ============================================
# 📄 EU Feedback Split by User Type Script + Summary Bar Chart
# Created by Plop (@plopnl)
# Last updated: 2025-11-21
# Description: Splits feedback by userType into data/split_by_usertyp e/,
#              then generates a horizontal bar chart in figures/.
# Usage: python 7_split_by_usertype.py
# ============================================

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# ⚙️ PATHS & PLOT SETTINGS
# =========================
DATA_DIR = "data"
FIGURES_DIR = "figures"
INPUT_FILE = os.path.join(DATA_DIR, "feedback_details.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "split_by_usertype")

FONT_TITLE = 14
FONT_LABEL = 11
FONT_TICK = 9
FONT_ANNOTATION = 9
FONT_FOOTER = 8
LINE_SPACING_FOOTER = 1.5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# =========================
# 📥 LOAD FEEDBACK DATA
# =========================
df = pd.read_csv(INPUT_FILE)

# =========================
# 🔧 FILENAME SANITIZATION
# =========================
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", str(name).strip()) or "Unknown"

# =========================
# 📤 SPLIT AND SAVE BY USERTYPE
# =========================
for user_type, group in df.groupby("userType"):
    safe_name = sanitize_filename(user_type if pd.notna(user_type) else "Unknown")
    filename = os.path.join(OUTPUT_DIR, f"{safe_name}.csv")
    group.to_csv(filename, index=False)
    print(f"✅ Saved: {filename} ({len(group)} rows)")

# ============================================
# 📊 EU Feedback Submission Categories Summary
# ============================================
# Collect counts from saved CSVs
counts = {}
for filepath in glob.glob(os.path.join(OUTPUT_DIR, "*.csv")):
    category = os.path.splitext(os.path.basename(filepath))[0]
    df_cat = pd.read_csv(filepath)
    counts[category] = len(df_cat)

df_counts = pd.DataFrame(list(counts.items()), columns=["Category", "Count"])
df_counts["Percentage"] = df_counts["Count"] / df_counts["Count"].sum() * 100

def format_label(label):
    return label.replace("_", " ").title()
df_counts["Category"] = df_counts["Category"].apply(format_label)

fig, (ax, ax_footer) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [6, 1]})

bars = ax.barh(df_counts["Category"], df_counts["Count"], color="steelblue")

# Annotate with percentages
for bar, pct in zip(bars, df_counts["Percentage"]):
    if pct > 80:
        x_pos = bar.get_width() * 0.95
        color = "white"
        ha = "right"
    else:
        x_pos = bar.get_width() + max(df_counts["Count"]) * 0.01
        color = "black"
        ha = "left"
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", ha=ha, color=color, fontsize=FONT_ANNOTATION)

ax.set_title("Distribution of submission categories", fontsize=FONT_TITLE)
ax.set_xlabel("Number of entries", fontsize=FONT_LABEL)
ax.set_ylabel("Category", fontsize=FONT_LABEL)
ax.tick_params(axis="both", labelsize=FONT_TICK)

ax_footer.axis("off")
ax_footer.text(
    0.5, 0.5,
    "Data source: EU consultation feedback (03 September 2025 - 31 October 2025)\n"
    "URL: https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/\n"
    "     12645-Tobacco-taxation-excise-duties-for-manufactured-tobacco-products-updated-rules-_en\n"
    "Chart shows distribution of submission categories by count and percentage.\n"
    "Method: Entries split by userType; grouped CSVs saved in data/split_by_usertype.",
    ha="center", va="center", fontsize=FONT_FOOTER, color="gray", linespacing=LINE_SPACING_FOOTER
)

plt.subplots_adjust(top=0.92, bottom=0.2)
plt.tight_layout()

summary_path = os.path.join(FIGURES_DIR, "split_by_usertype_summary.png")
plt.savefig(summary_path, dpi=150)

print(f"📊 Saved summary plot to {summary_path}")
