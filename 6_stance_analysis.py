#!/usr/bin/env python3
# ============================================
# 📊 EU Feedback Stance Analysis Script
# Created by Plop (@plopnl)
# Last updated: 2025-11-21
# Description: Summarizes stance detection outputs (rule-based, zero-shot, supervised)
#              and generates plots for distribution and consensus.
# Usage: python 6_stance_analysis.py
# ============================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# ⚙️ SETTINGS
# =========================
DATA_DIR = "data"
INPUT_DIR = os.path.join(DATA_DIR, "stance_detection")
FIGURES_DIR = "figures"
MODEL_NAME = "deberta-v3-large"

# Font sizes (centralized for easy adjustment)
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 7
FOOTER_FONTSIZE = 7

# Ensure figures folder exists
os.makedirs(FIGURES_DIR, exist_ok=True)

# =========================
# 📥 LOAD DETECTION FILES
# =========================
files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
print(f"🔎 Found {len(files)} stance detection files in {INPUT_DIR}:")
for f in files:
    print(" -", f)

# Load rule-based separately
rule_file = [f for f in files if f.startswith("stance_rule")]
rule_df = pd.read_csv(os.path.join(INPUT_DIR, rule_file[0])) if rule_file else None

# Load zero-shot models
zs_files = [f for f in files if f.startswith("stance_zero_shot")]
zs_dfs = {f: pd.read_csv(os.path.join(INPUT_DIR, f)) for f in zs_files}

# Load supervised
sup_path = os.path.join(INPUT_DIR, "stance_supervised.csv")
sup_df = pd.read_csv(sup_path) if os.path.exists(sup_path) else None

def add_centered_footer(ax, location=0.02, extra_lines=None):
    fig = ax.get_figure()
    fig.subplots_adjust(bottom=0.22)

    footer_lines = [
        "Data source: Official consultation contributions (03 September 2025 - 31 October 2025)",
        "URL: https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/",
        "12645-Tobacco-taxation-excise-duties-for-manufactured-tobacco-products-updated-rules-_en"
    ]
    if extra_lines:
        footer_lines.extend(extra_lines)

    fig.text(
        0.5, location,
        "\n".join(footer_lines),
        ha="center", va="center",
        fontsize=FOOTER_FONTSIZE, color="gray"
    )
    
# =========================
# 📊 RULE-BASED SUMMARY + PLOT
# =========================
if rule_df is not None:
    print("\n📊 Rule-based stance summary:")
    rb_summary = rule_df["stance_rule"].value_counts(normalize=True).mul(100).round(2)
    print(rb_summary)

    plt.figure(figsize=(8,5))
    ax = rb_summary.plot(kind="bar", color="orange")
    ax.set_title("Rule-based Stance Distribution (Keywords)", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Stance category", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Percentage of submissions (%)", fontsize=LABEL_FONTSIZE)

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=LABEL_FONTSIZE)

    # Add percentage labels above bars
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f"{val:.1f}%",
                        (p.get_x() + p.get_width() / 2, val),
                        ha="center", va="bottom",
                        fontsize=LABEL_FONTSIZE, color="#333333",
                        xytext=(0, 0), textcoords="offset points")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "stance_rule.png")

    # Add centered footer
    add_centered_footer(ax, location=0.06)


    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✅ Saved figure: {fig_path}")

# =========================
# 📊 ZERO-SHOT SUMMARIES + PLOTS
# =========================
for fname, df in zs_dfs.items():
    model_name = fname.replace("stance_zero_shot_", "").replace(".csv", "")
    display_name = model_name.replace("_", " ")

    print(f"\n📊 Zero-shot stance summary ({model_name}):")
    zs_summary = df["stance_zero_shot"].value_counts(normalize=True).mul(100).round(2)
    print(zs_summary)

    plt.figure(figsize=(8,5))
    ax = zs_summary.plot(kind="bar", color="green")
    ax.set_title(f"Zero-shot Stance Distribution ({display_name})", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Stance category", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Percentage of submissions (%)", fontsize=LABEL_FONTSIZE)

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=LABEL_FONTSIZE)

    # Add percentage labels above bars
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f"{val:.1f}%",
                        (p.get_x() + p.get_width() / 2, val),
                        ha="center", va="bottom",
                        fontsize=LABEL_FONTSIZE, color="#333333",
                        xytext=(0, 0), textcoords="offset points")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"stance_zero_shot_{model_name}.png")

    # Add centered footer
    add_centered_footer(ax, location=0.06)

    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✅ Saved figure: {fig_path}")

# =========================
# 🧩 CONSENSUS (Rule vs Zero-shot)
# =========================
if rule_df is not None and zs_dfs:
    for fname, df in zs_dfs.items():
        model_name = fname.replace("stance_zero_shot_", "").replace(".csv", "")
        display_name = model_name.replace("_", " ")

        consensus = []
        for r_rule, r_zs in zip(rule_df["stance_rule"], df["stance_zero_shot"]):
            consensus.append(r_rule if r_rule == r_zs else "disagreement")
        df["stance_consensus"] = consensus

        summary_consensus = df["stance_consensus"].value_counts().reset_index()
        summary_consensus.columns = ["Stance", "Count"]
        summary_consensus["Percentage"] = summary_consensus["Count"] / summary_consensus["Count"].sum() * 100

        print(f"\n📊 Consensus stance summary ({model_name}):")
        print(summary_consensus)

        plt.figure(figsize=(8,5))
        ax = summary_consensus.set_index("Stance")["Percentage"].plot(kind="bar", color="purple")
        ax.set_title(f"Consensus Stance Distribution ({display_name})", fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Stance category", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Percentage of submissions (%)", fontsize=LABEL_FONTSIZE)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=LABEL_FONTSIZE)

        # Add percentage labels above bars
        for p in ax.patches:
            val = p.get_height()
            if val > 0:
                ax.annotate(f"{val:.1f}%",
                            (p.get_x() + p.get_width() / 2, val),
                            ha="center", va="bottom",
                            fontsize=LABEL_FONTSIZE, color="#333333",
                            xytext=(0, 0), textcoords="offset points")

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, f"consensus_{model_name}.png")

        # Add centered footer
        add_centered_footer(
            ax,
            location=0.07,
            extra_lines=[
                f"Consensus = rule-based stance if it matches {display_name} prediction, otherwise marked as disagreement.",
                "This highlights alignment between keyword-based and model-inferred classifications."
            ]
        )

        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"✅ Saved figure: {fig_path}")

# =========================
# 📊 SUPERVISED SUMMARY + PLOT + CONSENSUS
# =========================
if sup_df is not None and "stance_supervised" in sup_df.columns:
    print("\n📊 Supervised stance summary:")
    sup_summary = sup_df["stance_supervised"].value_counts(normalize=True).mul(100).round(2)
    print(sup_summary)

    plt.figure(figsize=(8,5))
    ax = sup_summary.plot(kind="bar", color="blue")
    ax.set_title(f"{MODEL_NAME} Stance Distribution", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Stance category", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Percentage of submissions (%)", fontsize=LABEL_FONTSIZE)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=LABEL_FONTSIZE)

    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f"{val:.1f}%",
                        (p.get_x() + p.get_width() / 2, val),
                        ha="center", va="bottom",
                        fontsize=LABEL_FONTSIZE, color="#333333",
                        xytext=(0, 0), textcoords="offset points")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "stance_supervised.png")
    add_centered_footer(
        ax,
        location=0.07,
        extra_lines=[
            f"Consensus = rule-based stance if it matches {MODEL_NAME} prediction, otherwise marked as disagreement.",
            "This highlights alignment between keyword-based and model-trained classifications."
        ]
    )
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✅ Saved figure: {fig_path}")

    # Consensus with rule-based
    if rule_df is not None:
        merged = pd.merge(
            rule_df[["feedbackId", "stance_rule"]],
            sup_df[["feedbackId", "stance_supervised"]],
            on="feedbackId"
        )
        consensus = []
        for r_rule, r_sup in zip(merged["stance_rule"], merged["stance_supervised"]):
            consensus.append(r_rule if r_rule == r_sup else "disagreement")
        merged["stance_consensus"] = consensus

        summary_consensus = merged["stance_consensus"].value_counts().reset_index()
        summary_consensus.columns = ["Stance", "Count"]
        summary_consensus["Percentage"] = summary_consensus["Count"] / summary_consensus["Count"].sum() * 100
        print(f"\n📊 Consensus stance summary ({MODEL_NAME}):")
        print(summary_consensus)

        plt.figure(figsize=(8,5))
        ax = summary_consensus.set_index("Stance")["Percentage"].plot(kind="bar", color="red")
        ax.set_title(f"Consensus Stance Distribution ({MODEL_NAME})", fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Stance category", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Percentage of submissions (%)", fontsize=LABEL_FONTSIZE)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=LABEL_FONTSIZE)

        for p in ax.patches:
            val = p.get_height()
            if val > 0:
                ax.annotate(f"{val:.1f}%",
                            (p.get_x() + p.get_width() / 2, val),
                            ha="center", va="bottom",
                            fontsize=LABEL_FONTSIZE, color="#333333",
                            xytext=(0, 0), textcoords="offset points")

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, f"consensus_{MODEL_NAME.replace('/', '_')}.png")
        add_centered_footer(
            ax,
            location=0.07,
            extra_lines=[
                f"Consensus = rule-based stance if it matches {MODEL_NAME} prediction, otherwise marked as disagreement.",
                "This highlights alignment between keyword-based and model-trained classifications."
            ]
        )
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"✅ Saved figure: {fig_path}")

# =========================
# 📈 COMPARISON PLOT (grouped by stance + disagreement)
# =========================
if rule_df is not None and zs_dfs:
    comparison = pd.DataFrame()

    # Rule-based
    rb_summary = rule_df["stance_rule"].value_counts(normalize=True).mul(100)
    comparison["Rule-based"] = rb_summary

    # Zero-shot models
    for fname, df in zs_dfs.items():
        model_name = fname.replace("stance_zero_shot_", "").replace(".csv", "")
        zs_summary = df["stance_zero_shot"].value_counts(normalize=True).mul(100)
        disagreement = (rule_df["stance_rule"] != df["stance_zero_shot"]).mean() * 100
        zs_summary["disagreement"] = disagreement
        comparison[model_name.replace("_", " ")] = zs_summary

    # Consensus (per model)
    for fname, df in zs_dfs.items():
        model_name = fname.replace("stance_zero_shot_", "").replace(".csv", "")
        if "stance_consensus" in df.columns:
            cons_summary = df["stance_consensus"].value_counts(normalize=True).mul(100)
            comparison[f"Consensus {model_name.replace('_', ' ')}"] = cons_summary

    # Supervised
    if sup_df is not None and "stance_supervised" in sup_df.columns:
        merged = pd.merge(
            rule_df[["feedbackId", "stance_rule"]],
            sup_df[["feedbackId", "stance_supervised"]],
            on="feedbackId"
        )
        sup_summary = merged["stance_supervised"].value_counts(normalize=True).mul(100)
        disagreement = (merged["stance_rule"] != merged["stance_supervised"]).mean() * 100
        sup_summary["disagreement"] = disagreement
        comparison[MODEL_NAME.replace("/", "-")] = sup_summary

    comparison = comparison.fillna(0)

    # Reformat for grouped bar plot
    melted = comparison.T.reset_index().melt(id_vars="index", var_name="Stance", value_name="Percentage")
    melted.rename(columns={"index": "Method"}, inplace=True)

    plt.figure(figsize=(10,6))
    ax = sns.barplot(data=melted, x="Stance", y="Percentage", hue="Method")
    ax.set_title(
        "Stance Detection Comparison (Percentages):\nRule-based vs Zero-shot vs Consensus vs " + MODEL_NAME,
        fontsize=TITLE_FONTSIZE
    )
    ax.set_xlabel("Stance category", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Percentage of submissions (%)", fontsize=LABEL_FONTSIZE)
    ax.legend(
        title="Method",
        loc="upper right",
        bbox_to_anchor=(1, 1.00),
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LABEL_FONTSIZE,
        frameon=True,
        borderaxespad=0.5
    )

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "stance_comparison.png")

    # Add centered footer with explanation
    add_centered_footer(
        ax,
        location=0.07,
        extra_lines=[
            f"Comparison includes rule-based, zero-shot, consensus, and {MODEL_NAME} stance detection outputs."
        ]
    )

    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n✅ Saved comparison figure: {fig_path}")
