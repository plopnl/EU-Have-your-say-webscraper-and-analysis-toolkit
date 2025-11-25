import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as mdates

# === Constants ===
LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 10
TITLE_FONTSIZE = 14
FOOTER_FONTSIZE = 10

# Distinct palettes for bars vs. lines
BAR_COLORS = {
    "for": "#66c2a5",      # soft green
    "against": "#fc8d62",  # warm orange-red
    "unclear": "#8da0cb"   # muted blue-purple
}

LINE_COLORS = {
    "for": "#1b9e77",      # deep green
    "against": "#d95f02",  # strong orange
    "unclear": "#7570b3"   # rich indigo
}

# === Load datasets ===
df_bot = pd.read_csv("data/bot_detection_v4.csv")
df_stance = pd.read_csv("data/stance_detection/stance_supervised.csv")

# === Merge stance labels into bot detection dataset ===
df = df_bot.merge(df_stance[["feedbackId", "stance_supervised"]], on="feedbackId", how="left")

# === Timestamp & stance prep ===
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
df["stance_supervised"] = df["stance_supervised"].astype(str).str.strip().str.lower()

# Normalize to calendar days
df["date"] = df["timestamp"].dt.normalize()
start_day = df["date"].min()
end_day = df["date"].max()
days_idx = pd.date_range(start_day, end_day, freq="D")

# === Diagnostic: submission days present ===
print("\n📅 Submission days present:")
print(df.groupby("date").size().rename("count"))

# === Unique stances ===
stances = df["stance_supervised"].dropna().unique()
print("📊 Stances found:", stances)

# === Line overlay: submission density by stance ===
plt.figure(figsize=(12, 6))
for stance in stances:
    subset = df[df["stance_supervised"] == stance]
    daily_counts = (
        subset.groupby("date")
              .size()
              .reindex(days_idx, fill_value=0)
    )
    if daily_counts.sum() > 0:
        plt.plot(days_idx, daily_counts.values, label=f"{stance} submissions")
plt.title("Submission density by stance (overlay)")
plt.xlabel("Date")
plt.ylabel("Submission count")
plt.legend()
plt.tight_layout()
plt.savefig("figures/stance_submission_density_overlay.png", dpi=300)
print("📊 Saved stance submission density overlay")

# === Line overlay: semantic similarity by stance ===
plt.figure(figsize=(12, 6))
for stance in stances:
    subset = df[df["stance_supervised"] == stance]
    counts = (
        subset.groupby("date")
              .size()
              .reindex(days_idx, fill_value=0)
    )
    similarity = (
        subset.groupby("date")["semantic_similarity_original"]
              .mean()
              .reindex(days_idx)
    )

    # Mask days with no submissions
    similarity = similarity.where(counts > 0)

    # Volume-weighted smoothing (3-day)
    weighted = (similarity * counts).rolling(window=3, min_periods=1).sum()
    smoothed = weighted / counts.rolling(window=3, min_periods=1).sum()

    # Fallback: interpolate only where counts are reasonable; keep zeros masked
    filled = smoothed.where(counts >= 3).interpolate(method="nearest")
    final_similarity = filled.where(counts > 0).ffill() * 100

    plt.plot(days_idx, final_similarity.values, label=f"{stance} similarity")
plt.title("Semantic similarity by stance (overlay)")
plt.xlabel("Date")
plt.ylabel("Semantic similarity (%)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/stance_similarity_overlay.png", dpi=300)
print("📊 Saved stance similarity overlay")

# === Stacked bar + similarity lines ===
def plot_submission_density(df, outfile):
    # Calendar index (already computed above)
    # days_idx is available in outer scope; recompute here for clarity & encapsulation
    start_day = df["date"].min()
    end_day = df["date"].max()
    days = pd.date_range(start_day, end_day, freq="D")

    # Daily stance counts via date + stance groupby
    stance_daily = (
        df.groupby(["date", "stance_supervised"])
          .size()
          .unstack(level=1, fill_value=0)
          .reindex(days, fill_value=0)
    )
    stance_daily = stance_daily.reindex(columns=["for", "against", "unclear"], fill_value=0)

    # Build stacked bars
    x = mdates.date2num(days.to_pydatetime())
    fig, ax1 = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(days))
    for stance in ["for", "against", "unclear"]:
        y = stance_daily[stance].values
        ax1.bar(x, y, width=0.8, color=BAR_COLORS[stance], alpha=0.6,
                label=stance.capitalize(), bottom=bottom, align="center")
        bottom += y

    # Format x-axis
    ax1.xaxis_date()
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.tick_params(axis="x", labelrotation=15)

    # Axis labels
    ax1.set_xlim(x.min() - 1, x.max() + 1)
    ax1.set_ylabel("Submission count", fontsize=LABEL_FONTSIZE)
    ax1.set_xlabel("Submission time", fontsize=LABEL_FONTSIZE)

    # Semantic similarity overlay
    ax2 = ax1.twinx()
    for stance in ["for", "against", "unclear"]:
        subset = df[df["stance_supervised"] == stance]

        counts = (
            subset.groupby("date")
                  .size()
                  .reindex(days, fill_value=0)
        )
        similarity = (
            subset.groupby("date")["semantic_similarity_combined"]
                  .mean()
                  .reindex(days)
        )

        # Mask days with no submissions
        similarity = similarity.where(counts > 0)

        # Volume-weighted smoothing
        weighted = (similarity * counts).rolling(window=3, min_periods=1).sum()
        smoothed = weighted / counts.rolling(window=3, min_periods=1).sum()

        # Fallback: interpolate only with adequate volume; keep zeros masked
        filled = smoothed.where(counts >= 3).interpolate(method="nearest")
        final_similarity = filled.where(counts > 0).ffill() * 100

        ax2.plot(days, final_similarity.values, label=stance.capitalize(),
                 color=LINE_COLORS[stance], alpha=0.9, linewidth=2)

    ax2.set_ylabel("Semantic similarity (%)", fontsize=LABEL_FONTSIZE)
    ax2.set_ylim(0, 100)

    # Legends
    ax1.legend(title="Stance", loc="upper left", fontsize=LEGEND_FONTSIZE)
    ax2.legend(title="Similarity", loc="upper right", fontsize=LEGEND_FONTSIZE)

    # Footer and title
    fig.suptitle("Submission timing vs semantic similarity", fontsize=TITLE_FONTSIZE)
    fig.text(
        0.5, 0.09,
        "Stance-colored bars show submission density over time. "
        "Colored lines show semantic similarity by stance.\n"
        "Low volume days are statistically noisy — a single outlier can skew the average.\n"
        "Data source: EU consultation feedback (30 Mar 2021 – 22 Jun 2021)\n"
        "URL: https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/\n"
        "12645-Tobacco-taxation-excise-duties-for-manufactured-tobacco-products-updated-rules-_en",
        ha="center", va="center", fontsize=FOOTER_FONTSIZE, color="gray", linespacing=1.5
    )

    # Missing day diagnostics
    missing_days = (stance_daily.sum(axis=1) == 0)
    if missing_days.any():
        print(f"\n⚠️ Missing submission data on {missing_days.sum()} days:")
        print(stance_daily.index[missing_days].date)

    fig.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"📊 Saved timing vs similarity plot: {outfile}")

# === Run final plot ===
plot_submission_density(df, "figures/submission_density_vs_similarity_by_stance.png")

# === Optional: print daily 'for' counts
for_daily = (
    df[df["stance_supervised"] == "for"]
    .groupby("date")
    .size()
    .rename("for_count")
    .reindex(days_idx, fill_value=0)
)

print("\n📅 Daily 'for' submission counts:")
print(for_daily[for_daily > 0])
