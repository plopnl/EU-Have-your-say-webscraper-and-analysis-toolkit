import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# === Constants ===
LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 10
TITLE_FONTSIZE = 14
FOOTER_FONTSIZE = 10

STANCE_COLORS = {
    "for": "#1f77b4",      # blue
    "against": "#d62728",  # red
    "unclear": "#9467bd"   # purple
}

# === Load stance dataset ===
df = pd.read_csv("data/stance_detection/stance_supervised.csv")

# === Date & stance prep ===
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
df = df.dropna(subset=["date"])
df["stance_supervised"] = (
    df["stance_supervised"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# === Normalize to calendar days ===
start_day = df["date"].min()
end_day = df["date"].max()
days_idx = pd.date_range(start_day, end_day, freq="D")

# === Daily counts by stance ===
stance_daily = (
    df[df["stance_supervised"].isin(["for", "against", "unclear"])]
      .groupby(["date", "stance_supervised"])
      .size()
      .unstack(level=1, fill_value=0)
      .reindex(days_idx, fill_value=0)
      .reindex(columns=["for", "against", "unclear"], fill_value=0)
)

# === Plot setup: main + inset ===
x = mdates.date2num(days_idx.to_pydatetime())
fig, ax = plt.subplots(figsize=(12, 6))

bar_width = 0.25
offsets = {"for": -bar_width, "against": 0.0, "unclear": bar_width}

# Main plot: full range
for stance in ["for", "against", "unclear"]:
    ax.bar(
        x + offsets[stance],
        stance_daily[stance].values,
        width=bar_width,
        color=STANCE_COLORS[stance],
        alpha=0.7,
        label=stance,
        align="center"
    )

ax.set_ylabel("Submission count", fontsize=LABEL_FONTSIZE)
ax.set_xlabel("Submission time", fontsize=LABEL_FONTSIZE)
ax.set_xlim(x.min() - 1, x.max() + 1)

ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.tick_params(axis="x", labelrotation=15)

# Inset: zoom on early days (up to Oct 9)
early_mask = (stance_daily.index <= pd.Timestamp("2025-10-09"))
x_early = mdates.date2num(stance_daily.index[early_mask].to_pydatetime())

# Shifted slightly right from left edge using bbox_to_anchor
axins = inset_axes(
    ax,
    width="60%", height="60%",
    loc="upper left",
    bbox_to_anchor=(0.05, -0.07, 1, 1),  # x-shifted anchor
    bbox_transform=ax.transAxes,
    borderpad=0
)

for stance in ["for", "against", "unclear"]:
    axins.bar(
        x_early + offsets[stance],
        stance_daily.loc[early_mask, stance].values,
        width=bar_width,
        color=STANCE_COLORS[stance],
        alpha=0.7,
        align="center"
    )

axins.set_title("Early phase", fontsize=LEGEND_FONTSIZE)
axins.set_ylabel("Count", fontsize=LABEL_FONTSIZE - 2)
axins.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
axins.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
axins.tick_params(axis="x", labelrotation=45)


# Title and footer
fig.suptitle("Daily submission volume by stance", fontsize=TITLE_FONTSIZE)
fig.text(
    0.5, 0.09,
    "Side-by-side bars show daily submission counts for 'for', 'against', and 'unclear'.\n"
    "Inset zoom reveals early activity before the late surge.\n"
    "Data source: EU consultation feedback (30 Mar 2021 – 22 Jun 2021)\n"
    "URL: https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/\n"
    "12645-Tobacco-taxation-excise-duties-for-manufactured-tobacco-products-updated-rules-_en",
    ha="center", va="center", fontsize=FOOTER_FONTSIZE, color="gray", linespacing=1.5
)

# Legend and diagnostics
ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)

missing_days_total = (stance_daily.sum(axis=1) == 0)
if missing_days_total.any():
    print(f"\n⚠️ Missing submission data on {missing_days_total.sum()} days:")
    print(stance_daily.index[missing_days_total].date)

fig.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("figures/submission_volume_by_day_stance_split.png", dpi=300)
plt.close()
print("📊 Saved plot: figures/submission_volume_by_day_stance_split.png")
