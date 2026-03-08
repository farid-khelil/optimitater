"""
Target Class Distribution Chart
Generates a bar chart of the 'Family' column from RBA.xlsx.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH   = '/home/farid/pfe/data/processed/ransomware/RBA.xlsx'
TARGET_COL  = 'Family'
OUT_DIR     = '/home/farid/pfe/results/figures'
OUT_FILE    = os.path.join(OUT_DIR, 'target_class_distribution.png')

# ── RedBull dark theme ─────────────────────────────────────────────────────────
RB_BG    = '#0D0D0D'
RB_PANEL = '#1A1A1A'
RB_RED   = '#E8292A'
RB_GOLD  = '#FFD700'
RB_WHITE = '#F0F0F0'
RB_GRAY  = '#888888'

PALETTE = [
    '#E8292A', '#4FC3F7', '#69F0AE', '#FFD54F', '#CE93D8',
    '#FF8A65', '#80DEEA', '#A5D6A7', '#FFF176', '#EF9A9A',
    '#B0BEC5', '#FFCC80', '#80CBC4', '#C5E1A5', '#F48FB1',
]

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"Loading data from {DATA_PATH} ...")
df = pd.read_excel(DATA_PATH)
print(f"  Shape : {df.shape}")
print(f"  Target: '{TARGET_COL}'  —  {df[TARGET_COL].nunique()} unique classes")

counts    = df[TARGET_COL].value_counts().sort_values(ascending=False)
labels    = counts.index.tolist()
values    = counts.values
n_classes = len(labels)

# Assign colours (cycle through palette)
colors = [PALETTE[i % len(PALETTE)] for i in range(n_classes)]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                          gridspec_kw={'width_ratios': [2, 1]},
                          facecolor=RB_BG)
fig.patch.set_facecolor(RB_BG)

# ── Left: Bar chart ────────────────────────────────────────────────────────────
ax_bar = axes[0]
ax_bar.set_facecolor(RB_PANEL)

x = np.arange(n_classes)
bars = ax_bar.bar(x, values, color=colors, width=0.65, zorder=3,
                  linewidth=0.5, edgecolor='#2a2a2a')

# Value labels on bars
for bar, val in zip(bars, values):
    pct = val / values.sum() * 100
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + values.max() * 0.012,
                f'{val:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=7.5,
                color=RB_WHITE, fontweight='bold')

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, rotation=45, ha='right',
                        fontsize=9, color=RB_WHITE)
ax_bar.set_ylabel('Number of Samples', color=RB_WHITE, fontsize=11, labelpad=8)
ax_bar.set_xlabel('Ransomware Family', color=RB_WHITE, fontsize=11, labelpad=8)
ax_bar.set_title('Target Class Distribution — Ransomware Families',
                  color=RB_GOLD, fontsize=14, fontweight='bold', pad=14)
ax_bar.tick_params(axis='both', colors=RB_WHITE)
ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax_bar.spines[:].set_color('#333333')
ax_bar.grid(axis='y', color='#2e2e2e', linewidth=0.8, zorder=0)

# Red accent line at top
ax_bar.axhline(values.max(), color=RB_RED, linewidth=0.8,
               linestyle='--', alpha=0.4, zorder=2)

# ── Right: Pie / Donut chart ───────────────────────────────────────────────────
ax_pie = axes[1]
ax_pie.set_facecolor(RB_PANEL)

wedges, texts, autotexts = ax_pie.pie(
    values,
    labels=None,
    colors=colors,
    autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
    pctdistance=0.78,
    startangle=140,
    wedgeprops=dict(width=0.55, linewidth=1.2, edgecolor=RB_BG),  # donut
)
for at in autotexts:
    at.set_color(RB_BG)
    at.set_fontsize(7.5)
    at.set_fontweight('bold')

# Centre label
ax_pie.text(0, 0, f'{values.sum():,}\nSamples', ha='center', va='center',
            fontsize=11, color=RB_WHITE, fontweight='bold')

ax_pie.set_title('Class Share', color=RB_GOLD, fontsize=13,
                  fontweight='bold', pad=10)

# Legend
legend = ax_pie.legend(
    wedges, labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.28),
    ncol=2,
    fontsize=8,
    framealpha=0,
    labelcolor=RB_WHITE,
)

# ── Footer ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         f'Dataset: RBA.xlsx  |  {n_classes} classes  |  {values.sum():,} total samples',
         ha='center', fontsize=9, color=RB_GRAY, style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(OUT_FILE, dpi=160, bbox_inches='tight', facecolor=RB_BG)
plt.close()

print(f"\n✅  Chart saved → {OUT_FILE}")
print("\nClass breakdown:")
for label, count in zip(labels, values):
    pct = count / values.sum() * 100
    print(f"  {label:<20}  {count:>6,}  ({pct:5.1f}%)")
