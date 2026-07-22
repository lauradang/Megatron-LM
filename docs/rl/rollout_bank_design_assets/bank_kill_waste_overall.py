# Generates bank_kill_waste_overall.png — run-level totals (absolute tokens)
# across each run's analyzed jobs: trained vs destroyed, pooled over all kills.
# Companion to bank_kill_waste.py (per-kill shares). Same palette/order.
#
# Sources: per-kill losses and splits from bank_kill_waste_extract.py and the
# R-row JSONL analysis; G/G training total anchored on sacct (120 iterations,
# 12 mid-generation kills); B/B kill count is counter-reset-derived and
# unverified, so its destroyed total is an upper bound.
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
NEUTRAL = "#e1e0d9"

CATS = [
    ("output_queue — complete groups", "#2a78d6"),
    ("reorder buffer — complete groups", "#008300"),
    ("partial groups — finished members", "#e87ba4"),
    ("engine active — mid-decode", "#eda100"),
]

# Absolute M tokens over the analyzed span: [trained, outq, reorder, partial, act]
data = {
    "R/G": [186, 164, 0, 51, 53],
    "R/B": [420, 162, 33, 99, 78],
    "G/G": [1431, 603, 0, 109, 62],
    "B/B": [1032, 163, 101, 11, 1],
}
notes = {
    "R/G": "454M tok · 59% destroyed · 2 kills",
    "R/B": "792M tok · 48% destroyed · 3 kills",
    "G/G": "2.21B tok · 35% destroyed · 12 kills (sacct)",
    "B/B": "1.31B tok · ≤21% destroyed · 19 resets (kills unverified)",
}

fig, ax = plt.subplots(figsize=(11.4, 3.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
plt.subplots_adjust(left=0.075, right=0.72, top=0.80, bottom=0.24)

runs = list(data.keys())
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlim(0, 2300)
ax.set_ylim(-0.55, len(runs) - 0.45)
ax.set_xticks(range(0, 2301, 500))
ax.set_xticklabels([f"{v/1000:.1f}B" if v >= 1000 else f"{v}M" for v in range(0, 2301, 500)],
                   color=MUTED, fontsize=8)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.set_yticks(range(len(runs)))
ax.set_yticklabels(runs, color=INK, fontsize=10, fontweight="bold")
ax.invert_yaxis()
ax.set_title("Total tokens generated over the analyzed jobs — trained (gray) vs destroyed (color)",
             loc="left", color=INK, fontsize=10.5, fontweight="bold", pad=8)


def lum(hex_):
    h = hex_.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


for yi, run in enumerate(runs):
    left = 0.0
    segs = [(data[run][0], NEUTRAL)] + [(v, c) for (lab, c), v in zip(CATS, data[run][1:])]
    for v, color in segs:
        if v <= 0:
            continue
        ax.barh(yi, v, left=left, height=0.52, color=color,
                edgecolor=SURFACE, linewidth=2, zorder=3)
        if v >= 100:
            ink = INK if lum(color) > 0.45 else "#ffffff"
            ax.text(left + v / 2, yi, f"{v/1000:.2f}B" if v >= 1000 else f"{v}M",
                    ha="center", va="center", color=ink, fontsize=8.5, zorder=4)
        left += v
    ax.text(left + 25, yi, notes[run], ha="left", va="center",
            color=INK2, fontsize=8, clip_on=False)

fig.suptitle("Overall run view: what each run generated and what the kills destroyed",
             x=0.075, y=0.97, ha="left", color=INK, fontsize=13, fontweight="bold")
fig.text(0.075, 0.885,
         "lag 5, 64×16, 32 GPUs · pooled across each run's analyzed jobs (absolute tokens, not per kill)",
         color=INK2, fontsize=9)

handles = [Patch(facecolor=NEUTRAL, edgecolor=SURFACE, label="trained on (useful output)")]
handles += [Patch(facecolor=c, edgecolor=SURFACE, label=lab) for lab, c in CATS]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, -0.01),
           ncol=3, frameon=False, fontsize=8, labelcolor=INK2, handlelength=1.2,
           handleheight=1.0, columnspacing=1.6)

out = "/Users/laurad/dev/adlr/rollout_bank_design_assets/bank_kill_waste_overall.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.18)
print("wrote", out)
