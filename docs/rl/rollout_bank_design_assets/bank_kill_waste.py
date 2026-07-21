# Generates bank_kill_waste.png — where a 4h job's work dies at the SLURM kill,
# measured on the 2026-07-17 runs (R/G bl8qgebf, R/B rmunkfhb; averaged per kill).
# Top panel: composition of the job's total generated tokens (incl. trained-on).
# Bottom panel: headcount composition of the rollouts discarded at the kill.
# Palette: dataviz reference categorical slots 1-5 (validated, fixed order);
# trained-on is neutral context, not a series.
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
NEUTRAL = "#e1e0d9"  # trained-on (context, not a waste series)

CATS = [
    ("output_queue — complete groups", "#2a78d6", "Phase A"),
    ("reorder buffer — complete groups", "#008300", "Phase A"),
    ("partial groups — finished members", "#e87ba4", "Phase B"),
    ("engine active — mid-decode", "#eda100", "Phase C"),
    ("engine waiting — no GPU work yet", "#1baf7a", "no loss"),
]

# % of job's generated tokens (trained-on = remainder; snapshot/flow drift folded in)
tok = {
    "G/G": [65.2, 0.0, 11.8, 6.7, 0.1],
    "R/G": [36.0, 0.0, 11.3, 11.7, 0.3],
    "R/B": [21.0, 4.2, 13.1, 10.3, 0.3],
    "B/B": [12.4, 7.7, 0.9, 0.1, 0.1],
}
# % of rollouts discarded at the kill
cnt = {
    "G/G": [72.2, 0.0, 13.0, 14.8, 0.0],
    "R/G": [72.3, 0.0, 9.4, 8.3, 10.0],
    "R/B": [52.1, 6.8, 13.8, 12.4, 14.9],
    "B/B": [59.2, 35.7, 4.1, 0.2, 0.0],
}
tok_totals = {"G/G": "~76M tok / segment (kills ≈hourly)", "R/G": "~227M tok generated / job",
              "R/B": "~264M tok generated / job", "B/B": "~72M tok generated / job"}
cnt_totals = {"G/G": "5,986 rollouts lost / kill", "R/G": "33,575 rollouts lost / kill",
              "R/B": "22,464 rollouts lost / kill", "B/B": "1,210 rollouts lost / kill"}

fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
plt.subplots_adjust(left=0.075, right=0.985, top=0.845, bottom=0.225, hspace=0.62)


def lum(hex_):
    h = hex_.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def panel(ax, data, totals, title, with_trained):
    ax.set_facecolor(SURFACE)
    runs = list(data.keys())
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, len(runs) - 0.45)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{v}%" for v in range(0, 101, 20)], color=MUTED, fontsize=8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels(runs, color=INK, fontsize=10, fontweight="bold")
    ax.invert_yaxis()  # first run (R/G) on top, matching the doc's table order
    ax.set_title(title, loc="left", color=INK, fontsize=10.5, fontweight="bold", pad=8)

    for yi, run in enumerate(runs):
        y = yi
        left = 0.0
        segs = []
        if with_trained:
            trained = 100.0 - sum(data[run])
            segs.append((trained, NEUTRAL, "trained on"))
        segs += [(v, c, lab) for (lab, c, _), v in zip(CATS, data[run])]
        for v, color, lab in segs:
            if v <= 0:
                continue
            ax.barh(y, v, left=left, height=0.52, color=color,
                    edgecolor=SURFACE, linewidth=2, zorder=3)
            if v >= 4.0:
                txt = f"{v:.0f}%" if v >= 7 else f"{v:.0f}"
                ink = INK if lum(color) > 0.45 else "#ffffff"
                ax.text(left + v / 2, y, txt, ha="center", va="center",
                        color=ink, fontsize=8.5, zorder=4)
            left += v
        ax.text(101.2, y, totals[run], ha="left", va="center",
                color=INK2, fontsize=8, clip_on=False)


panel(axes[0], tok, tok_totals,
      "Share of the job's generated tokens  (≈ share of the job's generation GPU-time)",
      with_trained=True)
panel(axes[1], cnt, cnt_totals,
      "Share of rollouts discarded at the kill  (headcount)",
      with_trained=False)

fig.suptitle("Where each job's work dies at the SLURM kill",
             x=0.075, y=0.975, ha="left", color=INK, fontsize=13, fontweight="bold")
fig.text(0.075, 0.912,
         "lag 5, 64×16, 32 GPUs · G/G mkxx5cim (12 kills), R/G bl8qgebf (2), R/B rmunkfhb (3), B/B k9wstonf (19)",
         color=INK2, fontsize=9)

handles = [Patch(facecolor=NEUTRAL, edgecolor=SURFACE, label="trained on (useful output)")]
handles += [
    Patch(facecolor=c, edgecolor=SURFACE, label=f"{lab}  ·  {phase}")
    for lab, c, phase in CATS
]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, -0.005),
           ncol=3, frameon=False, fontsize=8, labelcolor=INK2, handlelength=1.2,
           handleheight=1.0, columnspacing=1.6)

out = "/Users/laurad/dev/adlr/rollout_bank_design_assets/bank_kill_waste.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.18)
print("wrote", out)
