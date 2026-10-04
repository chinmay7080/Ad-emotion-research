"""Draw an aggregate-only summary of the partial TVC35 follow-up.

The values come from the locked 2 October 2026 follow-up report. No participant
or advertisement-level data are read or stored by this script.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def interval(ax, y, value, lo, hi, color, label):
    ax.plot([lo, hi], [y, y], color=color, linewidth=3, solid_capstyle="round")
    ax.plot([lo, lo], [y - 0.07, y + 0.07], color=color, linewidth=2)
    ax.plot([hi, hi], [y - 0.07, y + 0.07], color=color, linewidth=2)
    ax.scatter([value], [y], s=70, color=color, zorder=3)
    ax.text(value, y + 0.18, label, ha="center", fontsize=9, color="#24323a")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.axvline(0, color="#87949a", linewidth=1, linestyle="--")
        ax.set_ylim(-0.45, 1.65)
        ax.set_yticks([0, 1])
        ax.tick_params(axis="both", length=0, pad=7, labelsize=9)
        ax.xaxis.grid(True, color="#e6ebed", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax = axes[0]
    interval(ax, 1, 0.250, 0.189, 0.290, "#167c80", "0.250")
    interval(ax, 0, 0.097, 0.070, 0.118, "#6c96a8", "0.097")
    ax.set_yticklabels(["Participant level", "Ad-level group"])
    ax.set_xlim(-0.02, 0.36)
    ax.set_title("Predicted vs measured fMRI", loc="left", fontweight="bold", fontsize=12)
    ax.set_xlabel("Residual correlation r (95% bootstrap CI)", fontsize=9)

    ax = axes[1]
    interval(ax, 1, -0.148, -0.470, 0.202, "#c76553", "−0.148")
    interval(ax, 0, 0.221, -0.172, 0.590, "#c76553", "0.221")
    ax.set_yticklabels(["Aided recall (24 ads)", "Preference (35 ads)"])
    ax.set_xlim(-0.58, 0.7)
    ax.set_title("Behavioral links not established", loc="left", fontweight="bold", fontsize=12)
    ax.set_xlabel("Spearman ρ (95% bootstrap CI)", fontsize=9)

    fig.suptitle(
        "TVC35 partial-cohort follow-up: 16 participants, 35 advertisements",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="#24323a",
    )
    fig.text(
        0.06,
        0.045,
        "After leave-one-ad-out generic-response removal. Exploratory partial cohort; no click outcomes.",
        fontsize=9,
        color="#52636b",
    )
    fig.subplots_adjust(left=0.18, right=0.97, top=0.81, bottom=0.23, wspace=0.40)

    out = Path(__file__).resolve().parent
    svg = out / "study1_aggregate_summary.svg"
    fig.savefig(svg, facecolor="white")
    fig.savefig(out / "study1_aggregate_summary.png", facecolor="white")
    plt.close(fig)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
