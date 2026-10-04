"""Render the Pitt Funny-rating comparison from aggregate result tables only."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_comparison(path, comparison):
    with path.open(newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["target"] == "funny" and row["comparison"] == comparison
        ]
    if len(rows) != 1:
        raise ValueError(f"Expected one aggregate Funny row for {comparison}")
    return rows[0]


def parse_interval(raw):
    lo, hi = raw.strip("[]").split(",")
    return float(lo), float(hi)


def main():
    directory = Path(__file__).resolve().parent
    aggregate = directory.parent / "aggregate_results"
    matched = read_comparison(
        aggregate / "primary_brain_comparisons.csv", "gated_mlp_vs_content_mlp"
    )
    ridge = read_comparison(
        aggregate / "primary_ridge_comparisons.csv", "gated_mlp_vs_content_ridge"
    )
    items = [
        (
            "vs matched content MLP",
            float(matched["positive_improvement"]),
            parse_interval(matched["95% CI"]),
            parse_interval(matched["99.5% adjusted CI"]),
        ),
        (
            "vs stronger content Ridge",
            float(ridge["positive_improvement"]),
            (float(ridge["ci95_low"]), float(ridge["ci95_high"])),
            (
                float(ridge["ci99_5_low_ten_test_bonferroni"]),
                float(ridge["ci99_5_high_ten_test_bonferroni"]),
            ),
        ),
    ]

    fig, ax = plt.subplots(figsize=(9.3, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axvline(0, color="#74868e", linewidth=1.3, linestyle="--")
    for y, (label, point, ci95, ci_adjusted) in zip([1, 0], items):
        ax.plot(ci_adjusted, [y, y], color="#9bafbb", linewidth=10,
                solid_capstyle="butt", label="99.5% family-adjusted CI" if y == 1 else None)
        ax.plot(ci95, [y, y], color="#22384a", linewidth=3,
                solid_capstyle="round", label="95% CI" if y == 1 else None)
        for bound in ci95:
            ax.plot([bound, bound], [y - 0.075, y + 0.075],
                    color="#22384a", linewidth=2)
        ax.scatter([point], [y], s=90,
                   color="#167c80" if y == 1 else "#da7f36", zorder=3)
        ax.text(point, y + 0.18, f"{point:+.4f}", ha="center", fontsize=10)
    ax.set_yticks([1, 0], [item[0] for item in items])
    ax.set_ylim(-0.50, 1.55)
    ax.set_xlim(-0.016, 0.026)
    ax.xaxis.grid(True, color="#e7ecee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Difference in out-of-fold Spearman correlation", fontsize=10,
                  labelpad=9)
    ax.tick_params(axis="both", length=0, labelsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="lower left", frameon=False, ncol=2,
              bbox_to_anchor=(0.0, -0.42), fontsize=9)
    fig.suptitle(
        "Pitt Funny rating: a small matched gain, not a reliable Ridge gain",
        x=0.07, y=0.97, ha="left", fontsize=13, fontweight="bold",
        color="#24323a",
    )
    fig.text(0.07, 0.865, "1,865 ads · five ad-disjoint folds · predicted cortical features",
             fontsize=10, color="#53636b")
    fig.text(0.07, 0.045,
             "Positive values favor the gated content + predicted-brain MLP. Survey ratings, not clicks.",
             fontsize=9, color="#53636b")
    fig.subplots_adjust(left=0.30, right=0.95, top=0.76, bottom=0.26)
    svg = directory / "funny_comparison.svg"
    fig.savefig(svg, facecolor="white")
    fig.savefig(directory / "funny_comparison.png", facecolor="white")
    plt.close(fig)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
