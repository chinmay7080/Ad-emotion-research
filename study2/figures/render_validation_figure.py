"""Render the independently audited aggregate development-validation scores.

Only published-level summary numbers are embedded here. No clip-level data or
restricted AdCumen assets are needed to reproduce the figure.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCORES = {
    "Audio": 0.274345,
    "Video": 0.349853,
    "Video + audio": 0.353422,
    "Language": 0.391238,
    "Audio + language": 0.429771,
    "Video + language": 0.444810,
    "Video + audio + language": 0.455055,
}


def main() -> None:
    figure_dir = Path(__file__).resolve().parent
    labels = list(SCORES)
    values = list(SCORES.values())
    colors = [
        "#da7f36" if label == "Language" else
        "#167c80" if label == "Video + audio + language" else
        "#91a9b4"
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(9.5, 6.4), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.barh(labels, values, height=0.66, color=colors)
    ax.set_xlim(0, 0.53)
    ax.set_xlabel("Fixed eight-class balanced accuracy", fontsize=11, labelpad=10)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e6ebed", linewidth=0.8)
    ax.tick_params(axis="both", length=0, labelsize=10, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(
            value + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#263238",
        )

    fig.suptitle(
        "Content representations predict held-out development labels",
        x=0.10,
        y=0.97,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="#263238",
    )
    fig.text(
        0.10,
        0.90,
        "1,180 five-second clips from 467 parent advertisements",
        ha="left",
        fontsize=10,
        color="#596970",
    )
    fig.text(
        0.10,
        0.14,
        "Primary paired comparison: full content vs train-selected language.",
        ha="left",
        fontsize=9,
        color="#263238",
    )
    fig.text(
        0.10,
        0.105,
        "Difference +0.064 (95% parent-ad bootstrap CI +0.035 to +0.091).",
        ha="left",
        fontsize=9,
        color="#263238",
    )
    fig.text(
        0.10,
        0.055,
        "Development validation only; final test sealed. Content-model features, not measured or predicted brain responses.",
        ha="left",
        fontsize=8.5,
        color="#596970",
    )
    fig.subplots_adjust(left=0.29, right=0.93, top=0.83, bottom=0.29)
    svg_path = figure_dir / "validation_balanced_accuracy.svg"
    fig.savefig(svg_path, facecolor="white")
    fig.savefig(figure_dir / "validation_balanced_accuracy.png", facecolor="white")
    plt.close(fig)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )


if __name__ == "__main__":
    main()
