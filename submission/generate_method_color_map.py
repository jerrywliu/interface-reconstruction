"""Render the proposed paper-wide method color and marker mapping."""

from pathlib import Path

import matplotlib.pyplot as plt


GROUPS = [
    (
        "Linear baselines",
        [
            ("Youngs", "#B14E5E", "o", "-"),
            ("ELVIRA", "#B3811B", "s", "--"),
            ("LVIRA", "#2D7D64", "D", "-."),
        ],
    ),
    (
        "Higher-order baselines",
        [
            ("PLVIRA", "#2D7D64", "o", "-"),
            ("PCIC", "#7C5AA6", "s", "--"),
            ("QUASI", "#4B5563", "D", "-."),
        ],
    ),
    (
        "Ours: linear family",
        [
            ("Linear, per-cell", "#74A9CF", "^", "--"),
            ("Linear, graph-coordinated", "#2F6FA3", "v", "-"),
            ("Linear + corners, graph-coordinated", "#008C95", "X", "-"),
            ("Linear, graph-coordinated + joint C0", "#2F6FA3", "P", ":"),
        ],
    ),
    (
        "Ours: circular family",
        [
            ("Circular, per-cell", "#E6AB02", "^", "--"),
            ("Circular, graph-coordinated", "#D55E00", "v", "-"),
            ("Circular + corners, graph-coordinated", "#B91C1C", "X", "-"),
            ("Circular, graph-coordinated + joint C0", "#D55E00", "P", ":"),
            ("Circular + corners, graph-coordinated + joint C0", "#B91C1C", "P", ":"),
        ],
    ),
]


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results" / "submission" / "figure_consistency_20260903" / "method_color_map"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = sum(len(methods) + 1 for _, methods in GROUPS)
    fig, ax = plt.subplots(figsize=(9.0, 0.38 * rows + 0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, rows)
    ax.axis("off")

    y = rows - 0.7
    for group_name, methods in GROUPS:
        ax.text(0.03, y, group_name, fontsize=10.5, fontweight="bold", va="center")
        y -= 0.85
        for label, color, marker, linestyle in methods:
            ax.plot(
                [0.08, 0.28],
                [y, y],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                marker=marker,
                markersize=7.0,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.65,
            )
            ax.text(0.32, y, label, fontsize=9.5, va="center")
            ax.text(0.95, y, color.upper(), fontsize=8.5, color="#4B5563", ha="right", va="center", family="monospace")
            y -= 0.72
        y -= 0.25

    fig.tight_layout(pad=0.5)
    fig.savefig(out_dir / "method_color_mapping_proposal.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "method_color_mapping_proposal.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
