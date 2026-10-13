"""Build the README results figure from the preserved comparison CSV.

Run from any directory: python docs/generate_results_figure.py
Requires matplotlib. Does not train models or modify original results.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets"
LABELS = {
    "XGBoost (best trial)": "XGBoost",
    "Soft Voting Ensemble": "Soft voting",
    "Stacking Ensemble": "Stacking",
    "1D-CNN (best trial)": "1D-CNN",
}


def main():
    with (ROOT / "plots" / "model_comparison.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    background, ink, muted = "#F4F7FC", "#14243B", "#52647B"
    fig, ax = plt.subplots(figsize=(12, 8), dpi=160)
    fig.set_facecolor(background)
    ax.set_facecolor(background)
    fig.subplots_adjust(left=0.19, right=0.90, bottom=0.23, top=0.70)

    fig.text(0.065, 0.92, "AI IN GENOMICS  /  NYU", color="#136D78", fontsize=12,
             fontweight="bold")
    fig.text(0.065, 0.855, "Breast cancer subtype classification", color=ink,
             fontsize=25, fontweight="bold")
    fig.text(0.065, 0.803, "TCGA pan-cancer data | BRCA cohort | XGBoost, 1D-CNN, and ensembles",
             color=muted, fontsize=13)

    for i, row in enumerate(rows):
        accuracy = float(row["Test Accuracy"])
        macro_f1 = float(row["Test F1 (macro)"])
        ax.barh(i - 0.16, accuracy, height=0.27, color="#147D92",
                label="Test accuracy" if i == 0 else None)
        ax.barh(i + 0.16, macro_f1, height=0.27, color="#7193D5",
                label="Macro F1" if i == 0 else None)
        ax.text(accuracy + 0.013, i - 0.16, f"{accuracy:.2%}", va="center",
                color=ink, fontsize=12, fontweight="bold")
        ax.text(macro_f1 + 0.013, i + 0.16, f"{macro_f1:.3f}", va="center",
                color=muted, fontsize=12)

    ax.set_yticks(range(len(rows)), [LABELS[row["Model"]] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel("Score (0 to 1; ticks shown as percentages)", color=muted, labelpad=12)
    ax.grid(axis="x", color="#DCE4EF", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, colors=muted, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.025), ncol=2,
              frameon=False, fontsize=12)

    fig.text(0.065, 0.105, "Saved experiment: soft voting matched XGBoost; stacking scored slightly lower.",
             color=ink, fontsize=12, fontweight="bold")
    fig.text(0.065, 0.063, "Reported results | Preprocessing rerun pending | Source: model_comparison.csv",
             color=muted, fontsize=10)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        output_path = OUTPUT / f"results-overview.{suffix}"
        fig.savefig(output_path, facecolor=background)
        if suffix == "svg":
            output_path.write_text(
                "\n".join(line.rstrip() for line in output_path.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


if __name__ == "__main__":
    main()
