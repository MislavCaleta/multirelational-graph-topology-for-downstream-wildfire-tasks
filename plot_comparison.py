import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = Path("outputs/tables/clean_eval_topology_sweep.csv")
OUT_PATH = Path("outputs/figures/clean_eval_topology_comparison.png")
TOPOLOGIES = ["spatial", "temporal", "combined", "multiplex"]
MODELS = ["GCN", "GAT", "TransformerConv", "RGCN"]
K_VALUES = [5, 7, 9]
COLORS = {
    "GCN":             "#888888",
    "GAT":             "#1f77b4",
    "TransformerConv": "#2ca02c",
    "RGCN":            "#d62728",
    "MLP":             "#9467bd",
}


def load():
    rows = list(csv.DictReader(open(CSV_PATH)))
    grid = {t: {k: {} for k in K_VALUES} for t in TOPOLOGIES}
    mlp = None
    for r in rows:
        if r["model"] == "BaselineMLP":
            mlp = (float(r["f1_mean"]), float(r["f1_std"]))
            continue
        grid[r["topology"]][int(r["k"])][r["model"]] = (float(r["f1_mean"]), float(r["f1_std"]))
    return grid, mlp


def plot_panel(ax, topology, grid, mlp, ylim=(0.45, 0.75)):
    n_k = len(K_VALUES)
    bar_w = 0.18
    x = np.arange(n_k)
    models_present = [m for m in MODELS if m in grid[topology][K_VALUES[0]]]
    n_models = len(models_present)
    for i, m in enumerate(models_present):
        offset = (i - (n_models - 1) / 2) * bar_w
        means = [grid[topology][k][m][0] for k in K_VALUES]
        stds = [grid[topology][k][m][1] for k in K_VALUES]
        ax.bar(x + offset, means, bar_w, yerr=stds, capsize=3,
               label=m, color=COLORS[m], edgecolor="black", linewidth=0.5)

    ax.axhline(mlp[0], color=COLORS["MLP"], linestyle="--", linewidth=1.5,
               label=f"MLP (no graph): {mlp[0]:.3f} ± {mlp[1]:.3f}")
    ax.axhspan(mlp[0] - mlp[1], mlp[0] + mlp[1], color=COLORS["MLP"], alpha=0.12)

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in K_VALUES])
    ax.set_ylabel("Macro F1")
    ax.set_title(f"topology = {topology}")
    ax.set_ylim(ylim)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=7)


def main():
    grid, mlp = load()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, topology in zip(axes.flat, TOPOLOGIES):
        plot_panel(ax, topology, grid, mlp)

    fig.suptitle(
        "F1 macro across topologies, architectures and k\n"
        "Clean eval (val-driven model selection, causal edges, group-aware split, seasonal features, 5 seeds)",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.text(
        0.5, 0.005,
        "RGCN requires multi-relational edges and is therefore reported only for multiplex.",
        ha="center", fontsize=8, style="italic", color="#444444"
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
