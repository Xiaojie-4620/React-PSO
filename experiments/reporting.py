"""Reporting utilities: LaTeX table generation and convergence curve plots."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt

from .runner import ExperimentReport


def generate_latex_table(
    reports: List[ExperimentReport],
    caption: str = "Benchmark results",
    label: str = "tab:results",
    precision: int = 2,
) -> str:
    """Generate a LaTeX table from experiment reports.

    Args:
        reports: List of ExperimentReport objects.
        caption: Table caption.
        label: LaTeX label.
        precision: Decimal places for numeric values.

    Returns:
        LaTeX source string.
    """
    mode_names = sorted(set(r.mode for r in reports))
    n_modes = len(mode_names)
    fmt = f"{{:.{precision}e}}"

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"  \begin{tabular}{l" + "c" * (2 * n_modes) + "}",
        r"    \hline",
    ]

    header = "    Function & Dim"
    for m in mode_names:
        header += f" & {m} (mean) & {m} (std)"
    header += r" \\"
    lines.append(header)
    lines.append(r"    \hline")

    grouped: Dict[str, Dict[int, Dict[str, ExperimentReport]]] = {}
    for r in reports:
        grouped.setdefault(r.func_name, {}).setdefault(r.dim, {})[r.mode] = r

    for func_name in sorted(grouped):
        for dim in sorted(grouped[func_name]):
            row = f"    {func_name} & {dim}"
            for m in mode_names:
                rep = grouped[func_name][dim].get(m)
                if rep:
                    row += f" & {fmt.format(rep.final_mean)} & {fmt.format(rep.final_std)}"
                else:
                    row += " & -- & --"
            row += r" \\"
            lines.append(row)

    lines.extend([
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def plot_convergence_curves(
    reports: List[ExperimentReport],
    output_path: Optional[str] = None,
    log_scale: bool = True,
    show_confidence: bool = True,
):
    """Plot convergence curves from experiment reports.

    Args:
        reports: Reports to plot, grouped by (func_name, dim).
        output_path: If set, save to file instead of showing.
        log_scale: Use log scale on y-axis.
        show_confidence: Show mean ± std band.
    """
    groups: Dict[tuple, List[ExperimentReport]] = {}
    for r in reports:
        key = (r.func_name, r.dim)
        groups.setdefault(key, []).append(r)

    n_groups = len(groups)
    cols = min(3, n_groups)
    rows = (n_groups + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, ((func_name, dim), group) in enumerate(sorted(groups.items())):
        ax = axes[idx // cols][idx % cols]
        for rep in sorted(group, key=lambda r: r.mode):
            histories = []
            for t in rep.trials:
                if t.history:
                    histories.append(np.array(t.history))

            if not histories:
                continue

            max_len = max(len(h) for h in histories)
            padded = np.full((len(histories), max_len), np.nan)
            for i, h in enumerate(histories):
                padded[i, : len(h)] = h

            mean = np.nanmean(padded, axis=0)
            iters = np.arange(len(mean))
            ax.plot(iters, mean, label=rep.mode, linewidth=1.5)

            if show_confidence and len(histories) >= 5:
                std = np.nanstd(padded, axis=0)
                ax.fill_between(iters, mean - std, mean + std, alpha=0.2)

        ax.set_title(f"{func_name} (D={dim})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Cost")
        if log_scale:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n_groups, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
