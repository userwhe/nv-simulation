"""Generate corrected process-fidelity plots for every spin pair in a register."""

from __future__ import annotations

import argparse
import csv
import itertools
import re
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["svg.fonttype"] = "none"

from nv_5level_two_c13_simulation import (
    choose_two_spins,
    load_dqp_spins,
    make_two_qubit_qec_code,
    model_params_from_spins,
    plot_process_fidelity,
    run_sanity_checks,
    save_process_csv,
    simulate_ideal_periodic_process,
)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def write_compact_grid_svg(spins, output_dir: Path, summary_rows: list[dict[str, object]]) -> None:
    """Write one compact SVG containing a process-fidelity plot for all 21 pairs."""

    width, height = 1500, 2380
    columns, rows = 3, 7
    margin_x, margin_top, panel_w, panel_h = 35, 120, 470, 305
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111}.title{font-size:25px;font-weight:600}.panel{font-size:15px;font-weight:600}.tick{font-size:11px}.note{font-size:12px}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:0.8}.qec{fill:none;stroke:#1f77b4;stroke-width:2}.s0{fill:none;stroke:#ff7f0e;stroke-width:1.5}.s1{fill:none;stroke:#2ca02c;stroke-width:1.5}.best{fill:none;stroke:#d62728;stroke-width:1.8;stroke-dasharray:5 4}</style>',
        '<text x="750" y="38" text-anchor="middle" class="title">All user-supplied spin pairs: QEC versus the best constituent spin</text>',
        '<text x="750" y="66" text-anchor="middle" class="note">Lab-frame five-level model; ideal recovery at 10 and 20 us; phase-calibrated six-state average fidelity</text>',
        '<line x1="390" y1="91" x2="430" y2="91" class="qec"/><text x="438" y="95" class="note">QEC</text>',
        '<line x1="505" y1="91" x2="545" y2="91" class="s0"/><text x="553" y="95" class="note">physical g1</text>',
        '<line x1="665" y1="91" x2="705" y2="91" class="s1"/><text x="713" y="95" class="note">physical g2</text>',
        '<line x1="825" y1="91" x2="865" y2="91" class="best"/><text x="873" y="95" class="note">best single</text>',
    ]

    by_name = {spin.name: spin for spin in spins}
    for panel_index, row in enumerate(summary_rows):
        pair = str(row["pair"])
        first_name, second_name = pair.split("-")
        ordered = sorted([by_name[first_name], by_name[second_name]], key=lambda spin: abs(spin.g_MHz), reverse=True)
        data = np.genfromtxt(output_dir / f"{safe_name(pair)}.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
        t = np.atleast_1d(data["t_us"]).astype(float)
        curves = [
            np.atleast_1d(data["qec_average_fidelity_phase_calibrated"]).astype(float),
            np.atleast_1d(data[f"{ordered[0].name}_average_fidelity_phase_calibrated"]).astype(float),
            np.atleast_1d(data[f"{ordered[1].name}_average_fidelity_phase_calibrated"]).astype(float),
            np.atleast_1d(data["best_single_average_fidelity"]).astype(float),
        ]
        col, grid_row = panel_index % columns, panel_index // columns
        x0 = margin_x + col * panel_w
        y0 = margin_top + grid_row * panel_h
        plot_x, plot_y, plot_w, plot_h = x0 + 54, y0 + 34, 390, 225
        minimum = min(float(np.min(curve)) for curve in curves)
        padding = max(0.0015, 0.12 * (1.0 - minimum))
        ymin, ymax = max(0.45, minimum - padding), 1.0005
        tmax = float(t[-1])

        def point_string(curve: np.ndarray) -> str:
            # Keep the committed overview compact while preserving endpoints and
            # recovery discontinuities. Individual CSV/SVG files retain all samples.
            keep = np.unique(np.linspace(0, len(t) - 1, min(21, len(t)), dtype=int))
            points = []
            for index in keep:
                tx, value = t[index], curve[index]
                px = plot_x + plot_w * float(tx) / tmax
                py = plot_y + plot_h * (ymax - float(value)) / (ymax - ymin)
                points.append(f"{px:.1f},{py:.1f}")
            return " ".join(points)

        lines.extend([
            f'<text x="{x0 + panel_w/2:.1f}" y="{y0 + 19}" text-anchor="middle" class="panel">{escape(pair)}</text>',
            f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" class="axis"/>',
            f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" class="axis"/>',
            f'<line x1="{plot_x}" y1="{plot_y + plot_h/2:.1f}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h/2:.1f}" class="grid"/>',
            f'<line x1="{plot_x + plot_w/2:.1f}" y1="{plot_y}" x2="{plot_x + plot_w/2:.1f}" y2="{plot_y + plot_h}" class="grid"/>',
            f'<text x="{plot_x - 5}" y="{plot_y + 4}" text-anchor="end" class="tick">{ymax:.4f}</text>',
            f'<text x="{plot_x - 5}" y="{plot_y + plot_h + 4}" text-anchor="end" class="tick">{ymin:.4f}</text>',
            f'<text x="{plot_x}" y="{plot_y + plot_h + 18}" text-anchor="middle" class="tick">0</text>',
            f'<text x="{plot_x + plot_w}" y="{plot_y + plot_h + 18}" text-anchor="middle" class="tick">{tmax:g} us</text>',
            f'<polyline points="{point_string(curves[0])}" class="qec"/>',
            f'<polyline points="{point_string(curves[1])}" class="s0"/>',
            f'<polyline points="{point_string(curves[2])}" class="s1"/>',
            f'<polyline points="{point_string(curves[3])}" class="best"/>',
            f'<text x="{plot_x + plot_w}" y="{plot_y + 14}" text-anchor="end" class="note">final Δ={100*float(row["final_qec_minus_best_single"]):+.4f} pp</text>',
        ])

    lines.append('</svg>')
    (output_dir / "all_pairs_grid.svg").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dqp-file", default="dqpspins.json")
    parser.add_argument("--time-us", type=float, default=20.0)
    parser.add_argument("--plot-dt-us", type=float, default=0.5)
    parser.add_argument("--recovery-interval-us", type=float, default=10.0)
    parser.add_argument("--B-T", type=float, default=0.05)
    parser.add_argument("--pump-rate", type=float, default=5.0)
    parser.add_argument("--output-dir", default="outputs/all_pairs")
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--force", action="store_true", help="recompute pairs even when CSV/SVG outputs exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.time_us <= 0 or args.plot_dt_us <= 0 or args.recovery_interval_us <= 0:
        raise ValueError("time, plot step, and recovery interval must be positive")

    spins = load_dqp_spins(args.dqp_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tlist = np.arange(0.0, args.time_us + 0.5 * args.plot_dt_us, args.plot_dt_us)
    summary_rows: list[dict[str, object]] = []

    for first, second in itertools.combinations(spins, 2):
        selected = choose_two_spins(spins, spin_names=(first.name, second.name))
        code = make_two_qubit_qec_code(tuple(s.g_MHz for s in selected))
        params = model_params_from_spins(selected, args.B_T, args.pump_rate)
        pair_label = f"{first.name}-{second.name}"
        stem = safe_name(pair_label)
        svg_path = output_dir / f"{stem}.svg"
        csv_path = output_dir / f"{stem}.csv"

        if csv_path.exists() and svg_path.exists() and not args.force:
            data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
            result = {
                "mode": "ideal",
                "t_us": np.atleast_1d(data["t_us"]).astype(float),
                "qec": np.atleast_1d(data["qec_average_fidelity_phase_calibrated"]).astype(float),
                "spin0": np.atleast_1d(data[f"{selected[0].name}_average_fidelity_phase_calibrated"]).astype(float),
                "spin1": np.atleast_1d(data[f"{selected[1].name}_average_fidelity_phase_calibrated"]).astype(float),
                "best_single": np.atleast_1d(data["best_single_average_fidelity"]).astype(float),
                "qec_minus_best": np.atleast_1d(data["qec_minus_best_single"]).astype(float),
            }
            print(f"{pair_label:8s} reused existing output")
        else:
            result = simulate_ideal_periodic_process(tlist, params, code, args.recovery_interval_us)
            if not args.skip_checks:
                run_sanity_checks(code, [result])
            save_process_csv(result, csv_path, (selected[0].name, selected[1].name))

        plot_process_fidelity(result, selected, svg_path)
        improvement = np.asarray(result["qec_minus_best"], dtype=float)
        nonzero_indices = np.where(np.asarray(result["t_us"], dtype=float) > 1e-12)[0]
        max_index = int(nonzero_indices[np.argmax(improvement[nonzero_indices])])
        ratio = min(abs(first.g_MHz), abs(second.g_MHz)) / max(abs(first.g_MHz), abs(second.g_MHz))
        summary_rows.append(
            {
                "pair": pair_label,
                "ordered_g1_spin": selected[0].name,
                "ordered_g2_spin": selected[1].name,
                "coupling_ratio_abs_gmin_over_gmax": ratio,
                "final_qec_average_fidelity": float(result["qec"][-1]),
                "final_best_single_average_fidelity": float(result["best_single"][-1]),
                "final_qec_minus_best_single": float(improvement[-1]),
                "maximum_qec_minus_best_single": float(improvement[max_index]),
                "time_of_maximum_improvement_us": float(result["t_us"][max_index]),
                "qec_beats_best_single_any_sample": bool(np.any(improvement[nonzero_indices] > 1e-9)),
                "qec_beats_best_single_at_final_time": bool(improvement[-1] > 1e-9),
            }
        )
        print(
            f"{pair_label:8s} final QEC={result['qec'][-1]:.6f}, "
            f"best={result['best_single'][-1]:.6f}, delta={improvement[-1]:+.6f}"
        )

    summary_path = output_dir / "all_pairs_summary.csv"
    fieldnames = list(summary_rows[0])
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    ranked = sorted(summary_rows, key=lambda row: float(row["maximum_qec_minus_best_single"]), reverse=True)
    labels = [str(row["pair"]) for row in ranked]
    values = [100.0 * float(row["maximum_qec_minus_best_single"]) for row in ranked]
    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    positions = np.arange(len(labels))
    ax.barh(positions, values)
    ax.set_yticks(positions, labels=labels)
    ax.invert_yaxis()
    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel("Maximum QEC advantage over best single spin (percentage points)")
    ax.set_title("All user-supplied spin pairs: corrected six-state process benchmark")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "all_pairs_ranking.svg")
    plt.close(fig)
    write_compact_grid_svg(spins, output_dir, summary_rows)

    print(f"Wrote {len(summary_rows)} pair plots and CSV files to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
