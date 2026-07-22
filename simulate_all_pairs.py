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


def write_compact_grid_svg(
    spins,
    output_dir: Path,
    summary_rows: list[dict[str, object]],
    excited_hyperfine_scale: float,
) -> None:
    """Write a compact 21-row SVG with one sparkline panel per spin pair."""
    width, height = 1180, 980
    label_x, plot_x, plot_w, top, row_h = 78, 130, 820, 105, 40
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111}.title{font-size:24px;font-weight:600}.label{font-size:13px;font-weight:600}.note{font-size:11px}.axis{stroke:#ddd;stroke-width:1}.qec{fill:none;stroke:#1f77b4;stroke-width:2}.s0{fill:none;stroke:#ff7f0e;stroke-width:1.3}.s1{fill:none;stroke:#2ca02c;stroke-width:1.3}.best{fill:none;stroke:#d62728;stroke-width:1.6;stroke-dasharray:4 3}</style>',
        '<text x="590" y="32" text-anchor="middle" class="title">All 21 spin pairs: QEC versus the best constituent spin</text>',
        f'<text x="590" y="54" text-anchor="middle" class="note">Lab-frame model; excited m_s=-1 hyperfine r={excited_hyperfine_scale:.3g}; ideal recovery at 10 and 20 us; six-state average fidelity</text>',
        '<line x1="280" y1="76" x2="315" y2="76" class="qec"/><text x="321" y="80" class="note">QEC</text>',
        '<line x1="385" y1="76" x2="420" y2="76" class="s0"/><text x="426" y="80" class="note">physical g1</text>',
        '<line x1="525" y1="76" x2="560" y2="76" class="s1"/><text x="566" y="80" class="note">physical g2</text>',
        '<line x1="665" y1="76" x2="700" y2="76" class="best"/><text x="706" y="80" class="note">best single</text>',
    ]

    by_name = {spin.name: spin for spin in spins}
    for row_index, row in enumerate(summary_rows):
        pair = str(row["pair"])
        first_name, second_name = pair.split("-")
        ordered = sorted(
            [by_name[first_name], by_name[second_name]],
            key=lambda spin: abs(spin.g_MHz),
            reverse=True,
        )
        data = np.genfromtxt(
            output_dir / f"{safe_name(pair)}.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        t = np.atleast_1d(data["t_us"]).astype(float)
        curves = [
            np.atleast_1d(data["qec_average_fidelity_phase_calibrated"]).astype(float),
            np.atleast_1d(data[f"{ordered[0].name}_average_fidelity_phase_calibrated"]).astype(float),
            np.atleast_1d(data[f"{ordered[1].name}_average_fidelity_phase_calibrated"]).astype(float),
            np.atleast_1d(data["best_single_average_fidelity"]).astype(float),
        ]
        keep = np.unique(np.linspace(0, len(t) - 1, min(5, len(t)), dtype=int))
        minimum = min(float(np.min(curve)) for curve in curves)
        ymax = 1.0005
        ymin = max(0.45, minimum - max(0.001, 0.08 * (1.0 - minimum)))
        y_mid = top + row_index * row_h
        plot_h = 27

        def points(curve: np.ndarray) -> str:
            out = []
            for index in keep:
                px = plot_x + plot_w * float(t[index]) / float(t[-1])
                py = y_mid + plot_h * (ymax - float(curve[index])) / (ymax - ymin)
                out.append(f"{px:.0f},{py:.1f}")
            return " ".join(out)

        delta = 100.0 * float(row["final_qec_minus_best_single"])
        lines.extend([
            f'<text x="{label_x}" y="{y_mid + 15}" text-anchor="end" class="label">{escape(pair)}</text>',
            f'<line x1="{plot_x}" y1="{y_mid + plot_h}" x2="{plot_x + plot_w}" y2="{y_mid + plot_h}" class="axis"/>',
            f'<polyline points="{points(curves[0])}" class="qec"/>',
            f'<polyline points="{points(curves[1])}" class="s0"/>',
            f'<polyline points="{points(curves[2])}" class="s1"/>',
            f'<polyline points="{points(curves[3])}" class="best"/>',
            f'<text x="975" y="{y_mid + 15}" class="note">Δ20={delta:+.4f} pp; y≥{ymin:.3f}</text>',
        ])

    lines.append('</svg>')
    (output_dir / "all_pairs_grid.svg").write_text("\n".join(lines), encoding="utf-8")


def write_compact_ranking_svg(
    output_dir: Path,
    ranked: list[dict[str, object]],
    excited_hyperfine_scale: float,
) -> None:
    """Write a compact dependency-free SVG ranking of every pair."""
    width, height = 1050, 760
    left, right, top, row_h = 125, 50, 75, 30
    values = [100.0 * float(row["maximum_qec_minus_best_single"]) for row in ranked]
    minimum, maximum = min(values), max(values)
    span = maximum - minimum if maximum > minimum else 1.0
    x_zero = left + (0.0 - minimum) / span * (width - left - right)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111}.title{font-size:23px;font-weight:600}.label{font-size:13px}.value{font-size:12px}.axis{stroke:#333;stroke-width:1}.bar{fill:#4c78a8}.positive{fill:#59a14f}</style>',
        '<text x="525" y="34" text-anchor="middle" class="title">Maximum QEC advantage over the best constituent spin</text>',
        f'<text x="525" y="56" text-anchor="middle" class="value">six-state average fidelity; percentage points; excited hyperfine r={excited_hyperfine_scale:.3g}</text>',
        f'<line x1="{x_zero:.1f}" y1="70" x2="{x_zero:.1f}" y2="710" class="axis"/>',
    ]
    for index, row in enumerate(ranked):
        value = values[index]
        y = top + index * row_h
        x_value = left + (value - minimum) / span * (width - left - right)
        x1, x2 = sorted((x_zero, x_value))
        klass = "positive" if value > 0 else "bar"
        lines.extend([
            f'<text x="{left - 8}" y="{y + 11}" text-anchor="end" class="label">{escape(str(row["pair"]))}</text>',
            f'<rect x="{x1:.1f}" y="{y}" width="{max(1.0, x2-x1):.1f}" height="17" class="{klass}"/>',
            f'<text x="{x_value + (5 if value >= 0 else -5):.1f}" y="{y + 12}" text-anchor="{"start" if value >= 0 else "end"}" class="value">{value:+.4f}</text>',
        ])
    lines.append('</svg>')
    (output_dir / "all_pairs_ranking.svg").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dqp-file", default="dqpspins.json")
    parser.add_argument("--time-us", type=float, default=20.0)
    parser.add_argument("--plot-dt-us", type=float, default=0.5)
    parser.add_argument("--recovery-interval-us", type=float, default=10.0)
    parser.add_argument("--B-T", type=float, default=0.05)
    parser.add_argument("--pump-rate", type=float, default=5.0)
    parser.add_argument(
        "--excited-hyperfine-scale",
        type=float,
        default=0.1,
        help="excited m_s=-1 tensor scale relative to the ground m_s=-1 tensor",
    )
    parser.add_argument("--output-dir", default="outputs/all_pairs")
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--force", action="store_true", help="recompute pairs even when CSV/SVG outputs exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.time_us <= 0 or args.plot_dt_us <= 0 or args.recovery_interval_us <= 0:
        raise ValueError("time, plot step, and recovery interval must be positive")
    if not np.isfinite(args.excited_hyperfine_scale):
        raise ValueError("excited hyperfine scale must be finite")

    spins = load_dqp_spins(args.dqp_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tlist = np.arange(0.0, args.time_us + 0.5 * args.plot_dt_us, args.plot_dt_us)
    summary_rows: list[dict[str, object]] = []

    for first, second in itertools.combinations(spins, 2):
        selected = choose_two_spins(spins, spin_names=(first.name, second.name))
        code = make_two_qubit_qec_code(tuple(s.g_MHz for s in selected))
        params = model_params_from_spins(
            selected,
            args.B_T,
            args.pump_rate,
            excited_state_hyperfine_scale=args.excited_hyperfine_scale,
        )
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
                "recovery_count": np.atleast_1d(data["recovery_count"]).astype(int),
                "rates": {"excited_state_hyperfine_scale": args.excited_hyperfine_scale},
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
                "excited_state_hyperfine_scale": float(args.excited_hyperfine_scale),
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
    write_compact_ranking_svg(output_dir, ranked, args.excited_hyperfine_scale)
    write_compact_grid_svg(spins, output_dir, summary_rows, args.excited_hyperfine_scale)

    print(f"Wrote {len(summary_rows)} pair plots and CSV files to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
