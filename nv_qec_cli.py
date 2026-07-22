"""Plotting, CSV output, validation, and command-line interface."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from nv_qec_types import *
from nv_qec_benchmarks import *


# ---------------------------------------------------------------------------
# Output and validation
# ---------------------------------------------------------------------------


def save_process_csv(result: dict, path: Path, spin_names: tuple[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t_us",
                "qec_average_fidelity_phase_calibrated",
                f"{spin_names[0]}_average_fidelity_phase_calibrated",
                f"{spin_names[1]}_average_fidelity_phase_calibrated",
                "best_single_average_fidelity",
                "qec_minus_best_single",
                "recovery_count",
            ]
        )
        for row in zip(
            result["t_us"],
            result["qec"],
            result["spin0"],
            result["spin1"],
            result["best_single"],
            result["qec_minus_best"],
            result["recovery_count"],
        ):
            writer.writerow([*(f"{float(value):.12g}" for value in row[:-1]), int(row[-1])])


def save_csv(result: dict, path: Path, spin_names: tuple[str, str] = ("spin0", "spin1")) -> None:
    save_process_csv(result, path, spin_names)


def plot_process_fidelity(result: dict, selected: Sequence[DQPSpin], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    names = (selected[0].name, selected[1].name)
    ax.plot(result["t_us"], result["qec"], label="Encoded + corrected")
    ax.plot(result["t_us"], result["spin0"], label=f"Physical {names[0]}")
    ax.plot(result["t_us"], result["spin1"], label=f"Physical {names[1]}")
    ax.plot(result["t_us"], result["best_single"], linestyle="--", label="Best single-spin envelope")
    ax.set_xlabel("Elapsed time (µs)")
    ax.set_ylabel("Phase-calibrated six-state average fidelity")
    protocol = "ideal instantaneous recovery" if result["mode"] in {"ideal", "single"} else "explicit pulsed ancilla sequence"
    ax.set_title(f"{names[0]}–{names[1]}: QEC versus both physical memories")
    plotted = np.concatenate([
        np.asarray(result["qec"], dtype=float),
        np.asarray(result["spin0"], dtype=float),
        np.asarray(result["spin1"], dtype=float),
        np.asarray(result["best_single"], dtype=float),
    ])
    minimum = float(np.min(plotted))
    padding = max(0.002, 0.15 * (1.0 - minimum))
    lower = max(0.45, minimum - padding)
    ax.set_ylim(lower, 1.002)
    if lower < 2.0 / 3.0:
        ax.axhline(2.0 / 3.0, linestyle=":", linewidth=1.0, label="Classical qubit threshold 2/3")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8.5)
    r = min(abs(selected[0].g_MHz), abs(selected[1].g_MHz)) / max(
        abs(selected[0].g_MHz), abs(selected[1].g_MHz)
    )
    excited_scale = result.get("rates", {}).get("excited_state_hyperfine_scale")
    excited_text = (
        f"excited m_s=-1 hyperfine scale={float(excited_scale):.3g}"
        if excited_scale is not None
        else "excited-state hyperfine scale not recorded"
    )
    ax.text(
        0.02,
        0.03,
        f"{protocol}\n|gmin/gmax|={r:.3f}; virtual-Z phase calibrated\n{excited_text}",
        transform=ax.transAxes,
        fontsize=7.8,
        va="bottom",
        bbox={"boxstyle": "round", "alpha": 0.12},
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_overlay(result: dict, selected: Sequence[DQPSpin], path: Path) -> None:
    plot_process_fidelity(result, selected, path)


def decoded_bell_fidelity(rho: np.ndarray, code: QECCode, de: int, dn: int) -> float:
    rho_n = partial_trace_electron(rho, de, dn)
    rho_decoded = code.decoding_unitary @ rho_n @ code.decoding_unitary.conj().T
    return bell_fidelity(rho_decoded)


def run_sanity_checks(code: QECCode, results: Sequence[dict]) -> None:
    basis = np.column_stack([code.zero_L, code.one_L, code.zero_E, code.one_E])
    orth_error = float(np.max(np.abs(basis.conj().T @ basis - np.eye(4))))
    cptp_error = float(
        np.max(np.abs(sum(K.conj().T @ K for K in code.recovery_kraus) - np.eye(4)))
    )
    if orth_error > 1e-12 or cptp_error > 1e-12:
        raise RuntimeError(f"QEC algebra check failed: orth={orth_error:.3e}, cptp={cptp_error:.3e}")
    for result in results:
        for key in ("qec", "spin0", "spin1", "best_single"):
            values = np.asarray(result[key])
            if np.min(values) < -1e-8 or np.max(values) > 1.0 + 1e-8:
                raise RuntimeError(f"Fidelity range check failed for {key}")


def _selected_spin_text(selected: Sequence[DQPSpin]) -> str:
    return ", ".join(
        f"{s.name}: -A_parallel={s.minus_A_par_kHz:.6g} kHz, A_perp={s.A_perp_kHz:.6g} kHz"
        for s in selected
    )


def parse_args(argv: Sequence[str] | None = None, default_protocol: str = "ideal") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dqp-file", default="dqpspins.json")
    parser.add_argument("--selection", choices=["strongest", "closest-positive"], default="strongest")
    parser.add_argument("--spin-indices", nargs=2, type=int, metavar=("I", "J"))
    parser.add_argument("--spin-names", nargs=2, metavar=("NAME1", "NAME2"))
    parser.add_argument("--protocol", choices=["ideal", "pulsed"], default=default_protocol)
    parser.add_argument("--time-us", type=float, default=20.0)
    parser.add_argument("--plot-dt-us", type=float, default=0.5)
    parser.add_argument("--recovery-interval-us", type=float, default=10.0)
    parser.add_argument("--B-T", type=float, default=0.05)
    parser.add_argument("--pump-rate", type=float, default=5.0)
    parser.add_argument(
        "--excited-hyperfine-scale",
        type=float,
        default=0.1,
        help=(
            "scale r for excited m_s=-1 hyperfine: A_parallel,e=r*A_parallel,g and "
            "A_perp,e=r*A_perp,g; use 0 for the former free-precession model"
        ),
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--laser-on-us", type=float, default=8.0)
    parser.add_argument("--laser-settle-us", type=float, default=0.25)
    parser.add_argument("--syndrome-gate-us", type=float, default=0.75)
    parser.add_argument("--measurement-us", type=float, default=0.50)
    parser.add_argument("--feedback-latency-us", type=float, default=0.50)
    parser.add_argument("--correction-gate-us", type=float, default=0.75)
    parser.add_argument("--electron-reset-fidelity", type=float, default=1.0)
    parser.add_argument("--syndrome-assignment-error", type=float, default=0.0)
    parser.add_argument("--nuclear-depolarizing-probability", type=float, default=0.0)
    parser.add_argument("--skip-checks", action="store_true")
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None, default_protocol: str = "ideal") -> None:
    args = parse_args(argv, default_protocol=default_protocol)
    spins = load_dqp_spins(args.dqp_file)
    selected = choose_two_spins(
        spins,
        args.selection,
        tuple(args.spin_indices) if args.spin_indices else None,
        tuple(args.spin_names) if args.spin_names else None,
    )
    params = model_params_from_spins(
        selected,
        args.B_T,
        args.pump_rate,
        excited_state_hyperfine_scale=args.excited_hyperfine_scale,
    )
    code = make_two_qubit_qec_code(tuple(s.g_MHz for s in selected))

    if args.protocol == "ideal":
        tlist = np.arange(0.0, args.time_us + 0.5 * args.plot_dt_us, args.plot_dt_us)
        result = simulate_ideal_periodic_process(tlist, params, code, args.recovery_interval_us)
        stem = f"{selected[0].name}_{selected[1].name}_ideal_process"
    else:
        config = PulsedSequenceConfig(
            cycles=args.cycles,
            laser_on_us=args.laser_on_us,
            laser_settle_us=args.laser_settle_us,
            syndrome_gate_us=args.syndrome_gate_us,
            measurement_us=args.measurement_us,
            feedback_latency_us=args.feedback_latency_us,
            correction_gate_us=args.correction_gate_us,
            electron_reset_fidelity=args.electron_reset_fidelity,
            syndrome_assignment_error=args.syndrome_assignment_error,
            nuclear_depolarizing_probability=args.nuclear_depolarizing_probability,
        )
        result = simulate_pulsed_experimental_process(params, code, config)
        stem = f"{selected[0].name}_{selected[1].name}_pulsed_process"

    if not args.skip_checks:
        run_sanity_checks(code, [result])

    output_dir = Path(args.output_dir)
    svg_path = output_dir / f"{stem}.svg"
    csv_path = output_dir / f"{stem}.csv"
    plot_process_fidelity(result, selected, svg_path)
    save_process_csv(result, csv_path, (selected[0].name, selected[1].name))

    print("Simulation completed.")
    print(f"Protocol: {args.protocol}")
    print(f"Selected spins: {_selected_spin_text(selected)}")
    print(f"Excited-state hyperfine scale: {args.excited_hyperfine_scale:.9g}")
    print(f"Final QEC average fidelity: {result['qec'][-1]:.6f}")
    print(f"Final best-single fidelity: {result['best_single'][-1]:.6f}")
    print(f"Final QEC - best single: {result['qec_minus_best'][-1]:+.6f}")
    print(f"Saved: {svg_path}")
    print(f"Saved: {csv_path}")


def main() -> None:
    run_cli()
