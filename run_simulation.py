"""Run the ideal instantaneous-recovery NV two-spin memory benchmark."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

from nv_qec import (
    BUILTIN_SPINS,
    SimulationConfig,
    SpinParameters,
    save_simulation_outputs,
    simulate_qec_memory,
)
from nv_qec.qec import infidelity_suppression_factor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spin-ids", nargs=2, type=int, metavar=("ID1", "ID2"))
    parser.add_argument("--spin1-a-parallel-khz", type=float)
    parser.add_argument("--spin1-a-perp-khz", type=float)
    parser.add_argument("--spin2-a-parallel-khz", type=float)
    parser.add_argument("--spin2-a-perp-khz", type=float)
    parser.add_argument("--duration-us", type=float, default=20.0)
    parser.add_argument("--sample-dt-us", type=float, default=0.5)
    parser.add_argument("--recovery-interval-us", type=float, default=10.0)
    parser.add_argument("--b-field-t", type=float, default=0.05)
    parser.add_argument("--pump-rate-per-us", type=float, default=5.0)
    parser.add_argument("--output-dir", default="outputs")
    return parser


def _resolve_spins(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[SpinParameters, SpinParameters]:
    direct_values = (
        args.spin1_a_parallel_khz,
        args.spin1_a_perp_khz,
        args.spin2_a_parallel_khz,
        args.spin2_a_perp_khz,
    )
    has_any_direct_value = any(value is not None for value in direct_values)
    has_all_direct_values = all(value is not None for value in direct_values)

    if args.spin_ids is not None and has_any_direct_value:
        parser.error("choose either --spin-ids or direct spin parameters, not both")
    if args.spin_ids is None and not has_all_direct_values:
        if has_any_direct_value:
            parser.error("all four direct spin parameters are required")
        parser.error("provide --spin-ids or all four direct spin parameters")

    if args.spin_ids is not None:
        first_id, second_id = args.spin_ids
        if first_id == second_id:
            parser.error("--spin-ids must contain two distinct IDs")
        unknown_ids = [spin_id for spin_id in args.spin_ids if spin_id not in BUILTIN_SPINS]
        if unknown_ids:
            choices = ", ".join(str(spin_id) for spin_id in BUILTIN_SPINS)
            parser.error(f"unknown spin ID(s) {unknown_ids}; available IDs are {choices}")
        return BUILTIN_SPINS[first_id], BUILTIN_SPINS[second_id]

    try:
        return (
            SpinParameters(1, direct_values[0], direct_values[1]),
            SpinParameters(2, direct_values[2], direct_values[3]),
        )
    except ValueError as error:
        parser.error(str(error))


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    spin_a, spin_b = _resolve_spins(args, parser)
    try:
        config = SimulationConfig(
            duration_us=args.duration_us,
            sample_dt_us=args.sample_dt_us,
            recovery_interval_us=args.recovery_interval_us,
            b_field_t=args.b_field_t,
            pump_rate_per_us=args.pump_rate_per_us,
        )
        result = simulate_qec_memory(config, spin_a, spin_b)
    except ValueError as error:
        parser.error(str(error))

    csv_path, svg_path = save_simulation_outputs(result, args.output_dir)
    qec_fidelity = float(result.qec_phase_corrected[-1])
    spin_a_fidelity = float(result.spin_a_phase_corrected[-1])
    spin_b_fidelity = float(result.spin_b_phase_corrected[-1])
    best_single = float(result.best_single[-1])
    advantage = float(result.qec_advantage[-1])
    suppression = infidelity_suppression_factor(qec_fidelity, best_single)

    print(
        f"Spin {spin_a.id}: A_parallel={spin_a.a_parallel_khz:g} kHz, "
        f"A_perp={spin_a.a_perp_khz:g} kHz"
    )
    print(
        f"Spin {spin_b.id}: A_parallel={spin_b.a_parallel_khz:g} kHz, "
        f"A_perp={spin_b.a_perp_khz:g} kHz"
    )
    print(f"QEC fidelity: {qec_fidelity:.9f}")
    print(f"Spin {spin_a.id} fidelity: {spin_a_fidelity:.9f}")
    print(f"Spin {spin_b.id} fidelity: {spin_b_fidelity:.9f}")
    print(f"Best single-spin fidelity: {best_single:.9f}")
    print(f"QEC advantage: {advantage:+.9f}")
    if suppression is None:
        print("Infidelity suppression factor: undefined (both fidelities are one)")
    elif math.isinf(suppression):
        print("Infidelity suppression factor: infinity")
    else:
        print(f"Infidelity suppression factor: {suppression:.9g}")
    if advantage > 0.0:
        print(
            "Interpretation: this is an ideal-model crossing, not experimental break-even."
        )
    print("Benchmark: ideal instantaneous recovery")
    print(f"Saved: {csv_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
