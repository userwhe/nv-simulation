"""Readout-only ideal-recovery protocol, fair controls, results, CSV, and plot."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .model import (
    NVModel,
    OpticalParameters,
    build_nv_model,
    lindblad_generator,
    propagate_states,
    reduced_nuclear_spin,
    to_nuclear_rotating_frame,
    trace_out_electron,
)
from .qec import (
    SixStateFidelity,
    TwoSpinCode,
    apply_recovery_in_lab_frame,
    build_common_fluctuator_code,
    decode_logical_density,
    density_matrix,
    encode_logical_density,
    six_state_average_memory_fidelity,
    six_state_test_states,
)
from .spins import SpinParameters


@dataclass(frozen=True)
class SimulationConfig:
    duration_us: float = 20.0
    sample_dt_us: float = 0.5
    b_field_t: float = 0.05
    optical: OpticalParameters = field(default_factory=OpticalParameters)

    def __post_init__(self) -> None:
        positive_fields = ("duration_us", "sample_dt_us")
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if not math.isfinite(self.b_field_t):
            raise ValueError("b_field_t must be finite")
        if not isinstance(self.optical, OpticalParameters):
            raise TypeError("optical must be an OpticalParameters instance")


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    spin_a: SpinParameters
    spin_b: SpinParameters
    t_us: np.ndarray
    qec_raw: np.ndarray
    qec_phase_corrected: np.ndarray
    spin_a_raw: np.ndarray
    spin_a_phase_corrected: np.ndarray
    spin_b_raw: np.ndarray
    spin_b_phase_corrected: np.ndarray
    best_single: np.ndarray
    qec_advantage: np.ndarray


def _times_through_duration(duration_us: float, step_us: float) -> np.ndarray:
    count = int(math.floor(duration_us / step_us + 1.0e-12))
    times = np.arange(count + 1, dtype=float) * step_us
    if times[-1] < duration_us - 1.0e-12:
        times = np.append(times, duration_us)
    else:
        times[-1] = duration_us
    return times


def _initial_states(
    model: NVModel,
    code: TwoSpinCode,
) -> tuple[list[np.ndarray], tuple[np.ndarray, ...]]:
    targets = six_state_test_states()
    electron_ground = np.zeros(model.electron_dimension, dtype=complex)
    electron_ground[0] = 1.0
    electron_state = density_matrix(electron_ground)
    spectator = np.eye(2, dtype=complex) / 2.0

    states: list[np.ndarray] = []
    for target in targets:
        encoded = encode_logical_density(density_matrix(target), code)
        states.append(np.kron(electron_state, encoded))
    for target in targets:
        physical_a = np.kron(density_matrix(target), spectator)
        states.append(np.kron(electron_state, physical_a))
    for target in targets:
        physical_b = np.kron(spectator, density_matrix(target))
        states.append(np.kron(electron_state, physical_b))
    return states, targets


def _rotating_nuclear_states(
    full_states: list[np.ndarray],
    model: NVModel,
    absolute_time_us: float,
) -> list[np.ndarray]:
    return [
        to_nuclear_rotating_frame(
            trace_out_electron(state, model),
            model,
            absolute_time_us,
        )
        for state in full_states
    ]


def _measure_memory_channels(
    states: list[np.ndarray],
    targets: tuple[np.ndarray, ...],
    model: NVModel,
    code: TwoSpinCode,
    absolute_time_us: float,
) -> tuple[SixStateFidelity, SixStateFidelity, SixStateFidelity]:
    logical_readout_states = [
        apply_recovery_in_lab_frame(state, code, model, absolute_time_us)
        for state in states[:6]
    ]

    logical_outputs = [
        decode_logical_density(nuclear_state, code)
        for nuclear_state in _rotating_nuclear_states(
            logical_readout_states, model, absolute_time_us
        )
    ]
    physical_a_outputs = [
        reduced_nuclear_spin(nuclear_state, 0)
        for nuclear_state in _rotating_nuclear_states(
            states[6:12], model, absolute_time_us
        )
    ]
    physical_b_outputs = [
        reduced_nuclear_spin(nuclear_state, 1)
        for nuclear_state in _rotating_nuclear_states(
            states[12:18], model, absolute_time_us
        )
    ]

    return (
        six_state_average_memory_fidelity(logical_outputs, targets),
        six_state_average_memory_fidelity(physical_a_outputs, targets),
        six_state_average_memory_fidelity(physical_b_outputs, targets),
    )


def simulate_qec_memory(
    config: SimulationConfig,
    spin_a: SpinParameters,
    spin_b: SpinParameters,
) -> SimulationResult:
    """Evolve uninterrupted and apply ideal recovery only to each readout copy."""

    if spin_a.id == spin_b.id:
        raise ValueError("spin ids must be distinct")

    model = build_nv_model(
        spin_a,
        spin_b,
        b_field_t=config.b_field_t,
        optical=config.optical,
    )
    code = build_common_fluctuator_code(spin_a, spin_b)
    generator = lindblad_generator(model)
    states, targets = _initial_states(model, code)

    sample_times = _times_through_duration(config.duration_us, config.sample_dt_us)
    qec_points: list[SixStateFidelity] = []
    spin_a_points: list[SixStateFidelity] = []
    spin_b_points: list[SixStateFidelity] = []

    current_time = 0.0
    for sample_time in sample_times:
        states = propagate_states(
            generator,
            states,
            sample_time - current_time,
            model.dimension,
        )
        current_time = sample_time
        qec, physical_a, physical_b = _measure_memory_channels(
            states,
            targets,
            model,
            code,
            sample_time,
        )
        qec_points.append(qec)
        spin_a_points.append(physical_a)
        spin_b_points.append(physical_b)

    qec_raw = np.array([point.raw for point in qec_points])
    qec_phase_corrected = np.array(
        [point.phase_corrected for point in qec_points]
    )
    spin_a_raw = np.array([point.raw for point in spin_a_points])
    spin_a_phase_corrected = np.array(
        [point.phase_corrected for point in spin_a_points]
    )
    spin_b_raw = np.array([point.raw for point in spin_b_points])
    spin_b_phase_corrected = np.array(
        [point.phase_corrected for point in spin_b_points]
    )
    best_single = np.maximum(spin_a_phase_corrected, spin_b_phase_corrected)
    qec_advantage = qec_phase_corrected - best_single
    return SimulationResult(
        config=config,
        spin_a=spin_a,
        spin_b=spin_b,
        t_us=sample_times,
        qec_raw=qec_raw,
        qec_phase_corrected=qec_phase_corrected,
        spin_a_raw=spin_a_raw,
        spin_a_phase_corrected=spin_a_phase_corrected,
        spin_b_raw=spin_b_raw,
        spin_b_phase_corrected=spin_b_phase_corrected,
        best_single=best_single,
        qec_advantage=qec_advantage,
    )


def save_results_csv(result: SimulationResult, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spin_a_prefix = f"spin_{result.spin_a.id}"
    spin_b_prefix = f"spin_{result.spin_b.id}"
    header = (
        "t_us",
        "qec_raw",
        "qec_phase_corrected",
        f"{spin_a_prefix}_raw",
        f"{spin_a_prefix}_phase_corrected",
        f"{spin_b_prefix}_raw",
        f"{spin_b_prefix}_phase_corrected",
        "best_single",
        "qec_advantage",
    )
    columns = (
        result.t_us,
        result.qec_raw,
        result.qec_phase_corrected,
        result.spin_a_raw,
        result.spin_a_phase_corrected,
        result.spin_b_raw,
        result.spin_b_phase_corrected,
        result.best_single,
        result.qec_advantage,
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(len(result.t_us)):
            writer.writerow(
                [f"{float(column[index]):.12g}" for column in columns]
            )
    return output_path


def save_fidelity_plot(result: SimulationResult, path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"

    figure, (fidelity_axis, advantage_axis) = plt.subplots(
        2,
        1,
        figsize=(8.4, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (2.1, 1.0)},
    )
    fidelity_axis.plot(
        result.t_us,
        result.qec_phase_corrected,
        linewidth=2.0,
        label="Logical QEC (recovery at readout)",
    )
    fidelity_axis.plot(
        result.t_us,
        result.spin_a_phase_corrected,
        label=(
            f"Physical spin {result.spin_a.id}: "
            f"A∥={result.spin_a.a_parallel_khz:g}, "
            f"A⊥={result.spin_a.a_perp_khz:g} kHz"
        ),
    )
    fidelity_axis.plot(
        result.t_us,
        result.spin_b_phase_corrected,
        label=(
            f"Physical spin {result.spin_b.id}: "
            f"A∥={result.spin_b.a_parallel_khz:g}, "
            f"A⊥={result.spin_b.a_perp_khz:g} kHz"
        ),
    )
    fidelity_axis.set_ylabel("Phase-corrected six-state\naverage memory fidelity")
    fidelity_axis.grid(alpha=0.25)
    fidelity_axis.legend(fontsize=8.5, loc="best")

    advantage_axis.plot(
        result.t_us,
        result.qec_advantage,
        color="#7c3aed",
        linewidth=1.8,
    )
    advantage_axis.axhline(0.0, color="black", linewidth=1.0)
    advantage_axis.set_xlabel("Elapsed time (µs)")
    advantage_axis.set_ylabel("QEC advantage")
    advantage_axis.grid(alpha=0.25)
    final_advantage = float(result.qec_advantage[-1])
    advantage_axis.annotate(
        f"final {final_advantage:+.4g}",
        xy=(result.t_us[-1], final_advantage),
        xytext=(-8, 12 if final_advantage <= 0.0 else -18),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )

    config = result.config
    optical = config.optical
    figure.suptitle(
        "Two-spin memory benchmark — ideal recovery at readout only\n"
        f"duration {config.duration_us:g} µs · B = {config.b_field_t:g} T · "
        f"P = {optical.laser_power_uw:g} µW · "
        f"pump = {optical.pump_rate_per_us:g} µs⁻¹",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(output_path, format="svg")
    plt.close(figure)
    return output_path


def save_all_pairs_plot(
    results: Sequence[SimulationResult],
    path: str | Path,
) -> Path:
    """Save a shared-scale small-multiple figure for a complete pair sweep."""

    if not results:
        raise ValueError("results must not be empty")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered_results = sorted(
        results,
        key=lambda result: (result.spin_a.id, result.spin_b.id),
    )
    config = ordered_results[0].config
    if any(result.config != config for result in ordered_results[1:]):
        raise ValueError("all pair results must use the same simulation configuration")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"

    column_count = 3
    row_count = math.ceil(len(ordered_results) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(15.0, 2.8 * row_count + 1.7),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fidelity_values = np.concatenate(
        [
            curve
            for result in ordered_results
            for curve in (
                result.qec_phase_corrected,
                result.spin_a_phase_corrected,
                result.spin_b_phase_corrected,
            )
        ]
    )
    minimum_fidelity = float(np.min(fidelity_values))
    lower_limit = max(
        0.0,
        minimum_fidelity - max(5.0e-4, 0.05 * (1.0 - minimum_fidelity)),
    )
    upper_limit = max(1.0005, float(np.max(fidelity_values)) + 1.0e-5)

    for axis, result in zip(axes.flat, ordered_results):
        axis.plot(
            result.t_us,
            result.qec_phase_corrected,
            linewidth=1.8,
            label="Logical QEC",
        )
        axis.plot(
            result.t_us,
            result.spin_a_phase_corrected,
            linewidth=1.2,
            label="First listed spin",
        )
        axis.plot(
            result.t_us,
            result.spin_b_phase_corrected,
            linewidth=1.2,
            label="Second listed spin",
        )
        final_advantage = float(result.qec_advantage[-1])
        axis.set_title(
            f"Spins {result.spin_a.id}–{result.spin_b.id} · "
            f"Δ={final_advantage:+.3g}\n"
            f"(A∥, A⊥) kHz: "
            f"{result.spin_a.id}=({result.spin_a.a_parallel_khz:g}, "
            f"{result.spin_a.a_perp_khz:g}); "
            f"{result.spin_b.id}=({result.spin_b.a_parallel_khz:g}, "
            f"{result.spin_b.a_perp_khz:g})",
            fontsize=8.5,
        )
        axis.set_ylim(lower_limit, upper_limit)
        axis.grid(alpha=0.22)
        axis.tick_params(labelsize=8)

    for axis in axes.flat[len(ordered_results) :]:
        axis.remove()

    optical = config.optical
    figure.suptitle(
        f"All {len(ordered_results)} two-spin combinations — "
        "ideal recovery at readout only\n"
        f"duration {config.duration_us:g} µs · B = {config.b_field_t:g} T · "
        f"P = {optical.laser_power_uw:g} µW · "
        f"pump = {optical.pump_rate_per_us:g} µs⁻¹ · shared fidelity scale",
        fontsize=14,
        y=0.995,
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=3,
        fontsize=9,
    )
    figure.supxlabel("Elapsed time (µs)", fontsize=11)
    figure.supylabel(
        "Phase-corrected six-state average memory fidelity",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.035, 0.035, 1.0, 0.925), h_pad=1.4)
    figure.savefig(output_path, format="svg")
    plt.close(figure)
    return output_path


def simulate_laser_power_sweep(
    config: SimulationConfig,
    spin_a: SpinParameters,
    spin_b: SpinParameters,
    laser_powers_uw: Sequence[float],
) -> tuple[SimulationResult, ...]:
    """Evaluate final-time memory fidelity at each requested laser power."""

    powers = tuple(float(power) for power in laser_powers_uw)
    if not powers:
        raise ValueError("laser_powers_uw must not be empty")
    if any(not math.isfinite(power) or power < 0.0 for power in powers):
        raise ValueError("laser powers must be finite and nonnegative")
    if len(set(powers)) != len(powers):
        raise ValueError("laser powers must be distinct")

    stopping_time_config = replace(config, sample_dt_us=config.duration_us)
    return tuple(
        simulate_qec_memory(
            replace(
                stopping_time_config,
                optical=replace(config.optical, laser_power_uw=power),
            ),
            spin_a,
            spin_b,
        )
        for power in powers
    )


def _ordered_power_sweep_results(
    results: Sequence[SimulationResult],
) -> list[SimulationResult]:
    if not results:
        raise ValueError("results must not be empty")

    ordered = sorted(results, key=lambda result: result.config.optical.laser_power_uw)
    reference = ordered[0]
    reference_optical = reference.config.optical
    for result in ordered[1:]:
        if (result.spin_a, result.spin_b) != (reference.spin_a, reference.spin_b):
            raise ValueError("all sweep results must use the same spin pair")
        if not math.isclose(result.config.duration_us, reference.config.duration_us):
            raise ValueError("all sweep results must use the same duration")
        if not math.isclose(result.config.b_field_t, reference.config.b_field_t):
            raise ValueError("all sweep results must use the same magnetic field")
        if replace(
            result.config.optical,
            laser_power_uw=reference_optical.laser_power_uw,
        ) != reference_optical:
            raise ValueError("only laser power may vary across sweep results")
    return ordered


def save_power_sweep_csv(
    results: Sequence[SimulationResult],
    path: str | Path,
) -> Path:
    """Save final-time channel metrics versus laser power."""

    ordered = _ordered_power_sweep_results(results)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spin_a_prefix = f"spin_{ordered[0].spin_a.id}"
    spin_b_prefix = f"spin_{ordered[0].spin_b.id}"
    header = (
        "laser_power_uw",
        "pump_rate_per_us",
        "qec_raw",
        "qec_phase_corrected",
        f"{spin_a_prefix}_raw",
        f"{spin_a_prefix}_phase_corrected",
        f"{spin_b_prefix}_raw",
        f"{spin_b_prefix}_phase_corrected",
        "best_single",
        "qec_advantage",
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for result in ordered:
            writer.writerow(
                f"{value:.12g}"
                for value in (
                    result.config.optical.laser_power_uw,
                    result.config.optical.pump_rate_per_us,
                    result.qec_raw[-1],
                    result.qec_phase_corrected[-1],
                    result.spin_a_raw[-1],
                    result.spin_a_phase_corrected[-1],
                    result.spin_b_raw[-1],
                    result.spin_b_phase_corrected[-1],
                    result.best_single[-1],
                    result.qec_advantage[-1],
                )
            )
    return output_path


def save_power_sweep_plot(
    results: Sequence[SimulationResult],
    path: str | Path,
) -> Path:
    """Save final-time memory fidelity and QEC advantage versus laser power."""

    ordered = _ordered_power_sweep_results(results)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"

    powers = np.array(
        [result.config.optical.laser_power_uw for result in ordered],
        dtype=float,
    )
    qec = np.array([result.qec_phase_corrected[-1] for result in ordered])
    spin_a_values = np.array(
        [result.spin_a_phase_corrected[-1] for result in ordered]
    )
    spin_b_values = np.array(
        [result.spin_b_phase_corrected[-1] for result in ordered]
    )
    advantage = np.array([result.qec_advantage[-1] for result in ordered])
    reference = ordered[0]

    figure, (fidelity_axis, advantage_axis) = plt.subplots(
        2,
        1,
        figsize=(8.4, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (2.1, 1.0)},
    )
    fidelity_axis.plot(
        powers,
        qec,
        marker="o",
        linewidth=2.0,
        label="Logical QEC (recovery at readout)",
    )
    fidelity_axis.plot(
        powers,
        spin_a_values,
        marker="o",
        label=(
            f"Physical spin {reference.spin_a.id}: "
            f"A∥={reference.spin_a.a_parallel_khz:g}, "
            f"A⊥={reference.spin_a.a_perp_khz:g} kHz"
        ),
    )
    fidelity_axis.plot(
        powers,
        spin_b_values,
        marker="o",
        label=(
            f"Physical spin {reference.spin_b.id}: "
            f"A∥={reference.spin_b.a_parallel_khz:g}, "
            f"A⊥={reference.spin_b.a_perp_khz:g} kHz"
        ),
    )
    fidelity_axis.set_ylabel("Phase-corrected six-state\naverage memory fidelity")
    fidelity_axis.grid(alpha=0.25)
    fidelity_axis.legend(fontsize=8.5, loc="best")

    advantage_axis.plot(
        powers,
        advantage,
        marker="o",
        color="#7c3aed",
        linewidth=1.8,
    )
    advantage_axis.axhline(0.0, color="black", linewidth=1.0)
    advantage_axis.set_xlabel("Laser power (µW)")
    advantage_axis.set_ylabel("QEC advantage")
    advantage_axis.grid(alpha=0.25)

    if np.all(powers > 0.0) and powers[-1] / powers[0] >= 10.0:
        advantage_axis.set_xscale("log")
    if len(powers) <= 12:
        advantage_axis.set_xticks(powers, [f"{power:g}" for power in powers])

    config = reference.config
    optical = config.optical
    figure.suptitle(
        "Laser-power sweep — ideal recovery at readout only\n"
        f"spins {reference.spin_a.id}–{reference.spin_b.id} · "
        f"readout at {config.duration_us:g} µs · B = {config.b_field_t:g} T · "
        f"pump/P = {optical.excitation_rate_per_us_per_uw:g} µs⁻¹/µW",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(output_path, format="svg")
    plt.close(figure)
    return output_path


def save_simulation_outputs(
    result: SimulationResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    csv_path = save_results_csv(result, directory / "results.csv")
    svg_path = save_fidelity_plot(result, directory / "fidelity_comparison.svg")
    return csv_path, svg_path


__all__ = [
    "SimulationConfig",
    "SimulationResult",
    "save_all_pairs_plot",
    "save_fidelity_plot",
    "save_power_sweep_csv",
    "save_power_sweep_plot",
    "save_results_csv",
    "save_simulation_outputs",
    "simulate_qec_memory",
    "simulate_laser_power_sweep",
]
