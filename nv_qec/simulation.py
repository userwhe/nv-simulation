"""Ideal periodic-recovery protocol, fair controls, results, CSV, and plot."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import (
    NVModel,
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
    recovery_interval_us: float = 10.0
    b_field_t: float = 0.05
    pump_rate_per_us: float = 5.0

    def __post_init__(self) -> None:
        positive_fields = (
            "duration_us",
            "sample_dt_us",
            "recovery_interval_us",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if not math.isfinite(self.b_field_t):
            raise ValueError("b_field_t must be finite")
        if not math.isfinite(self.pump_rate_per_us) or self.pump_rate_per_us < 0.0:
            raise ValueError("pump_rate_per_us must be finite and nonnegative")


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
    recovery_count: np.ndarray
    recovery_times_us: np.ndarray


def _times_through_duration(duration_us: float, step_us: float) -> np.ndarray:
    count = int(math.floor(duration_us / step_us + 1.0e-12))
    times = np.arange(count + 1, dtype=float) * step_us
    if times[-1] < duration_us - 1.0e-12:
        times = np.append(times, duration_us)
    else:
        times[-1] = duration_us
    return times


def _recovery_times(config: SimulationConfig) -> np.ndarray:
    count = int(
        math.floor(config.duration_us / config.recovery_interval_us + 1.0e-12)
    )
    times = np.arange(1, count + 1, dtype=float) * config.recovery_interval_us
    if len(times) and math.isclose(times[-1], config.duration_us, abs_tol=1.0e-12):
        times[-1] = config.duration_us
    return times


def _time_key(time_us: float) -> float:
    return round(float(time_us), 12)


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
    follows_scheduled_recovery: bool,
) -> tuple[SixStateFidelity, SixStateFidelity, SixStateFidelity]:
    logical_states = states[:6]
    if not follows_scheduled_recovery:
        logical_states = [
            apply_recovery_in_lab_frame(state, code, model, absolute_time_us)
            for state in logical_states
        ]

    logical_outputs = [
        decode_logical_density(nuclear_state, code)
        for nuclear_state in _rotating_nuclear_states(
            logical_states, model, absolute_time_us
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
    """Run the ideal-recovery benchmark in the user's original spin order."""

    if spin_a.id == spin_b.id:
        raise ValueError("spin ids must be distinct")

    model = build_nv_model(
        spin_a,
        spin_b,
        b_field_t=config.b_field_t,
        pump_rate_per_us=config.pump_rate_per_us,
    )
    code = build_common_fluctuator_code(spin_a, spin_b)
    generator = lindblad_generator(model)
    states, targets = _initial_states(model, code)

    sample_times = _times_through_duration(config.duration_us, config.sample_dt_us)
    recovery_times = _recovery_times(config)
    sample_keys = {_time_key(time_us) for time_us in sample_times}
    recovery_keys = {_time_key(time_us) for time_us in recovery_times}
    event_times = sorted(sample_keys | recovery_keys)
    qec_points: list[SixStateFidelity] = []
    spin_a_points: list[SixStateFidelity] = []
    spin_b_points: list[SixStateFidelity] = []
    recovery_counts: list[int] = []

    current_time = 0.0
    completed_recoveries = 0
    for event_time in event_times:
        states = propagate_states(
            generator,
            states,
            event_time - current_time,
            model.dimension,
        )
        current_time = event_time

        follows_scheduled_recovery = event_time in recovery_keys
        if follows_scheduled_recovery:
            states[:6] = [
                apply_recovery_in_lab_frame(state, code, model, event_time)
                for state in states[:6]
            ]
            completed_recoveries += 1

        if event_time not in sample_keys:
            continue
        qec, physical_a, physical_b = _measure_memory_channels(
            states,
            targets,
            model,
            code,
            event_time,
            follows_scheduled_recovery,
        )
        qec_points.append(qec)
        spin_a_points.append(physical_a)
        spin_b_points.append(physical_b)
        recovery_counts.append(completed_recoveries)

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
        recovery_count=np.asarray(recovery_counts, dtype=int),
        recovery_times_us=recovery_times,
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
        "recovery_count",
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
                [
                    *(f"{float(column[index]):.12g}" for column in columns),
                    int(result.recovery_count[index]),
                ]
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
        label="Logical QEC",
    )
    fidelity_axis.plot(
        result.t_us,
        result.spin_a_phase_corrected,
        label=f"Physical spin {result.spin_a.id}",
    )
    fidelity_axis.plot(
        result.t_us,
        result.spin_b_phase_corrected,
        label=f"Physical spin {result.spin_b.id}",
    )
    fidelity_axis.plot(
        result.t_us,
        result.best_single,
        linestyle="--",
        color="black",
        linewidth=1.3,
        label="Best single-spin envelope",
    )
    for index, recovery_time in enumerate(result.recovery_times_us):
        fidelity_axis.axvline(
            recovery_time,
            color="0.55",
            linestyle=":",
            linewidth=1.0,
            label="Scheduled recovery" if index == 0 else None,
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
    figure.suptitle(
        "Two-spin memory benchmark — ideal instantaneous recovery\n"
        f"duration {config.duration_us:g} µs · B = {config.b_field_t:g} T · "
        f"recovery interval {config.recovery_interval_us:g} µs",
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
    "save_fidelity_plot",
    "save_results_csv",
    "save_simulation_outputs",
    "simulate_qec_memory",
]
