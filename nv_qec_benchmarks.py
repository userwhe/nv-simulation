"""Six-state best-single benchmarks and ideal/pulsed QEC protocol models."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

from nv_qec_types import *
from nv_qec_dynamics import *

# ---------------------------------------------------------------------------
# Six-state process benchmark
# ---------------------------------------------------------------------------


def build_benchmark_initial_states(model: NVModel, code: QECCode) -> tuple[list[np.ndarray], list[np.ndarray]]:
    _, targets = memory_test_states()
    electron_g0 = density(ket(model.de, 0))
    spectator = I2 / 2.0
    states: list[np.ndarray] = []

    for target in targets:
        rho_in = density(target)
        rho_logical = code.logical_isometry @ rho_in @ code.logical_isometry.conj().T
        states.append(np.kron(electron_g0, rho_logical))
    for target in targets:
        states.append(np.kron(electron_g0, np.kron(density(target), spectator)))
    for target in targets:
        states.append(np.kron(electron_g0, np.kron(spectator, density(target))))
    return [hermitize(rho) for rho in states], targets


def _benchmark_outputs(
    states: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    model: NVModel,
    code: QECCode,
    t_us: float,
) -> dict[str, float]:
    if len(states) != 18:
        raise ValueError("Benchmark state batch must contain 18 density matrices")

    logical_outputs: list[np.ndarray] = []
    spin0_outputs: list[np.ndarray] = []
    spin1_outputs: list[np.ndarray] = []
    for index, rho in enumerate(states):
        rho_n = partial_trace_electron(rho, model.de, model.dn)
        rho_rot = to_rotating_nuclear_frame(rho_n, model, t_us)
        if index < 6:
            logical_outputs.append(hermitize(code.logical_isometry.conj().T @ rho_rot @ code.logical_isometry))
        elif index < 12:
            spin0_outputs.append(hermitize(partial_trace_two_qubit(rho_rot, 0)))
        else:
            spin1_outputs.append(hermitize(partial_trace_two_qubit(rho_rot, 1)))

    q_raw, q_cal, q_phi = phase_calibrated_six_state_fidelity(logical_outputs, targets)
    s0_raw, s0_cal, s0_phi = phase_calibrated_six_state_fidelity(spin0_outputs, targets)
    s1_raw, s1_cal, s1_phi = phase_calibrated_six_state_fidelity(spin1_outputs, targets)
    return {
        "qec_raw": q_raw,
        "qec": q_cal,
        "qec_phase": q_phi,
        "spin0_raw": s0_raw,
        "spin0": s0_cal,
        "spin0_phase": s0_phi,
        "spin1_raw": s1_raw,
        "spin1": s1_cal,
        "spin1_phase": s1_phi,
        "best_single": max(s0_cal, s1_cal),
        "qec_minus_best": q_cal - max(s0_cal, s1_cal),
    }


def _empty_process_result(tlist_us: np.ndarray) -> dict[str, np.ndarray]:
    keys = (
        "qec_raw",
        "qec",
        "qec_phase",
        "spin0_raw",
        "spin0",
        "spin0_phase",
        "spin1_raw",
        "spin1",
        "spin1_phase",
        "best_single",
        "qec_minus_best",
    )
    return {key: np.zeros(len(tlist_us), dtype=float) for key in keys}


def _store_benchmark_point(
    result: dict[str, np.ndarray],
    index: int,
    states: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    model: NVModel,
    code: QECCode,
    t_us: float,
) -> None:
    values = _benchmark_outputs(states, targets, model, code, t_us)
    for key, value in values.items():
        result[key][index] = value


def simulate_ideal_recovery_schedule(
    tlist_us: Sequence[float],
    model_params: dict,
    code: QECCode,
    recovery_times_us: Iterable[float],
) -> dict:
    """Lab-frame continuous optical evolution with ideal instantaneous recovery."""

    tlist = np.asarray(tlist_us, dtype=float)
    if len(tlist) < 2 or abs(tlist[0]) > 1e-15 or np.any(np.diff(tlist) <= 0):
        raise ValueError("tlist_us must be strictly increasing and start at zero")
    model = build_nv_two_c13_lab_model(**model_params)
    L = lindblad_liouvillian(model.H_lab, model.collapse_ops)
    states, targets = build_benchmark_initial_states(model, code)
    result = _empty_process_result(tlist)
    recovery_count = np.zeros(len(tlist), dtype=int)
    _store_benchmark_point(result, 0, states, targets, model, code, 0.0)

    schedule = sorted({float(t) for t in recovery_times_us if 0.0 < float(t) <= tlist[-1] + 1e-12})
    boundaries = sorted(set(schedule + [float(tlist[-1])]))
    current_time = 0.0
    count = 0
    tol = 1e-11

    for boundary in boundaries:
        indices = np.where((tlist > current_time + tol) & (tlist <= boundary + tol))[0]
        output_times = tlist[indices]
        eval_times = list(output_times)
        boundary_sampled = len(output_times) > 0 and abs(output_times[-1] - boundary) <= 1e-9
        is_recovery = any(abs(boundary - t) <= 1e-9 for t in schedule)
        if is_recovery and not boundary_sampled:
            eval_times.append(boundary)
        eval_times_array = np.asarray(eval_times, dtype=float)
        local_times = eval_times_array - current_time
        evolved_batches = evolve_state_batch_to_times(L, states, local_times, model.dim) if len(eval_times) else []
        by_time = {round(float(t), 12): batch for t, batch in zip(eval_times_array, evolved_batches)}

        for idx, t in zip(indices, output_times):
            _store_benchmark_point(result, int(idx), by_time[round(float(t), 12)], targets, model, code, float(t))
            recovery_count[int(idx)] = count

        if is_recovery:
            end_states = by_time.get(round(boundary, 12))
            if end_states is None:
                end_states = propagate_state_batch(L, states, boundary - current_time, model.dim)
            post_states = list(end_states)
            for i in range(6):
                post_states[i] = apply_recovery_to_full_state(end_states[i], code, model.de, model, boundary)
            states = post_states
            count += 1
            matches = np.where(np.abs(tlist - boundary) <= 1e-9)[0]
            if len(matches):
                idx = int(matches[0])
                _store_benchmark_point(result, idx, states, targets, model, code, boundary)
                recovery_count[idx] = count
        elif evolved_batches:
            states = evolved_batches[-1]
        current_time = boundary

    result.update(
        {
            "mode": "ideal",
            "t_us": tlist,
            "recovery_count": recovery_count,
            "recovery_times_us": np.asarray(schedule, dtype=float),
            "rates": model.rates,
        }
    )
    return result


def simulate_ideal_periodic_process(
    tlist_us: Sequence[float],
    model_params: dict,
    code: QECCode,
    recovery_interval_us: float,
) -> dict:
    if recovery_interval_us <= 0:
        raise ValueError("recovery_interval_us must be positive")
    end = float(np.asarray(tlist_us)[-1])
    count = int(math.floor(end / recovery_interval_us + 1e-12))
    schedule = [recovery_interval_us * i for i in range(1, count + 1)]
    result = simulate_ideal_recovery_schedule(tlist_us, model_params, code, schedule)
    result["recovery_interval_us"] = float(recovery_interval_us)
    return result


def simulate_single_recovery_process(
    tlist_us: Sequence[float],
    model_params: dict,
    code: QECCode,
    recovery_time_us: float,
) -> dict:
    result = simulate_ideal_recovery_schedule(tlist_us, model_params, code, [recovery_time_us])
    result["mode"] = "single"
    result["single_recovery_time_us"] = float(recovery_time_us)
    return result


# Backward-compatible names now return the corrected process benchmark.
def simulate_periodic_recovery(tlist_us, model_params, code, recovery_interval_us):
    return simulate_ideal_periodic_process(tlist_us, model_params, code, recovery_interval_us)


def simulate_single_recovery_trace(tlist_us, model_params, code, single_recovery_time_us):
    return simulate_single_recovery_process(tlist_us, model_params, code, single_recovery_time_us)


def simulate_pulsed_experimental_process(
    model_params: dict,
    code: QECCode,
    config: PulsedSequenceConfig,
) -> dict:
    """Segmented optical-noise and explicit electron-ancilla recovery model.

    Each cycle executes laser-on noise, pump-off settling, electron reset,
    finite syndrome-gate time, ancilla measurement/dephasing, feedback latency,
    finite conditional-correction time, and a final electron reset.  The QEC
    branch receives the syndrome and correction unitaries; the physical-spin
    controls experience the same elapsed time and electron resets but no QEC
    gates.  This function is intentionally separate from the ideal map model.
    """

    config.validate()
    model_on = build_nv_two_c13_lab_model(**model_params)
    dark_params = dict(model_params)
    dark_params["W_pump_per_us"] = 0.0
    model_off = build_nv_two_c13_lab_model(**dark_params)
    L_on = lindblad_liouvillian(model_on.H_lab, model_on.collapse_ops)
    L_off = lindblad_liouvillian(model_off.H_lab, model_off.collapse_ops)
    states, targets = build_benchmark_initial_states(model_on, code)
    U_syn, U_corr = ancilla_recovery_unitaries(code, model_on.de)

    times = [0.0]
    rows = [_benchmark_outputs(states, targets, model_on, code, 0.0)]
    current_time = 0.0

    def evolve_all(L: sp.csr_matrix, duration: float) -> None:
        nonlocal states, current_time
        if duration > 0:
            states = propagate_state_batch(L, states, duration, model_on.dim)
            current_time += duration

    for _ in range(config.cycles):
        evolve_all(L_on, config.laser_on_us)
        evolve_all(L_off, config.laser_settle_us)
        states = [
            reset_electron(rho, model_on.de, model_on.dn, config.electron_reset_fidelity)
            for rho in states
        ]

        evolve_all(L_off, 0.5 * config.syndrome_gate_us)
        for i in range(6):
            states[i] = apply_full_unitary_at_time(states[i], U_syn, model_on, current_time)
        evolve_all(L_off, 0.5 * config.syndrome_gate_us)

        for i in range(6):
            states[i] = dephase_electron_measurement(states[i], model_on.de, model_on.dn)
            states[i] = apply_syndrome_assignment_error(
                states[i], config.syndrome_assignment_error, model_on.de, model_on.dn
            )
        evolve_all(L_off, config.measurement_us + config.feedback_latency_us)

        evolve_all(L_off, 0.5 * config.correction_gate_us)
        for i in range(6):
            states[i] = apply_full_unitary_at_time(states[i], U_corr, model_on, current_time)
        evolve_all(L_off, 0.5 * config.correction_gate_us)

        states = [
            reset_electron(rho, model_on.de, model_on.dn, config.electron_reset_fidelity)
            for rho in states
        ]
        for i in range(6):
            states[i] = depolarize_nuclear_after_reset(
                states[i], config.nuclear_depolarizing_probability, model_on.de, model_on.dn
            )

        times.append(current_time)
        rows.append(_benchmark_outputs(states, targets, model_on, code, current_time))

    result = {key: np.asarray([row[key] for row in rows], dtype=float) for key in rows[0]}
    result.update(
        {
            "mode": "pulsed",
            "t_us": np.asarray(times, dtype=float),
            "recovery_count": np.arange(config.cycles + 1, dtype=int),
            "config": config,
            "rates": model_on.rates,
        }
    )
    return result
