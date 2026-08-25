"""The ideal two-spin common-fluctuator code, recovery, and memory metric."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .model import NVModel, hermitian_part, rotating_nuclear_operator_to_lab
from .spins import SpinParameters


_I2 = np.eye(2, dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_SWAP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=complex,
)


@dataclass(frozen=True)
class TwoSpinCode:
    """Code objects expressed in the user's original two-spin tensor order."""

    logical_isometry: np.ndarray
    logical_projector: np.ndarray
    error_projector: np.ndarray
    recovery_kraus: tuple[np.ndarray, np.ndarray]
    logical_basis: tuple[np.ndarray, np.ndarray]
    error_basis: tuple[np.ndarray, np.ndarray]
    ordered_spin_ids: tuple[int, int]
    design_couplings_khz: tuple[float, float]


@dataclass(frozen=True)
class SixStateFidelity:
    """Raw and optimally virtual-Z-corrected six-state average fidelity."""

    raw: float
    phase_corrected: float
    correction_phase_rad: float


def _basis_state(index: int) -> np.ndarray:
    state = np.zeros(2, dtype=complex)
    state[index] = 1.0
    return state


def density_matrix(state: np.ndarray) -> np.ndarray:
    return np.outer(state, state.conj())


def code_design_coupling_khz(spin: SpinParameters) -> float:
    """Return the simplified-model assumption g_j = -A_parallel,j."""

    return -spin.a_parallel_khz


def build_common_fluctuator_code(
    spin_a: SpinParameters,
    spin_b: SpinParameters,
) -> TwoSpinCode:
    """Construct the exact n=2 Layden-Chen-Cappellaro code and recovery."""

    if spin_a.id == spin_b.id:
        raise ValueError("spin ids must be distinct")

    spins = (spin_a, spin_b)
    user_couplings = tuple(code_design_coupling_khz(spin) for spin in spins)
    order = tuple(sorted(range(2), key=lambda index: abs(user_couplings[index]), reverse=True))
    ordered_spins = tuple(spins[index] for index in order)
    couplings = np.array([user_couplings[index] for index in order], dtype=float)
    if abs(couplings[0]) <= 1.0e-15:
        raise ValueError("the common-fluctuator code requires a nonzero longitudinal coupling")
    if couplings[0] < 0.0:
        couplings *= -1.0

    g1, g2 = (float(value) for value in couplings)
    if abs(g2) > g1 + 1.0e-12:
        raise RuntimeError("internal coupling order does not satisfy |g1| >= |g2|")

    coefficient_a = math.sqrt(max(0.0, (g1 - g2) / (2.0 * g1)))
    coefficient_b = math.sqrt(max(0.0, (g1 + g2) / (2.0 * g1)))
    chi0 = np.array([coefficient_a, coefficient_b], dtype=complex)
    chi1 = np.array([-coefficient_b, coefficient_a], dtype=complex)
    zero = _basis_state(0)
    one = _basis_state(1)

    ordered_logical_basis = (
        np.kron(chi0, zero),
        np.kron(chi1, one),
    )
    ordered_error_basis = (
        np.kron(chi1, zero),
        np.kron(chi0, one),
    )
    reorder_to_user = np.eye(4, dtype=complex) if order == (0, 1) else _SWAP
    logical_basis = tuple(reorder_to_user @ state for state in ordered_logical_basis)
    error_basis = tuple(reorder_to_user @ state for state in ordered_error_basis)

    logical_projector = sum(density_matrix(state) for state in logical_basis)
    error_projector = sum(density_matrix(state) for state in error_basis)
    logical_isometry = np.column_stack(logical_basis)
    flip_chi = np.outer(chi0, chi1.conj()) + np.outer(chi1, chi0.conj())
    correction_ordered = np.kron(flip_chi, _I2)
    correction_user = reorder_to_user @ correction_ordered @ reorder_to_user.conj().T
    recovery_kraus = (
        logical_projector,
        correction_user @ error_projector,
    )
    return TwoSpinCode(
        logical_isometry=logical_isometry,
        logical_projector=logical_projector,
        error_projector=error_projector,
        recovery_kraus=recovery_kraus,
        logical_basis=logical_basis,
        error_basis=error_basis,
        ordered_spin_ids=(ordered_spins[0].id, ordered_spins[1].id),
        design_couplings_khz=(g1, g2),
    )


def common_fluctuator_error_hamiltonian(
    spin_a: SpinParameters,
    spin_b: SpinParameters,
) -> np.ndarray:
    """Return H_E = g_a Z_a + g_b Z_b in user order, with g=-A_parallel."""

    g_a = code_design_coupling_khz(spin_a)
    g_b = code_design_coupling_khz(spin_b)
    return g_a * np.kron(_Z, _I2) + g_b * np.kron(_I2, _Z)


def encode_logical_density(logical_state: np.ndarray, code: TwoSpinCode) -> np.ndarray:
    return hermitian_part(
        code.logical_isometry @ logical_state @ code.logical_isometry.conj().T
    )


def decode_logical_density(nuclear_state: np.ndarray, code: TwoSpinCode) -> np.ndarray:
    """Project through the logical isometry without renormalizing leakage."""

    return hermitian_part(
        code.logical_isometry.conj().T @ nuclear_state @ code.logical_isometry
    )


def apply_recovery(nuclear_state: np.ndarray, code: TwoSpinCode) -> np.ndarray:
    recovered = np.zeros_like(nuclear_state, dtype=complex)
    for kraus in code.recovery_kraus:
        recovered += kraus @ nuclear_state @ kraus.conj().T
    return hermitian_part(recovered)


def apply_recovery_in_lab_frame(
    full_state_lab: np.ndarray,
    code: TwoSpinCode,
    model: NVModel,
    absolute_time_us: float,
) -> np.ndarray:
    """Apply ideal instantaneous recovery using R(t) K R(t)† at absolute time t."""

    recovered = np.zeros_like(full_state_lab, dtype=complex)
    electron_identity = np.eye(model.electron_dimension, dtype=complex)
    for kraus_rotating in code.recovery_kraus:
        kraus_lab = rotating_nuclear_operator_to_lab(
            kraus_rotating, model, absolute_time_us
        )
        full_kraus = np.kron(electron_identity, kraus_lab)
        recovered += full_kraus @ full_state_lab @ full_kraus.conj().T
    return hermitian_part(recovered)


def six_state_test_states() -> tuple[np.ndarray, ...]:
    """Return the +Z, -Z, +X, -X, +Y, and -Y qubit states."""

    zero = _basis_state(0)
    one = _basis_state(1)
    return (
        zero,
        one,
        (zero + one) / math.sqrt(2.0),
        (zero - one) / math.sqrt(2.0),
        (zero + 1j * one) / math.sqrt(2.0),
        (zero - 1j * one) / math.sqrt(2.0),
    )


def _mean_target_overlap(
    outputs: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    phase_rad: float,
) -> float:
    virtual_z = np.diag(
        [np.exp(-0.5j * phase_rad), np.exp(0.5j * phase_rad)]
    )
    overlaps = []
    for output, target in zip(outputs, targets, strict=True):
        corrected = virtual_z @ output @ virtual_z.conj().T
        overlaps.append(float(np.real(target.conj() @ corrected @ target)))
    return float(np.mean(overlaps))


def _clean_probability(value: float) -> float:
    if -1.0e-12 < value < 0.0:
        return 0.0
    if 1.0 < value < 1.0 + 1.0e-12:
        return 1.0
    return value


def six_state_average_memory_fidelity(
    outputs: Sequence[np.ndarray],
    targets: Sequence[np.ndarray] | None = None,
) -> SixStateFidelity:
    """Evaluate F6 and its exact optimal final virtual-Z correction.

    The phase objective is a single sinusoid. Evaluating it at 0, π/2, and π
    determines its continuous optimum analytically. Outputs are never normalized,
    so logical leakage remains a fidelity loss.
    """

    target_states = six_state_test_states() if targets is None else tuple(targets)
    if len(outputs) != 6 or len(target_states) != 6:
        raise ValueError("six-state fidelity requires exactly six outputs and targets")

    raw = _mean_target_overlap(outputs, target_states, 0.0)
    at_pi = _mean_target_overlap(outputs, target_states, math.pi)
    at_half_pi = _mean_target_overlap(outputs, target_states, math.pi / 2.0)
    offset = 0.5 * (raw + at_pi)
    cosine_coefficient = 0.5 * (raw - at_pi)
    sine_coefficient = at_half_pi - offset
    optimum = offset + math.hypot(cosine_coefficient, sine_coefficient)
    phase = math.atan2(sine_coefficient, cosine_coefficient)
    if raw > optimum:
        optimum = raw
        phase = 0.0
    return SixStateFidelity(
        raw=_clean_probability(raw),
        phase_corrected=_clean_probability(optimum),
        correction_phase_rad=phase,
    )


def infidelity_suppression_factor(
    qec_fidelity: float,
    best_single_fidelity: float,
) -> float | None:
    """Return (1-best_single)/(1-qec), or None for the exact 0/0 case."""

    qec_infidelity = 1.0 - qec_fidelity
    single_infidelity = 1.0 - best_single_fidelity
    if math.isclose(qec_infidelity, 0.0, abs_tol=1.0e-14):
        if math.isclose(single_infidelity, 0.0, abs_tol=1.0e-14):
            return None
        return math.inf
    return single_infidelity / qec_infidelity


__all__ = [
    "SixStateFidelity",
    "TwoSpinCode",
    "apply_recovery",
    "apply_recovery_in_lab_frame",
    "build_common_fluctuator_code",
    "code_design_coupling_khz",
    "common_fluctuator_error_hamiltonian",
    "decode_logical_density",
    "density_matrix",
    "encode_logical_density",
    "infidelity_suppression_factor",
    "six_state_average_memory_fidelity",
    "six_state_test_states",
]
