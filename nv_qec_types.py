"""Shared data types, code construction, and qubit benchmark helpers."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.linalg as la

# User-supplied register. ``minus_A_parallel_kHz`` is the JSON/DQP convention
# a_JSON = -A_parallel used by the original repository.
DEFAULT_DQP_SPINS = [
    {"name": "S1", "minus_A_parallel_kHz": 5.616, "A_perp_kHz": 32.847},
    {"name": "S2", "minus_A_parallel_kHz": 224.217, "A_perp_kHz": 189.318},
    {"name": "S3", "minus_A_parallel_kHz": 49.837, "A_perp_kHz": 101.007},
    {"name": "S4", "minus_A_parallel_kHz": 15.734, "A_perp_kHz": 19.295},
    {"name": "S5", "minus_A_parallel_kHz": 3.832, "A_perp_kHz": 21.140},
    {"name": "S6", "minus_A_parallel_kHz": -7.015, "A_perp_kHz": 38.250},
    {"name": "X2", "minus_A_parallel_kHz": 33.962, "A_perp_kHz": 26.000},
]

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


@dataclass(frozen=True)
class DQPSpin:
    """One nuclear spin in the ``(-A_parallel, A_perp)`` convention."""

    index: int
    name: str
    minus_A_par_kHz: float
    A_perp_kHz: float

    @property
    def A_par_MHz(self) -> float:
        return -self.minus_A_par_kHz / 1000.0

    @property
    def A_perp_MHz(self) -> float:
        return self.A_perp_kHz / 1000.0

    @property
    def g_MHz(self) -> float:
        """Signed coefficient used to construct the efficient n=2 code."""
        return self.minus_A_par_kHz / 1000.0


@dataclass
class QECCode:
    g_MHz: tuple[float, float]
    recovery_kraus: tuple[np.ndarray, np.ndarray]
    encoding_unitary: np.ndarray
    decoding_unitary: np.ndarray
    logical_isometry: np.ndarray
    P_L: np.ndarray
    P_E: np.ndarray
    U_x: np.ndarray
    zero_L: np.ndarray
    one_L: np.ndarray
    zero_E: np.ndarray
    one_E: np.ndarray


@dataclass(frozen=True)
class PulsedSequenceConfig:
    """Timing and imperfection parameters for the explicit pulsed model."""

    cycles: int = 2
    laser_on_us: float = 8.0
    laser_settle_us: float = 0.25
    syndrome_gate_us: float = 0.75
    measurement_us: float = 0.50
    feedback_latency_us: float = 0.50
    correction_gate_us: float = 0.75
    electron_reset_fidelity: float = 1.0
    syndrome_assignment_error: float = 0.0
    nuclear_depolarizing_probability: float = 0.0

    def validate(self) -> None:
        if self.cycles < 1:
            raise ValueError("cycles must be at least one")
        for field_name in (
            "laser_on_us",
            "laser_settle_us",
            "syndrome_gate_us",
            "measurement_us",
            "feedback_latency_us",
            "correction_gate_us",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be nonnegative")
        for field_name in (
            "electron_reset_fidelity",
            "syndrome_assignment_error",
            "nuclear_depolarizing_probability",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must lie in [0, 1]")


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------


def ket(dim: int, index: int) -> np.ndarray:
    out = np.zeros(dim, dtype=complex)
    out[index] = 1.0
    return out


def transition(dim: int, final: int, initial: int) -> np.ndarray:
    return np.outer(ket(dim, final), ket(dim, initial).conj())


def density(state: np.ndarray) -> np.ndarray:
    return np.outer(state, state.conj())


def vec(rho: np.ndarray) -> np.ndarray:
    return np.asarray(rho, dtype=complex).reshape(-1, order="F")


def mat(vector: np.ndarray, dim: int) -> np.ndarray:
    return np.asarray(vector, dtype=complex).reshape((dim, dim), order="F")


def hermitize(rho: np.ndarray) -> np.ndarray:
    return 0.5 * (rho + rho.conj().T)


def partial_trace_electron(rho: np.ndarray, de: int, dn: int) -> np.ndarray:
    return np.trace(rho.reshape(de, dn, de, dn), axis1=0, axis2=2)


def partial_trace_nuclear(rho: np.ndarray, de: int, dn: int) -> np.ndarray:
    return np.trace(rho.reshape(de, dn, de, dn), axis1=1, axis2=3)


def partial_trace_two_qubit(rho: np.ndarray, target: int) -> np.ndarray:
    reshaped = rho.reshape(2, 2, 2, 2)
    if target == 0:
        return np.trace(reshaped, axis1=1, axis2=3)
    if target == 1:
        return np.trace(reshaped, axis1=0, axis2=2)
    raise ValueError("target must be 0 or 1")


def bell_state_psi_plus() -> np.ndarray:
    return (np.kron(ket(2, 0), ket(2, 1)) + np.kron(ket(2, 1), ket(2, 0))) / np.sqrt(2)


def bell_fidelity(rho_n: np.ndarray, bell: np.ndarray | None = None) -> float:
    state = bell_state_psi_plus() if bell is None else bell
    return float(np.real(state.conj() @ rho_n @ state))


def memory_test_states() -> tuple[list[str], list[np.ndarray]]:
    """The six Pauli eigenstates, a qubit projective 2-design."""
    zero, one = ket(2, 0), ket(2, 1)
    return (
        ["+Z", "-Z", "+X", "-X", "+Y", "-Y"],
        [
            zero,
            one,
            (zero + one) / np.sqrt(2),
            (zero - one) / np.sqrt(2),
            (zero + 1j * one) / np.sqrt(2),
            (zero - 1j * one) / np.sqrt(2),
        ],
    )


def rz(phi: float) -> np.ndarray:
    return la.expm(-0.5j * float(phi) * Z2)


def state_fidelity(target: np.ndarray, rho: np.ndarray) -> float:
    return float(np.real(target.conj() @ rho @ target))


def phase_calibrated_six_state_fidelity(
    outputs: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    phase_samples: int = 181,
) -> tuple[float, float, float]:
    """Return raw and best virtual-Z-corrected six-state mean fidelities.

    The final Z phase is calibrated separately for each memory implementation,
    matching the virtual-Z correction normally available in an experiment.
    No renormalization is applied, so leakage counts as failure.
    """
    raw = float(np.mean([state_fidelity(t, r) for t, r in zip(targets, outputs)]))
    best = raw
    best_phi = 0.0
    for phi in np.linspace(-np.pi, np.pi, phase_samples, endpoint=True):
        U = rz(phi)
        value = float(
            np.mean([state_fidelity(t, U @ r @ U.conj().T) for t, r in zip(targets, outputs)])
        )
        if value > best:
            best = value
            best_phi = float(phi)
    return raw, best, best_phi


# ---------------------------------------------------------------------------
# Spin loading and code construction
# ---------------------------------------------------------------------------


def _parse_spin_row(index: int, row: object) -> DQPSpin:
    if isinstance(row, dict):
        name = str(row.get("name", f"S{index}"))
        minus = row.get("minus_A_parallel_kHz", row.get("minus_A_par_kHz", row.get("a_JSON")))
        perp = row.get("A_perp_kHz", row.get("A_perp"))
        if minus is None or perp is None:
            raise ValueError(f"Spin row {index} lacks longitudinal or transverse coupling")
        return DQPSpin(index, name, float(minus), float(perp))
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        return DQPSpin(index, f"S{index}", float(row[0]), float(row[1]))
    raise ValueError(f"Unsupported spin row {index}: {row!r}")


def load_dqp_spins(path: str | Path | None = None) -> list[DQPSpin]:
    if path is None or not Path(path).exists():
        raw = DEFAULT_DQP_SPINS
    else:
        text = re.sub(r",\s*([\]\}])", r"\1", Path(path).read_text(encoding="utf-8"))
        raw = json.loads(text)
    return [_parse_spin_row(i, row) for i, row in enumerate(raw, start=1)]


def choose_two_spins(
    spins: Sequence[DQPSpin],
    strategy: str = "strongest",
    spin_indices: tuple[int, int] | None = None,
    spin_names: tuple[str, str] | None = None,
) -> list[DQPSpin]:
    if spin_indices is not None and spin_names is not None:
        raise ValueError("choose either spin_indices or spin_names, not both")
    if spin_indices is not None:
        by_index = {s.index: s for s in spins}
        selected = [by_index[i] for i in spin_indices]
    elif spin_names is not None:
        by_name = {s.name.lower(): s for s in spins}
        missing = [name for name in spin_names if name.lower() not in by_name]
        if missing:
            raise ValueError(f"Unknown spin names: {missing}")
        selected = [by_name[name.lower()] for name in spin_names]
    elif strategy == "strongest":
        selected = sorted(spins, key=lambda s: abs(s.minus_A_par_kHz), reverse=True)[:2]
    elif strategy == "closest-positive":
        pairs: list[tuple[float, float, DQPSpin, DQPSpin]] = []
        for i, a in enumerate(spins):
            for b in spins[i + 1 :]:
                if a.minus_A_par_kHz > 0 and b.minus_A_par_kHz > 0:
                    mismatch = abs(a.minus_A_par_kHz - b.minus_A_par_kHz) / max(
                        abs(a.minus_A_par_kHz), abs(b.minus_A_par_kHz)
                    )
                    strength = 0.5 * (abs(a.minus_A_par_kHz) + abs(b.minus_A_par_kHz))
                    pairs.append((mismatch, -strength, a, b))
        if not pairs:
            raise ValueError("No positive same-sign pair found")
        _, _, a, b = min(pairs, key=lambda row: (row[0], row[1]))
        selected = [a, b]
    else:
        raise ValueError(f"Unknown spin-selection strategy: {strategy}")
    if selected[0].name == selected[1].name:
        raise ValueError("Select two distinct spins")
    return sorted(selected, key=lambda s: abs(s.minus_A_par_kHz), reverse=True)


def make_two_qubit_qec_code(g_MHz: tuple[float, float]) -> QECCode:
    """Construct the exact n=2 efficient common-fluctuator code and recovery."""
    g = np.asarray(g_MHz, dtype=float)
    if abs(g[0]) < abs(g[1]) - 1e-15:
        raise ValueError("Order spins so |g1| >= |g2|")
    if g[0] < 0:
        g = -g
    g1, g2 = float(g[0]), float(g[1])
    if g1 <= 0 or abs(g2) > g1 + 1e-14:
        raise ValueError("The n=2 construction requires g1 > 0 and |g2| <= g1")

    a = math.sqrt(max(0.0, (g1 - g2) / (2.0 * g1)))
    b = math.sqrt(max(0.0, (g1 + g2) / (2.0 * g1)))
    chi0 = np.array([a, b], dtype=complex)
    chi1 = np.array([-b, a], dtype=complex)
    zero, one = ket(2, 0), ket(2, 1)

    zero_L = np.kron(chi0, zero)
    one_L = np.kron(chi1, one)
    zero_E = np.kron(chi1, zero)
    one_E = np.kron(chi0, one)
    P_L = density(zero_L) + density(one_L)
    P_E = density(zero_E) + density(one_E)
    U_x = np.outer(chi0, chi1.conj()) + np.outer(chi1, chi0.conj())
    K_no_error = P_L
    K_error = np.kron(U_x, I2) @ P_E
    U_enc = np.column_stack([zero_E, zero_L, one_L, one_E])
    W = np.column_stack([zero_L, one_L])
    return QECCode(
        (g1, g2),
        (K_no_error, K_error),
        U_enc,
        U_enc.conj().T,
        W,
        P_L,
        P_E,
        U_x,
        zero_L,
        one_L,
        zero_E,
        one_E,
    )


def model_params_from_spins(
    spins: Sequence[DQPSpin],
    B_T: float,
    pump_rate: float,
    excited_state_hyperfine_scale: float = 0.1,
) -> dict:
    """Build model parameters for one pair.

    ``excited_state_hyperfine_scale`` sets the excited ``m_s=-1`` tensor to
    ``r`` times the ground ``m_s=-1`` tensor for both nuclei.  ``r=0`` recovers
    the former free-precession excited-state model.
    """
    return {
        "B_T": float(B_T),
        "A_par_MHz": tuple(s.A_par_MHz for s in spins),
        "A_perp_MHz": tuple(s.A_perp_MHz for s in spins),
        "W_pump_per_us": float(pump_rate),
        "excited_state_hyperfine_scale": float(excited_state_hyperfine_scale),
    }
