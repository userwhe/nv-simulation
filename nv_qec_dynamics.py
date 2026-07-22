"""Laboratory-frame NV dynamics and ideal/ancilla recovery operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from nv_qec_types import *


@dataclass
class NVModel:
    H_lab: np.ndarray
    collapse_ops: list[np.ndarray]
    rates: dict[str, float]
    de: int
    dn: int
    H_free_n: np.ndarray

    @property
    def dim(self) -> int:
        return self.de * self.dn


# ---------------------------------------------------------------------------
# Lindblad dynamics and frame handling
# ---------------------------------------------------------------------------


def lindblad_liouvillian(H: np.ndarray, collapse_ops: Sequence[np.ndarray]) -> sp.csr_matrix:
    """Return L for column-stacked density matrices: d vec(rho)/dt = L vec(rho)."""
    Hs = sp.csr_matrix(H)
    dim = Hs.shape[0]
    ident = sp.identity(dim, format="csr", dtype=complex)
    L = -1j * (sp.kron(ident, Hs, format="csr") - sp.kron(Hs.T, ident, format="csr"))
    for collapse in collapse_ops:
        C = sp.csr_matrix(collapse)
        CdC = C.getH() @ C
        L += sp.kron(C.conjugate(), C, format="csr")
        L -= 0.5 * sp.kron(ident, CdC, format="csr")
        L -= 0.5 * sp.kron(CdC.T, ident, format="csr")
    return L.tocsr()


def _states_to_vectors(states: Sequence[np.ndarray]) -> np.ndarray:
    return np.column_stack([vec(rho) for rho in states])


def _vectors_to_states(vectors: np.ndarray, dim: int) -> list[np.ndarray]:
    if vectors.ndim == 1:
        vectors = vectors[:, None]
    return [hermitize(mat(vectors[:, i], dim)) for i in range(vectors.shape[1])]


def propagate_state_batch(
    L: sp.csr_matrix,
    states: Sequence[np.ndarray],
    dt_us: float,
    dim: int,
) -> list[np.ndarray]:
    """Propagate a batch without global caching.

    Removing the old cache keyed by ``id(L)`` makes parameter sweeps independent
    of Python object-id reuse and therefore order-independent.
    """
    if abs(dt_us) < 1e-15:
        return [hermitize(rho) for rho in states]
    vectors = spla.expm_multiply(L * float(dt_us), _states_to_vectors(states))
    return _vectors_to_states(vectors, dim)


def evolve_state_batch_to_times(
    L: sp.csr_matrix,
    states: Sequence[np.ndarray],
    local_times_us: Sequence[float],
    dim: int,
) -> list[list[np.ndarray]]:
    """Evolve the same state batch to each supplied local time."""
    times = np.asarray(local_times_us, dtype=float)
    if len(times) == 0:
        return []
    if np.any(times < -1e-12):
        raise ValueError("local times must be nonnegative")
    if len(times) == 1:
        return [propagate_state_batch(L, states, float(times[0]), dim)]

    diffs = np.diff(times)
    if np.allclose(diffs, diffs[0], rtol=1e-10, atol=1e-12):
        evolved = spla.expm_multiply(
            L,
            _states_to_vectors(states),
            start=float(times[0]),
            stop=float(times[-1]),
            num=len(times),
            endpoint=True,
        )
        return [_vectors_to_states(evolved[k], dim) for k in range(len(times))]
    return [propagate_state_batch(L, states, float(t), dim) for t in times]


def _nuclear_spin_operators() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    Ix = (np.kron(X2 / 2, I2), np.kron(I2, X2 / 2))
    Iz = (np.kron(Z2 / 2, I2), np.kron(I2, Z2 / 2))
    return Ix, Iz


def set_excited_state_hyperfine_from_ground(
    A_par_MHz: Sequence[float],
    A_perp_MHz: Sequence[float],
    scale: float = 0.1,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return an excited-state tensor proportional to the ground-state tensor.

    For the optically excited ``m_s=-1`` manifold this implements

    ``A_parallel,e = scale * A_parallel,g`` and
    ``A_perp,e     = scale * A_perp,g``.

    The default is ``scale=0.1``.  Setting ``scale=0`` recovers the previous
    model in which the excited manifolds had only free nuclear precession.
    A negative finite scale is allowed because the excited-state tensor can, in
    principle, have the opposite sign from the ground-state tensor.
    """
    scale = float(scale)
    if not np.isfinite(scale):
        raise ValueError("excited-state hyperfine scale must be finite")
    A_par = np.asarray(A_par_MHz, dtype=float)
    A_perp = np.asarray(A_perp_MHz, dtype=float)
    if A_par.shape != A_perp.shape:
        raise ValueError("A_parallel and A_perp must have matching shapes")
    return tuple((scale * A_par).tolist()), tuple((scale * A_perp).tolist())


def _ms_minus_one_nuclear_hamiltonian(
    omega_L: float,
    A_par_rad_per_us: Sequence[float],
    A_perp_rad_per_us: Sequence[float],
    Ix: Sequence[np.ndarray],
    Iz: Sequence[np.ndarray],
) -> np.ndarray:
    return sum(
        (omega_L - float(A_par_rad_per_us[j])) * Iz[j]
        + float(A_perp_rad_per_us[j]) * Ix[j]
        for j in range(len(Iz))
    )


def build_nv_two_c13_lab_model(
    B_T: float = 0.05,
    A_par_MHz: tuple[float, float] = (-0.049837, -0.033962),
    A_perp_MHz: tuple[float, float] = (0.101007, 0.026),
    W_pump_per_us: float = 5.0,
    T_e0_us: float = 0.013,
    T_e1_us: float = 0.007,
    p_isc0: float = 0.14,
    p_isc1: float = 0.55,
    tau_s_us: float = 0.180,
    beta_s_to_g0: float = 0.85,
    excited_state_hyperfine_scale: float = 0.1,
) -> NVModel:
    """Build the simplified five-electron-level x two-nuclear-spin lab model.

    Electron levels are ``g0, g1, e0, e1, singlet``.  Ground ``g1`` uses the
    supplied ground-state tensor.  Excited ``e1`` uses the same tensor scaled by
    ``excited_state_hyperfine_scale``; ``e0`` and the singlet retain free nuclear
    precession.  The default scale is 0.1.
    """
    de, dn = 5, 4
    g0, g1, e0, e1, singlet = range(de)
    Ix, Iz = _nuclear_spin_operators()
    gamma_13c_MHz_per_T = 10.705
    omega_L = 2 * np.pi * gamma_13c_MHz_per_T * B_T

    A_par_ground_MHz = tuple(float(v) for v in A_par_MHz)
    A_perp_ground_MHz = tuple(float(v) for v in A_perp_MHz)
    A_par_excited_MHz, A_perp_excited_MHz = set_excited_state_hyperfine_from_ground(
        A_par_ground_MHz,
        A_perp_ground_MHz,
        excited_state_hyperfine_scale,
    )

    A_par_ground = tuple(2 * np.pi * np.asarray(A_par_ground_MHz, dtype=float))
    A_perp_ground = tuple(2 * np.pi * np.asarray(A_perp_ground_MHz, dtype=float))
    A_par_excited = tuple(2 * np.pi * np.asarray(A_par_excited_MHz, dtype=float))
    A_perp_excited = tuple(2 * np.pi * np.asarray(A_perp_excited_MHz, dtype=float))

    H_free_n = omega_L * (Iz[0] + Iz[1])
    H_g1 = _ms_minus_one_nuclear_hamiltonian(
        omega_L, A_par_ground, A_perp_ground, Ix, Iz
    )
    H_e1 = _ms_minus_one_nuclear_hamiltonian(
        omega_L, A_par_excited, A_perp_excited, Ix, Iz
    )

    H = np.zeros((de * dn, de * dn), dtype=complex)
    for electron_index, Hn in (
        (g0, H_free_n),
        (g1, H_g1),
        (e0, H_free_n),
        (e1, H_e1),
        (singlet, H_free_n),
    ):
        H += np.kron(transition(de, electron_index, electron_index), Hn)

    total_e0 = 1.0 / T_e0_us
    total_e1 = 1.0 / T_e1_us
    rates = {
        "W_pump_per_us": float(W_pump_per_us),
        "gamma0_rad_per_us": (1.0 - p_isc0) * total_e0,
        "gamma1_rad_per_us": (1.0 - p_isc1) * total_e1,
        "kappa0_isc_per_us": p_isc0 * total_e0,
        "kappa1_isc_per_us": p_isc1 * total_e1,
        "Gamma_s_to_g0_per_us": beta_s_to_g0 / tau_s_us,
        "Gamma_s_to_g1_per_us": (1.0 - beta_s_to_g0) / tau_s_us,
        "omega_L_MHz": omega_L / (2 * np.pi),
        "excited_state_hyperfine_scale": float(excited_state_hyperfine_scale),
    }

    In = np.eye(dn, dtype=complex)

    def collapse(rate: float, final: int, initial: int) -> np.ndarray:
        return np.sqrt(rate) * np.kron(transition(de, final, initial), In)

    collapse_ops = [
        collapse(rates["W_pump_per_us"], e0, g0),
        collapse(rates["W_pump_per_us"], e1, g1),
        collapse(rates["gamma0_rad_per_us"], g0, e0),
        collapse(rates["gamma1_rad_per_us"], g1, e1),
        collapse(rates["kappa0_isc_per_us"], singlet, e0),
        collapse(rates["kappa1_isc_per_us"], singlet, e1),
        collapse(rates["Gamma_s_to_g0_per_us"], g0, singlet),
        collapse(rates["Gamma_s_to_g1_per_us"], g1, singlet),
    ]
    return NVModel(H, collapse_ops, rates, de, dn, H_free_n)


def make_nv_two_c13_model(
    B_T: float = 0.05,
    A_par_MHz: tuple[float, float] = (-0.049837, -0.033962),
    A_perp_MHz: tuple[float, float] = (0.101007, 0.026),
    W_pump_per_us: float = 5.0,
    T_e0_us: float = 0.013,
    T_e1_us: float = 0.007,
    p_isc0: float = 0.14,
    p_isc1: float = 0.55,
    tau_s_us: float = 0.180,
    beta_s_to_g0: float = 0.85,
    excited_state_hyperfine_scale: float = 0.1,
    interaction_picture: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, float], tuple[int, int]]:
    """Backward-compatible model builder with a safe frame rule.

    A static interaction-picture Hamiltonian is valid only when every
    transverse coupling is zero.  For nonzero A_perp, the transverse term must
    rotate at omega_L, so this function raises instead of silently using the
    incorrect static ``H_g1 - H_g0`` approximation.  Production simulations use
    the lab-frame model and rotate operations/readout exactly.
    """
    model = build_nv_two_c13_lab_model(
        B_T=B_T,
        A_par_MHz=A_par_MHz,
        A_perp_MHz=A_perp_MHz,
        W_pump_per_us=W_pump_per_us,
        T_e0_us=T_e0_us,
        T_e1_us=T_e1_us,
        p_isc0=p_isc0,
        p_isc1=p_isc1,
        tau_s_us=tau_s_us,
        beta_s_to_g0=beta_s_to_g0,
        excited_state_hyperfine_scale=excited_state_hyperfine_scale,
    )
    if not interaction_picture:
        return model.H_lab, model.collapse_ops, model.rates, (model.de, model.dn)
    if np.max(np.abs(A_perp_MHz)) > 1e-15:
        raise ValueError(
            "A static interaction-picture Hamiltonian is invalid for nonzero A_perp; "
            "use lab-frame evolution or an explicitly time-dependent rotating-frame model."
        )
    H_rot = model.H_lab - np.kron(np.eye(model.de, dtype=complex), model.H_free_n)
    return H_rot, model.collapse_ops, model.rates, (model.de, model.dn)


def nuclear_free_unitary(model: NVModel, t_us: float) -> np.ndarray:
    return la.expm(-1j * model.H_free_n * float(t_us))


def to_rotating_nuclear_frame(rho_n_lab: np.ndarray, model: NVModel, t_us: float) -> np.ndarray:
    R = nuclear_free_unitary(model, t_us)
    return hermitize(R.conj().T @ rho_n_lab @ R)


def rotating_operator_to_lab(operator_n: np.ndarray, model: NVModel, t_us: float) -> np.ndarray:
    R = nuclear_free_unitary(model, t_us)
    return R @ operator_n @ R.conj().T


# ---------------------------------------------------------------------------
# Recovery channels and explicit electron-ancilla operations
# ---------------------------------------------------------------------------


def apply_recovery_to_full_state(
    rho: np.ndarray,
    code: QECCode,
    de: int,
    model: NVModel | None = None,
    t_us: float = 0.0,
) -> np.ndarray:
    """Apply the ideal nuclear recovery, optionally transformed to the lab frame."""
    out = np.zeros_like(rho)
    I_e = np.eye(de, dtype=complex)
    for K_n in code.recovery_kraus:
        K_use = rotating_operator_to_lab(K_n, model, t_us) if model is not None else K_n
        K = np.kron(I_e, K_use)
        out += K @ rho @ K.conj().T
    return hermitize(out)


def ancilla_recovery_unitaries(code: QECCode, de: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return ideal syndrome-extraction and conditional-correction unitaries."""
    if de < 2:
        raise ValueError("The electron ancilla needs at least two levels")
    P_comp = transition(de, 0, 0) + transition(de, 1, 1)
    X01 = transition(de, 0, 1) + transition(de, 1, 0)
    P_other = np.eye(de, dtype=complex) - P_comp
    I_n = np.eye(4, dtype=complex)
    U_syn = np.kron(P_comp, code.P_L) + np.kron(X01, code.P_E) + np.kron(P_other, I_n)
    U_corr_n = np.kron(code.U_x, I2)
    U_corr = (
        np.kron(transition(de, 0, 0), I_n)
        + np.kron(transition(de, 1, 1), U_corr_n)
        + np.kron(P_other, I_n)
    )
    return U_syn, U_corr


def full_rotating_operator_to_lab(U_rot: np.ndarray, model: NVModel, t_us: float) -> np.ndarray:
    frame = np.kron(np.eye(model.de, dtype=complex), nuclear_free_unitary(model, t_us))
    return frame @ U_rot @ frame.conj().T


def apply_full_unitary_at_time(rho: np.ndarray, U_rot: np.ndarray, model: NVModel, t_us: float) -> np.ndarray:
    U = full_rotating_operator_to_lab(U_rot, model, t_us)
    return hermitize(U @ rho @ U.conj().T)


def reset_electron(
    rho: np.ndarray,
    de: int,
    dn: int,
    g0_fidelity: float = 1.0,
) -> np.ndarray:
    """Trace out the old electron and prepare g0/g1 with stated g0 fidelity."""
    if not 0.0 <= g0_fidelity <= 1.0:
        raise ValueError("g0_fidelity must lie in [0, 1]")
    rho_n = partial_trace_electron(rho, de, dn)
    rho_e = g0_fidelity * density(ket(de, 0)) + (1.0 - g0_fidelity) * density(ket(de, 1))
    return hermitize(np.kron(rho_e, rho_n))


def dephase_electron_measurement(rho: np.ndarray, de: int, dn: int) -> np.ndarray:
    reshaped = rho.reshape(de, dn, de, dn).copy()
    for i in range(de):
        for j in range(de):
            if i != j:
                reshaped[i, :, j, :] = 0.0
    return hermitize(reshaped.reshape(de * dn, de * dn))


def apply_syndrome_assignment_error(
    rho: np.ndarray,
    probability: float,
    de: int,
    dn: int,
) -> np.ndarray:
    if probability <= 0:
        return hermitize(rho)
    X01 = transition(de, 0, 1) + transition(de, 1, 0)
    X01 += np.eye(de, dtype=complex) - transition(de, 0, 0) - transition(de, 1, 1)
    U = np.kron(X01, np.eye(dn, dtype=complex))
    return hermitize((1.0 - probability) * rho + probability * (U @ rho @ U.conj().T))


def depolarize_nuclear_after_reset(
    rho: np.ndarray,
    probability: float,
    de: int,
    dn: int,
) -> np.ndarray:
    if probability <= 0:
        return hermitize(rho)
    rho_e = partial_trace_nuclear(rho, de, dn)
    mixed = np.kron(rho_e, np.eye(dn, dtype=complex) / dn)
    return hermitize((1.0 - probability) * rho + probability * mixed)


def apply_explicit_ancilla_recovery(
    rho: np.ndarray,
    code: QECCode,
    model: NVModel,
    t_us: float,
    reset_fidelity: float = 1.0,
    assignment_error: float = 0.0,
) -> np.ndarray:
    """Zero-duration explicit ancilla realization of the ideal recovery map."""
    U_syn, U_corr = ancilla_recovery_unitaries(code, model.de)
    out = reset_electron(rho, model.de, model.dn, reset_fidelity)
    out = apply_full_unitary_at_time(out, U_syn, model, t_us)
    out = dephase_electron_measurement(out, model.de, model.dn)
    out = apply_syndrome_assignment_error(out, assignment_error, model.de, model.dn)
    out = apply_full_unitary_at_time(out, U_corr, model, t_us)
    return reset_electron(out, model.de, model.dn, reset_fidelity)
