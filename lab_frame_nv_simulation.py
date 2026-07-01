"""Lab-frame NV-center + two-13C simulation utilities.

This script keeps the common 13C Larmor Hamiltonian in the lab frame,

    H_g0 = H_e0 = H_e1 = H_s = omega_L (I_z1 + I_z2),

and uses the conditional ground-state ms=1 Hamiltonian

    H_g1 = sum_j [(omega_L - A_parallel,j) I_z,j + A_perp,j I_x,j].

The old static interaction-picture shortcut is not used here.  If a QEC
recovery is plotted, the default is to apply/score the recovery in the
co-rotating Larmor frame, i.e. K_lab(t) = R(t) K R(t)^dagger with
R(t)=exp[-i omega_L(I_z1+I_z2)t].  This avoids treating ordinary common
nuclear Larmor precession as a QEC error.

Examples:
    python lab_frame_nv_simulation.py --mode qec --spin-indices 1 2
    python lab_frame_nv_simulation.py --mode hyperfine --spin-indices 2 3 --time-us 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


DEFAULT_DQP_SPINS = [
    [7.880658979484622, 35.18660952634182],
    [224.2991967686721, 199.07207268788648],
    [50.10705093918639, 108.7263051655999],
    [17.11980460341208, 19.548809690807488],
    [5.928129614204922, 21.140094092827646],
    [-4.919015344947458, 38.25005007258501],
]


@dataclass(frozen=True)
class DQPSpin:
    """One DQP spin row stored as (-A_parallel, A_perp) in kHz."""

    index: int
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
        return self.minus_A_par_kHz / 1000.0


@dataclass
class LabFrameModel:
    H: np.ndarray
    collapse_ops: list[np.ndarray]
    rates: dict
    dims: tuple[int, int]
    H_larmor_nuclear: np.ndarray


@dataclass
class QECCode:
    g_MHz: tuple[float, float]
    recovery_kraus: tuple[np.ndarray, np.ndarray]
    encoding_unitary: np.ndarray
    decoding_unitary: np.ndarray


def ket(dim: int, i: int) -> np.ndarray:
    v = np.zeros(dim, dtype=complex)
    v[i] = 1.0
    return v


def transition(dim: int, final: int, initial: int) -> np.ndarray:
    return np.outer(ket(dim, final), ket(dim, initial).conj())


def density(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, psi.conj())


def vec(rho: np.ndarray) -> np.ndarray:
    return np.asarray(rho).reshape(-1, order="F")


def mat(v: np.ndarray, dim: int) -> np.ndarray:
    return np.asarray(v).reshape((dim, dim), order="F")


def hermitize(rho: np.ndarray) -> np.ndarray:
    return 0.5 * (rho + rho.conj().T)


def partial_trace_electron(rho: np.ndarray, de: int, dn: int) -> np.ndarray:
    return np.trace(rho.reshape(de, dn, de, dn), axis1=0, axis2=2)


def bell_state_psi_plus() -> np.ndarray:
    up, down = ket(2, 0), ket(2, 1)
    return (np.kron(up, down) + np.kron(down, up)) / np.sqrt(2)


def bell_fidelity(rho_n: np.ndarray, bell: np.ndarray | None = None) -> float:
    bell = bell_state_psi_plus() if bell is None else bell
    fid = np.real(bell.conj() @ rho_n @ bell)
    return float(np.clip(fid, -1e-12, 1.0 + 1e-12))


def lindblad_liouvillian(H: np.ndarray, collapse_ops: list[np.ndarray]) -> sp.csr_matrix:
    """Return L such that d vec(rho)/dt = L vec(rho), using column stacking."""

    Hs = sp.csr_matrix(H)
    d = Hs.shape[0]
    I = sp.identity(d, format="csr", dtype=complex)
    L = -1j * (sp.kron(I, Hs, format="csr") - sp.kron(Hs.T, I, format="csr"))

    for C in collapse_ops:
        Cs = sp.csr_matrix(C)
        CdC = Cs.getH() @ Cs
        L += sp.kron(Cs.conjugate(), Cs, format="csr")
        L += -0.5 * sp.kron(I, CdC, format="csr")
        L += -0.5 * sp.kron(CdC.T, I, format="csr")
    return L.tocsr()


def evolve_liouvillian(L: sp.csr_matrix, rho0: np.ndarray, tlist_us: np.ndarray) -> np.ndarray:
    tlist_us = np.asarray(tlist_us, dtype=float)
    if len(tlist_us) < 2:
        raise ValueError("tlist_us must contain at least two points")
    return spla.expm_multiply(
        L,
        vec(rho0),
        start=float(tlist_us[0]),
        stop=float(tlist_us[-1]),
        num=len(tlist_us),
        endpoint=True,
    )


_DENSE_PROPAGATOR_CACHE: dict[tuple[int, float], np.ndarray] = {}


def propagate_density(L: sp.csr_matrix, rho: np.ndarray, dt_us: float, dim: int) -> np.ndarray:
    dt = float(dt_us)
    if abs(dt) < 1e-15:
        return hermitize(rho)
    key = (id(L), round(dt, 15))
    U = _DENSE_PROPAGATOR_CACHE.get(key)
    if U is None:
        U = la.expm((L * dt).toarray())
        _DENSE_PROPAGATOR_CACHE[key] = U
    return hermitize(mat(U @ vec(rho), dim))


def make_lab_frame_nv_two_c13_model(
    B_T: float,
    A_par_MHz: tuple[float, float],
    A_perp_MHz: tuple[float, float],
    W_pump_per_us: float,
    T_e0_us: float = 0.013,
    T_e1_us: float = 0.007,
    p_isc0: float = 0.14,
    p_isc1: float = 0.55,
    tau_s_us: float = 0.180,
    beta_s_to_g0: float = 0.85,
) -> LabFrameModel:
    """Build the corrected lab-frame five-electron-level x two-13C model."""

    de, dn = 5, 4
    g0, g1, e0, e1, singlet = range(5)
    I2, In = np.eye(2, dtype=complex), np.eye(dn, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    Ix = (np.kron(sx / 2, I2), np.kron(I2, sx / 2))
    Iz = (np.kron(sz / 2, I2), np.kron(I2, sz / 2))

    gamma_13C_MHz_per_T = 10.705
    omega_L = 2 * np.pi * gamma_13C_MHz_per_T * B_T
    A_par = tuple(2 * np.pi * np.asarray(A_par_MHz, dtype=float))
    A_perp = tuple(2 * np.pi * np.asarray(A_perp_MHz, dtype=float))

    # Lab-frame common Larmor Hamiltonian:
    # H_g0 = H_e0 = H_e1 = H_s = omega_L * I_z1 + omega_L * I_z2.
    H_larmor = omega_L * Iz[0] + omega_L * Iz[1]
    H_g0 = H_larmor
    H_e0 = H_larmor.copy()
    H_e1 = H_larmor.copy()
    H_s = H_larmor.copy()

    H_g1 = sum((omega_L - A_par[j]) * Iz[j] + A_perp[j] * Ix[j] for j in range(2))

    H = np.zeros((de * dn, de * dn), dtype=complex)
    for electron_index, Hn in [(g0, H_g0), (g1, H_g1), (e0, H_e0), (e1, H_e1), (singlet, H_s)]:
        H += np.kron(transition(de, electron_index, electron_index), Hn)

    total_e0 = 1.0 / T_e0_us
    total_e1 = 1.0 / T_e1_us
    rates = {
        "W_pump_per_us": W_pump_per_us,
        "gamma0_rad_per_us": (1.0 - p_isc0) * total_e0,
        "gamma1_rad_per_us": (1.0 - p_isc1) * total_e1,
        "kappa0_isc_per_us": p_isc0 * total_e0,
        "kappa1_isc_per_us": p_isc1 * total_e1,
        "Gamma_s_to_g0_per_us": beta_s_to_g0 / tau_s_us,
        "Gamma_s_to_g1_per_us": (1.0 - beta_s_to_g0) / tau_s_us,
        "omega_L_MHz": omega_L / (2 * np.pi),
    }

    def C(rate: float, final: int, initial: int) -> np.ndarray:
        return np.sqrt(rate) * np.kron(transition(de, final, initial), In)

    collapse_ops = [
        C(rates["W_pump_per_us"], e0, g0),
        C(rates["W_pump_per_us"], e1, g1),
        C(rates["gamma0_rad_per_us"], g0, e0),
        C(rates["gamma1_rad_per_us"], g1, e1),
        C(rates["kappa0_isc_per_us"], singlet, e0),
        C(rates["kappa1_isc_per_us"], singlet, e1),
        C(rates["Gamma_s_to_g0_per_us"], g0, singlet),
        C(rates["Gamma_s_to_g1_per_us"], g1, singlet),
    ]
    return LabFrameModel(H, collapse_ops, rates, (de, dn), H_larmor)


def load_dqp_spins(path: str | Path | None) -> list[DQPSpin]:
    if path is None or not Path(path).exists():
        raw = DEFAULT_DQP_SPINS
    else:
        text = re.sub(r",\s*([\]\}])", r"\1", Path(path).read_text())
        raw = json.loads(text)
    return [DQPSpin(i, float(pair[0]), float(pair[1])) for i, pair in enumerate(raw, start=1)]


def choose_two_spins(spins: list[DQPSpin], spin_indices: tuple[int, int]) -> list[DQPSpin]:
    by_index = {s.index: s for s in spins}
    selected = [by_index[i] for i in spin_indices]
    return sorted(selected, key=lambda s: abs(s.minus_A_par_kHz), reverse=True)


def make_two_qubit_qec_code(g_MHz: tuple[float, float]) -> QECCode:
    g = np.asarray(g_MHz, dtype=float)
    if abs(g[0]) < abs(g[1]) - 1e-15:
        raise ValueError("Order spins so |g1| >= |g2|")
    if g[0] < 0:
        g = -g
    g1, g2 = float(g[0]), float(g[1])
    if g1 <= 0 or g1 + g2 < -1e-14 or g1 - g2 < -1e-14:
        raise ValueError("The n=2 construction requires g1 > 0 and |g2| <= g1 after sign normalization")

    a = math.sqrt(max(0.0, (g1 - g2) / (2 * g1)))
    b = math.sqrt(max(0.0, (g1 + g2) / (2 * g1)))

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
    K_error = np.kron(U_x, np.eye(2, dtype=complex)) @ P_E
    U_enc = np.column_stack([zero_E, zero_L, one_L, one_E])
    return QECCode((g1, g2), (K_no_error, K_error), U_enc, U_enc.conj().T)


def larmor_unitary(H_larmor: np.ndarray, t_us: float) -> np.ndarray:
    return la.expm(-1j * H_larmor * float(t_us))


def decoded_bell_fidelity(
    rho: np.ndarray,
    code: QECCode,
    de: int,
    dn: int,
    H_larmor: np.ndarray,
    t_us: float,
    qec_tracks_larmor: bool = True,
) -> float:
    rho_n = partial_trace_electron(rho, de, dn)
    if qec_tracks_larmor:
        R = larmor_unitary(H_larmor, t_us)
        rho_n = R.conj().T @ rho_n @ R
    rho_decoded = code.decoding_unitary @ rho_n @ code.decoding_unitary.conj().T
    return bell_fidelity(rho_decoded)


def apply_recovery_to_full_state(
    rho: np.ndarray,
    code: QECCode,
    de: int,
    H_larmor: np.ndarray,
    t_us: float,
    qec_tracks_larmor: bool = True,
) -> np.ndarray:
    out = np.zeros_like(rho)
    I_e = np.eye(de, dtype=complex)
    if qec_tracks_larmor:
        R = larmor_unitary(H_larmor, t_us)
    else:
        R = np.eye(H_larmor.shape[0], dtype=complex)
    for K_n in code.recovery_kraus:
        K_lab = R @ K_n @ R.conj().T
        K = np.kron(I_e, K_lab)
        out += K @ rho @ K.conj().T
    return hermitize(out)


def no_protocol_curve(model: LabFrameModel, tlist_us: np.ndarray) -> np.ndarray:
    L = lindblad_liouvillian(model.H, model.collapse_ops)
    de, dn = model.dims
    d = de * dn
    e0, bell = ket(de, 0), bell_state_psi_plus()
    rho0 = density(np.kron(e0, bell))
    raw_vecs = evolve_liouvillian(L, rho0, tlist_us)
    return np.array([
        bell_fidelity(partial_trace_electron(hermitize(mat(v, d)), de, dn), bell)
        for v in raw_vecs
    ])


def _states_from_segment(L: sp.csr_matrix, rho_start: np.ndarray, local_times_us: np.ndarray, dim: int) -> list[np.ndarray]:
    if len(local_times_us) == 0:
        return []
    local_times_us = np.asarray(local_times_us, dtype=float)
    if len(local_times_us) == 1:
        return [propagate_density(L, rho_start, float(local_times_us[0]), dim)]
    dt = np.diff(local_times_us)
    if np.allclose(dt, dt[0], rtol=1e-10, atol=1e-12):
        vecs = spla.expm_multiply(
            L,
            vec(rho_start),
            start=float(local_times_us[0]),
            stop=float(local_times_us[-1]),
            num=len(local_times_us),
            endpoint=True,
        )
        return [hermitize(mat(v, dim)) for v in vecs]
    return [propagate_density(L, rho_start, float(t), dim) for t in local_times_us]


def periodic_qec_curve(
    model: LabFrameModel,
    code: QECCode,
    tlist_us: np.ndarray,
    recovery_interval_us: float,
    qec_tracks_larmor: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    L = lindblad_liouvillian(model.H, model.collapse_ops)
    de, dn = model.dims
    dim = de * dn
    f_without = no_protocol_curve(model, tlist_us)

    e0, bell = ket(de, 0), bell_state_psi_plus()
    rho_start = density(np.kron(e0, code.encoding_unitary @ bell))
    f_with = np.zeros(len(tlist_us))
    f_with[0] = decoded_bell_fidelity(rho_start, code, de, dn, model.H_larmor_nuclear, 0.0, qec_tracks_larmor)

    seg_start = 0.0
    t_end = float(tlist_us[-1])
    tol = 1e-12
    while seg_start < t_end - tol:
        seg_end = min(seg_start + recovery_interval_us, t_end)
        output_indices = np.where((tlist_us > seg_start + tol) & (tlist_us <= seg_end + tol))[0]
        output_times = tlist_us[output_indices]

        need_recovery = seg_end < t_end + tol and abs(seg_end - (seg_start + recovery_interval_us)) <= 1e-9
        seg_end_is_output = len(output_times) > 0 and abs(output_times[-1] - seg_end) <= 1e-9

        eval_times = list(output_times)
        if need_recovery and not seg_end_is_output:
            eval_times.append(seg_end)
        eval_times = np.array(eval_times, dtype=float)
        states = _states_from_segment(L, rho_start, eval_times - seg_start, dim) if len(eval_times) else []
        state_by_time = {round(float(t), 12): rho for t, rho in zip(eval_times, states)}

        for idx, t_us in zip(output_indices, output_times):
            f_with[idx] = decoded_bell_fidelity(
                state_by_time[round(float(t_us), 12)],
                code,
                de,
                dn,
                model.H_larmor_nuclear,
                float(t_us),
                qec_tracks_larmor,
            )

        if need_recovery:
            rho_end = state_by_time.get(round(float(seg_end), 12))
            if rho_end is None:
                rho_end = propagate_density(L, rho_start, seg_end - seg_start, dim)
            rho_start = apply_recovery_to_full_state(
                rho_end,
                code,
                de,
                model.H_larmor_nuclear,
                seg_end,
                qec_tracks_larmor,
            )
            boundary_matches = np.where(np.abs(tlist_us - seg_end) <= 1e-9)[0]
            if len(boundary_matches):
                idx = int(boundary_matches[0])
                f_with[idx] = decoded_bell_fidelity(
                    rho_start,
                    code,
                    de,
                    dn,
                    model.H_larmor_nuclear,
                    seg_end,
                    qec_tracks_larmor,
                )
        else:
            rho_start = states[-1] if states else rho_start
        seg_start = seg_end

    return f_without, f_with


def spin_text(spins: list[DQPSpin]) -> str:
    return "\n".join(
        f"#{s.index} (-A_par={s.minus_A_par_kHz:.3g} kHz, A_perp={s.A_perp_kHz:.3g} kHz)"
        for s in spins
    )


def plot_qec(args: argparse.Namespace) -> None:
    spins = choose_two_spins(load_dqp_spins(args.dqp_file), tuple(args.spin_indices))
    model = make_lab_frame_nv_two_c13_model(
        B_T=args.B_T,
        A_par_MHz=tuple(s.A_par_MHz for s in spins),
        A_perp_MHz=tuple(s.A_perp_MHz for s in spins),
        W_pump_per_us=args.pump_rate,
    )
    code = make_two_qubit_qec_code(tuple(s.g_MHz for s in spins))
    tlist = np.arange(0.0, args.time_us + 0.5 * args.plot_dt_us, args.plot_dt_us)
    f_without, f_with = periodic_qec_curve(
        model,
        code,
        tlist,
        args.recovery_interval_us,
        qec_tracks_larmor=not args.fixed_lab_qec,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"lab_frame_qec_spins_{args.spin_indices[0]}_{args.spin_indices[1]}"

    fig = plt.figure(figsize=(8.2, 5.0))
    ax = fig.add_subplot(111)
    ax.plot(tlist, f_with, linewidth=2.4, label="with QEC")
    ax.plot(tlist, f_without, linewidth=2.4, label="without QEC")
    ax.set_title(f"DQP spins #{args.spin_indices[0]} and #{args.spin_indices[1]}\nQEC recovery every {args.recovery_interval_us:g} us")
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Bell-state fidelity")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.text(0.03, 0.06, spin_text(spins), transform=ax.transAxes, fontsize=9.5, bbox={"boxstyle": "round", "alpha": 0.15})
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)

    with stem.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_us", "with_qec", "without_qec"])
        for row in zip(tlist, f_with, f_without):
            w.writerow([f"{row[0]:.12g}", f"{row[1]:.12g}", f"{row[2]:.12g}"])

    print(f"Saved {stem}.png, {stem}.svg, {stem}.csv")


def plot_hyperfine(args: argparse.Namespace) -> None:
    spins = choose_two_spins(load_dqp_spins(args.dqp_file), tuple(args.spin_indices))
    tlist = np.arange(0.0, args.time_us + 0.5 * args.plot_dt_us, args.plot_dt_us)

    base = dict(
        B_T=args.B_T,
        A_par_MHz=tuple(s.A_par_MHz for s in spins),
        A_perp_MHz=tuple(s.A_perp_MHz for s in spins),
        W_pump_per_us=args.pump_rate,
    )
    held = dict(base)
    held["W_pump_per_us"] = 0.0
    apar_only = dict(base)
    apar_only["A_perp_MHz"] = (0.0, 0.0)

    y_held = no_protocol_curve(make_lab_frame_nv_two_c13_model(**held), tlist)
    y_full = no_protocol_curve(make_lab_frame_nv_two_c13_model(**base), tlist)
    y_apar = no_protocol_curve(make_lab_frame_nv_two_c13_model(**apar_only), tlist)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"lab_frame_hyperfine_spins_{args.spin_indices[0]}_{args.spin_indices[1]}"

    fig = plt.figure(figsize=(10.5, 6.3))
    ax = fig.add_subplot(111)
    ax.plot(tlist, y_held, linewidth=2.0, label=r"electron held in $|g,m_s=0\rangle$ $(W=0)$")
    ax.plot(tlist, y_full, linewidth=2.0, label=r"continuous laser excitation, full $A_\parallel$ and $A_\perp$")
    ax.plot(tlist, y_apar, linewidth=2.0, label=r"continuous laser excitation, $A_\parallel$ only $(A_\perp=0)$")
    ax.set_title("Bell-state fidelity")
    ax.set_xlabel("Evolution time (us)")
    ax.set_ylabel(r"Bell-state fidelity $F(t)=\langle\Psi^+|\rho_C(t)|\Psi^+\rangle$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)

    with stem.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_us", "electron_held_g0_W0", "continuous_laser_full_Apar_Aperp", "continuous_laser_Apar_only"])
        for row in zip(tlist, y_held, y_full, y_apar):
            w.writerow([f"{row[0]:.12g}", f"{row[1]:.12g}", f"{row[2]:.12g}", f"{row[3]:.12g}"])

    print(f"Saved {stem}.png, {stem}.svg, {stem}.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["qec", "hyperfine"], default="qec")
    p.add_argument("--dqp-file", default="dqpspins.json")
    p.add_argument("--spin-indices", nargs=2, type=int, default=(1, 2), metavar=("I", "J"))
    p.add_argument("--time-us", type=float, default=20.0)
    p.add_argument("--plot-dt-us", type=float, default=0.5)
    p.add_argument("--recovery-interval-us", type=float, default=10.0)
    p.add_argument("--B-T", type=float, default=0.05)
    p.add_argument("--pump-rate", type=float, default=5.0)
    p.add_argument("--fixed-lab-qec", action="store_true", help="Do not rotate QEC operations with the common Larmor frame.")
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "qec":
        plot_qec(args)
    else:
        plot_hyperfine(args)


if __name__ == "__main__":
    main()
