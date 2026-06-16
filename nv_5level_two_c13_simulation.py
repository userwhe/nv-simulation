'''Five-level NV-center plus two 13C Bell-state QEC simulation.

DQP rows are (-A_parallel, A_perp) in kHz.  The script selects two
DQP spins, embeds them in a five-level NV Lindblad model, and overlays
Bell-state fidelity with and without the n=2 common-fluctuator QEC
protocol of Layden, Chen, and Cappellaro.
'''

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    def dephasing_g_MHz(self) -> float:
        return self.minus_A_par_kHz / 1000.0


@dataclass
class QECCode:
    g_MHz: tuple[float, float]
    recovery_kraus: tuple[np.ndarray, np.ndarray]
    encoding_unitary: np.ndarray
    decoding_unitary: np.ndarray
    basis: np.ndarray


def ket(dim: int, i: int) -> np.ndarray:
    v = np.zeros(dim, dtype=complex)
    v[i] = 1.0
    return v


def density(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, psi.conj())


def transition(dim: int, final: int, initial: int) -> np.ndarray:
    return np.outer(ket(dim, final), ket(dim, initial).conj())


def vec(rho: np.ndarray) -> np.ndarray:
    return np.asarray(rho).reshape(-1, order='F')


def mat(v: np.ndarray, dim: int) -> np.ndarray:
    return np.asarray(v).reshape((dim, dim), order='F')


def hermitize(rho: np.ndarray) -> np.ndarray:
    return 0.5 * (rho + rho.conj().T)


def partial_trace_electron(rho: np.ndarray, de: int, dn: int) -> np.ndarray:
    return np.trace(rho.reshape(de, dn, de, dn), axis1=0, axis2=2)


def bell_state_psi_plus() -> np.ndarray:
    up = ket(2, 0)
    down = ket(2, 1)
    return (np.kron(up, down) + np.kron(down, up)) / np.sqrt(2)


def bell_fidelity(rho_n: np.ndarray) -> float:
    bell = bell_state_psi_plus()
    return float(np.real(bell.conj() @ rho_n @ bell))


def lindblad_liouvillian(H: np.ndarray, collapse_ops: list[np.ndarray]) -> sp.csr_matrix:
    Hs = sp.csr_matrix(H)
    d = Hs.shape[0]
    eye = sp.identity(d, format='csr', dtype=complex)
    L = -1j * (sp.kron(eye, Hs, format='csr') - sp.kron(Hs.T, eye, format='csr'))
    for C in collapse_ops:
        Cs = sp.csr_matrix(C)
        CdC = Cs.getH() @ Cs
        L += sp.kron(Cs.conjugate(), Cs, format='csr')
        L += -0.5 * sp.kron(eye, CdC, format='csr')
        L += -0.5 * sp.kron(CdC.T, eye, format='csr')
    return L.tocsr()


def evolve_liouvillian(L: sp.csr_matrix, rho0: np.ndarray, tlist_us: np.ndarray) -> np.ndarray:
    tlist_us = np.asarray(tlist_us, dtype=float)
    dt = np.diff(tlist_us)
    if len(tlist_us) < 2 or not np.allclose(dt, dt[0], rtol=1e-12, atol=1e-15):
        raise ValueError('Use a uniform time grid with at least two points.')
    return spla.expm_multiply(
        L,
        vec(rho0),
        start=float(tlist_us[0]),
        stop=float(tlist_us[-1]),
        num=len(tlist_us),
        endpoint=True,
    )


def make_nv_two_c13_model(
    B_T: float = 0.05,
    A_par_MHz: tuple[float, float] = (-0.2242991967686721, -0.05010705093918639),
    A_perp_MHz: tuple[float, float] = (0.19907207268788648, 0.1087263051655999),
    W_pump_per_us: float = 5.0,
    T_e0_us: float = 0.013,
    T_e1_us: float = 0.007,
    p_isc0: float = 0.14,
    p_isc1: float = 0.55,
    tau_s_us: float = 0.180,
    beta_s_to_g0: float = 0.85,
    interaction_picture: bool = True,
) -> tuple[np.ndarray, list[np.ndarray], dict, tuple[int, int]]:
    de, dn = 5, 4
    g0, g1, e0, e1, singlet = range(5)
    I2 = np.eye(2, dtype=complex)
    In = np.eye(dn, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    Ix = (np.kron(sx / 2, I2), np.kron(I2, sx / 2))
    Iz = (np.kron(sz / 2, I2), np.kron(I2, sz / 2))

    omega_L = 2 * np.pi * 10.705 * B_T
    A_par = tuple(2 * np.pi * np.asarray(A_par_MHz, dtype=float))
    A_perp = tuple(2 * np.pi * np.asarray(A_perp_MHz, dtype=float))
    H_g0_lab = omega_L * (Iz[0] + Iz[1])
    H_g1_lab = sum((omega_L - A_par[j]) * Iz[j] + A_perp[j] * Ix[j] for j in range(2))
    if interaction_picture:
        z = np.zeros((dn, dn), dtype=complex)
        H_g0, H_g1, H_e, H_s = z, H_g1_lab - H_g0_lab, z, z
    else:
        H_g0, H_g1, H_e, H_s = H_g0_lab, H_g1_lab, H_g0_lab.copy(), H_g0_lab.copy()

    H = np.zeros((de * dn, de * dn), dtype=complex)
    for e_idx, Hn in [(g0, H_g0), (g1, H_g1), (e0, H_e), (e1, H_e), (singlet, H_s)]:
        H += np.kron(transition(de, e_idx, e_idx), Hn)

    total_e0 = 1.0 / T_e0_us
    total_e1 = 1.0 / T_e1_us
    rates = {
        'W_pump_per_us': W_pump_per_us,
        'gamma0_rad_per_us': (1.0 - p_isc0) * total_e0,
        'gamma1_rad_per_us': (1.0 - p_isc1) * total_e1,
        'kappa0_isc_per_us': p_isc0 * total_e0,
        'kappa1_isc_per_us': p_isc1 * total_e1,
        'Gamma_s_to_g0_per_us': beta_s_to_g0 / tau_s_us,
        'Gamma_s_to_g1_per_us': (1.0 - beta_s_to_g0) / tau_s_us,
    }

    def C(rate: float, final: int, initial: int) -> np.ndarray:
        return np.sqrt(rate) * np.kron(transition(de, final, initial), In)

    collapse_ops = [
        C(rates['W_pump_per_us'], e0, g0),
        C(rates['W_pump_per_us'], e1, g1),
        C(rates['gamma0_rad_per_us'], g0, e0),
        C(rates['gamma1_rad_per_us'], g1, e1),
        C(rates['kappa0_isc_per_us'], singlet, e0),
        C(rates['kappa1_isc_per_us'], singlet, e1),
        C(rates['Gamma_s_to_g0_per_us'], g0, singlet),
        C(rates['Gamma_s_to_g1_per_us'], g1, singlet),
    ]
    return H, collapse_ops, rates, (de, dn)


def load_dqp_spins(path: str | Path | None) -> list[DQPSpin]:
    if path is None or not Path(path).exists():
        raw = DEFAULT_DQP_SPINS
    else:
        text = Path(path).read_text()
        nl = chr(10)
        text = text.replace(',' + nl + ']', nl + ']').replace(',' + nl + nl + ']', nl + ']')
        raw = json.loads(text)
    return [DQPSpin(i, float(pair[0]), float(pair[1])) for i, pair in enumerate(raw, start=1)]


def choose_two_spins(spins: list[DQPSpin], strategy: str, spin_indices: tuple[int, int] | None) -> list[DQPSpin]:
    if spin_indices is not None:
        by_index = {s.index: s for s in spins}
        selected = [by_index[i] for i in spin_indices]
    elif strategy == 'strongest':
        selected = sorted(spins, key=lambda s: abs(s.minus_A_par_kHz), reverse=True)[:2]
    elif strategy == 'closest-positive':
        pairs = []
        for i, a in enumerate(spins):
            for b in spins[i + 1:]:
                if a.minus_A_par_kHz > 0 and b.minus_A_par_kHz > 0:
                    mismatch = abs(a.minus_A_par_kHz - b.minus_A_par_kHz) / max(abs(a.minus_A_par_kHz), abs(b.minus_A_par_kHz))
                    strength = 0.5 * (abs(a.minus_A_par_kHz) + abs(b.minus_A_par_kHz))
                    pairs.append((mismatch, -strength, a, b))
        if not pairs:
            raise ValueError('No positive same-sign DQP pair found.')
        _, _, a, b = min(pairs, key=lambda row: (row[0], row[1]))
        selected = [a, b]
    else:
        raise ValueError(f'Unknown selection strategy: {strategy}')
    return sorted(selected, key=lambda s: abs(s.minus_A_par_kHz), reverse=True)


def make_two_qubit_qec_code(g_MHz: tuple[float, float]) -> QECCode:
    g = np.asarray(g_MHz, dtype=float)
    if abs(g[0]) < abs(g[1]) - 1e-15:
        raise ValueError('Order spins so |g1| >= |g2|.')
    if g[0] < 0:
        g = -g
    g1, g2 = float(g[0]), float(g[1])
    if g1 <= 0 or g1 + g2 < -1e-14 or g1 - g2 < -1e-14:
        raise ValueError('Need g1 > 0 and |g2| <= g1.')

    a = np.sqrt((g1 - g2) / (2 * g1))
    b = np.sqrt((g1 + g2) / (2 * g1))
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
    K0 = P_L
    K1 = np.kron(U_x, np.eye(2, dtype=complex)) @ P_E
    U_enc = np.column_stack([zero_E, zero_L, one_L, one_E])
    return QECCode((g1, g2), (K0, K1), U_enc, U_enc.conj().T, np.column_stack([zero_L, one_L, zero_E, one_E]))


def model_params_from_spins(spins: list[DQPSpin], B_T: float, pump_rate: float) -> dict:
    return {
        'B_T': B_T,
        'A_par_MHz': tuple(s.A_par_MHz for s in spins),
        'A_perp_MHz': tuple(s.A_perp_MHz for s in spins),
        'W_pump_per_us': pump_rate,
        'interaction_picture': True,
    }


def apply_recovery_to_full_state(rho_full: np.ndarray, code: QECCode, de: int) -> np.ndarray:
    out = np.zeros_like(rho_full)
    I_e = np.eye(de, dtype=complex)
    for K_n in code.recovery_kraus:
        K = np.kron(I_e, K_n)
        out += K @ rho_full @ K.conj().T
    return out


def decoded_bell_fidelity(rho_full: np.ndarray, code: QECCode, de: int, dn: int) -> float:
    rho_n = partial_trace_electron(rho_full, de, dn)
    rho_decoded = code.decoding_unitary @ rho_n @ code.decoding_unitary.conj().T
    return bell_fidelity(rho_decoded)


def simulate_qec_overlay(tlist_us: np.ndarray, model_params: dict, code: QECCode) -> dict:
    tlist_us = np.asarray(tlist_us, dtype=float)
    dt = np.diff(tlist_us)
    if len(tlist_us) < 2 or not np.allclose(dt, dt[0], rtol=1e-12, atol=1e-15):
        raise ValueError('Use a uniform time grid with at least two points.')
    dt_us = float(dt[0])
    H, collapse_ops, rates, (de, dn) = make_nv_two_c13_model(**model_params)
    L = lindblad_liouvillian(H, collapse_ops)
    d = de * dn
    electron_g0 = ket(de, 0)
    bell = bell_state_psi_plus()

    rho0_raw = density(np.kron(electron_g0, bell))
    raw_vecs = evolve_liouvillian(L, rho0_raw, tlist_us)
    f_without = np.zeros(len(tlist_us))
    for n, rho_v in enumerate(raw_vecs):
        rho = hermitize(mat(rho_v, d))
        f_without[n] = bell_fidelity(partial_trace_electron(rho, de, dn))

    encoded_bell = code.encoding_unitary @ bell
    rho_qec = density(np.kron(electron_g0, encoded_bell))
    f_with = np.zeros(len(tlist_us))
    traces = np.zeros(len(tlist_us))
    for n in range(len(tlist_us)):
        rho_qec = hermitize(rho_qec)
        f_with[n] = decoded_bell_fidelity(rho_qec, code, de, dn)
        traces[n] = np.trace(rho_qec).real
        if n < len(tlist_us) - 1:
            rho_next = mat(spla.expm_multiply(L * dt_us, vec(rho_qec)), d)
            rho_qec = apply_recovery_to_full_state(hermitize(rho_next), code, de)
    return {
        't_us': tlist_us,
        'fidelity_without_protocol': f_without,
        'fidelity_with_protocol': f_with,
        'trace_with_protocol': traces,
        'dt_us': dt_us,
        'rates': rates,
    }


def save_csv(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t_us', 'bell_fidelity_without_protocol', 'decoded_bell_fidelity_with_qec'])
        for row in zip(result['t_us'], result['fidelity_without_protocol'], result['fidelity_with_protocol']):
            writer.writerow([f'{row[0]:.12g}', f'{row[1]:.12g}', f'{row[2]:.12g}'])


def plot_overlay(result: dict, selected: list[DQPSpin], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spin_text = ', '.join(f'#{s.index}: -Apar={s.minus_A_par_kHz:.3g} kHz, Aperp={s.A_perp_kHz:.3g} kHz' for s in selected)
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(result['t_us'], result['fidelity_without_protocol'], label='Without protocol: physical |Psi+>')
    ax.plot(result['t_us'], result['fidelity_with_protocol'], label='With QEC protocol: encode + recover + decode')
    ax.set_xlabel('Time (us)')
    ax.set_ylabel('Bell-state fidelity')
    ax.set_title('Two-13C Bell-state fidelity with common-fluctuator QEC')
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    dt_value = result['dt_us']
    note = f'DQP spins {spin_text}' + chr(10) + f'recovery interval = {dt_value:.3g} us'
    ax.text(0.02, 0.04, note, transform=ax.transAxes, fontsize=8.5, va='bottom', bbox={'boxstyle': 'round', 'alpha': 0.12})
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_sanity_checks(code: QECCode, result: dict) -> None:
    orth_err = np.max(np.abs(code.basis.conj().T @ code.basis - np.eye(4)))
    cptp_err = np.max(np.abs(sum(K.conj().T @ K for K in code.recovery_kraus) - np.eye(4)))
    trace_err = np.max(np.abs(result['trace_with_protocol'] - 1.0))
    if orth_err > 1e-12 or cptp_err > 1e-12 or trace_err > 1e-9:
        raise RuntimeError(f'Sanity check failed: orth_err={orth_err}, cptp_err={cptp_err}, trace_err={trace_err}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--dqp-file', default='dqpspins.json')
    p.add_argument('--selection', choices=['strongest', 'closest-positive'], default='strongest')
    p.add_argument('--spin-indices', nargs=2, type=int, metavar=('I', 'J'))
    p.add_argument('--time-us', type=float, default=20.0)
    p.add_argument('--dt-us', type=float, default=0.2, help='Recovery interval and plot step in microseconds.')
    p.add_argument('--B-T', type=float, default=0.05)
    p.add_argument('--pump-rate', type=float, default=5.0)
    p.add_argument('--output-dir', default='outputs')
    p.add_argument('--skip-checks', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spins = load_dqp_spins(args.dqp_file)
    selected = choose_two_spins(spins, args.selection, tuple(args.spin_indices) if args.spin_indices else None)
    params = model_params_from_spins(selected, B_T=args.B_T, pump_rate=args.pump_rate)
    code = make_two_qubit_qec_code(tuple(s.dephasing_g_MHz for s in selected))
    tlist = np.arange(0.0, args.time_us + 0.5 * args.dt_us, args.dt_us)
    result = simulate_qec_overlay(tlist, params, code)
    if not args.skip_checks:
        run_sanity_checks(code, result)

    out = Path(args.output_dir)
    png = out / 'qec_bell_fidelity_overlay.png'
    svg = out / 'qec_bell_fidelity_overlay.svg'
    csv_path = out / 'qec_bell_fidelity_overlay.csv'
    plot_overlay(result, selected, png)
    plot_overlay(result, selected, svg)
    save_csv(result, csv_path)

    print('Simulation completed.')
    print('Selected DQP spins, ordered as |g1| >= |g2|:')
    for s in selected:
        print(f'  spin #{s.index}: -A_parallel={s.minus_A_par_kHz:.9g} kHz, A_parallel={-s.minus_A_par_kHz:.9g} kHz, A_perp={s.A_perp_kHz:.9g} kHz')
    print(f'QEC couplings: g1={code.g_MHz[0]:.9g} MHz, g2={code.g_MHz[1]:.9g} MHz')
    dt_value = result['dt_us']
    f_without = result['fidelity_without_protocol'][-1]
    f_with = result['fidelity_with_protocol'][-1]
    print(f'Recovery interval: {dt_value:.9g} us')
    print('Final Bell-state fidelities:')
    print(f'  without protocol           : {f_without:.6f}')
    print(f'  with QEC protocol, decoded : {f_with:.6f}')
    print(f'  final improvement          : {f_with - f_without:+.6f}')
    print('Saved:')
    print(f'  {png}')
    print(f'  {svg}')
    print(f'  {csv_path}')


if __name__ == '__main__':
    main()
