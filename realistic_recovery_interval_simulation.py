'''Gate-budget-aware QEC recovery interval simulations for nv-simulation.

This script reuses the five-level NV + two-13C model from
nv_5level_two_c13_simulation.py, but separates the plotted sample interval from
how often the error-correction recovery is applied.

It writes two overlays by default:

1. qec_bell_fidelity_periodic_realistic_interval.*
   Ideal QEC recovery every --recovery-interval-us, default 5 us.

2. qec_bell_fidelity_single_recovery.*
   Exactly one ideal recovery operation at --single-recovery-time-us, default
   half of --time-us.

The recovery map is still ideal and instantaneous. The longer interval is a
cadence-level proxy for the fact that the protocol uses several one- and
two-qubit gates; finite gate durations and gate errors are not yet modeled.
'''
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla

from nv_5level_two_c13_simulation import (
    apply_recovery_to_full_state,
    bell_fidelity,
    bell_state_psi_plus,
    choose_two_spins,
    decoded_bell_fidelity,
    density,
    evolve_liouvillian,
    hermitize,
    ket,
    lindblad_liouvillian,
    load_dqp_spins,
    make_nv_two_c13_model,
    make_two_qubit_qec_code,
    mat,
    model_params_from_spins,
    partial_trace_electron,
    vec,
)


def selected_spin_g_mhz(spin):
    '''Return the dephasing coefficient used by the n=2 code.'''
    if hasattr(spin, 'dephasing_g_MHz'):
        return spin.dephasing_g_MHz
    return spin.g_MHz


def setup_problem(model_params: dict, code):
    H, collapse_ops, rates, dims = make_nv_two_c13_model(**model_params)
    de, dn = dims
    dim = de * dn
    L = lindblad_liouvillian(H, collapse_ops)
    electron_g0 = ket(de, 0)
    bell = bell_state_psi_plus()
    raw_initial = density(np.kron(electron_g0, bell))
    encoded_initial = density(np.kron(electron_g0, code.encoding_unitary @ bell))
    return L, rates, de, dn, dim, raw_initial, encoded_initial


def no_protocol_curve(L, raw_initial, tlist_us, de, dn, dim):
    raw_vecs = evolve_liouvillian(L, raw_initial, tlist_us)
    values = []
    for rho_v in raw_vecs:
        rho = hermitize(mat(rho_v, dim))
        rho_n = partial_trace_electron(rho, de, dn)
        values.append(bell_fidelity(rho_n))
    return np.asarray(values, dtype=float)


def evolve_one_step(L, rho, dt_us, dim):
    if abs(dt_us) < 1e-14:
        return rho
    return mat(spla.expm_multiply(L * float(dt_us), vec(rho)), dim)


def simulate_periodic_recovery(tlist_us, model_params, code, recovery_interval_us):
    if recovery_interval_us <= 0:
        raise ValueError('recovery_interval_us must be positive')
    L, rates, de, dn, dim, raw_initial, rho_qec = setup_problem(model_params, code)
    f_without = no_protocol_curve(L, raw_initial, tlist_us, de, dn, dim)

    f_with = np.zeros(len(tlist_us))
    traces = np.zeros(len(tlist_us))
    recovery_count = np.zeros(len(tlist_us), dtype=int)

    current_t = 0.0
    next_recovery = float(recovery_interval_us)
    count = 0
    tol = 1e-12

    for n, target_t in enumerate(np.asarray(tlist_us, dtype=float)):
        while next_recovery <= target_t + tol:
            rho_qec = hermitize(evolve_one_step(L, rho_qec, next_recovery - current_t, dim))
            rho_qec = hermitize(apply_recovery_to_full_state(rho_qec, code, de))
            current_t = next_recovery
            next_recovery += float(recovery_interval_us)
            count += 1
        if target_t > current_t + tol:
            rho_qec = hermitize(evolve_one_step(L, rho_qec, target_t - current_t, dim))
            current_t = target_t
        f_with[n] = decoded_bell_fidelity(rho_qec, code, de, dn)
        traces[n] = float(np.trace(rho_qec).real)
        recovery_count[n] = count

    return {
        'mode': 'periodic',
        't_us': np.asarray(tlist_us, dtype=float),
        'fidelity_without_protocol': f_without,
        'fidelity_with_protocol': f_with,
        'trace_with_protocol': traces,
        'recovery_count': recovery_count,
        'recovery_interval_us': float(recovery_interval_us),
        'rates': rates,
    }


def simulate_single_recovery_trace(tlist_us, model_params, code, recovery_time_us):
    tlist_us = np.asarray(tlist_us, dtype=float)
    if recovery_time_us < 0 or recovery_time_us > float(tlist_us[-1]):
        raise ValueError('single recovery time must lie inside the plotted time range')

    L, rates, de, dn, dim, raw_initial, rho_qec = setup_problem(model_params, code)
    f_without = no_protocol_curve(L, raw_initial, tlist_us, de, dn, dim)

    f_with = np.zeros(len(tlist_us))
    traces = np.zeros(len(tlist_us))
    recovery_count = np.zeros(len(tlist_us), dtype=int)

    current_t = 0.0
    recovered = False
    tol = 1e-12

    for n, target_t in enumerate(tlist_us):
        if (not recovered) and recovery_time_us <= target_t + tol:
            rho_qec = hermitize(evolve_one_step(L, rho_qec, recovery_time_us - current_t, dim))
            rho_qec = hermitize(apply_recovery_to_full_state(rho_qec, code, de))
            current_t = float(recovery_time_us)
            recovered = True
        if target_t > current_t + tol:
            rho_qec = hermitize(evolve_one_step(L, rho_qec, target_t - current_t, dim))
            current_t = float(target_t)
        f_with[n] = decoded_bell_fidelity(rho_qec, code, de, dn)
        traces[n] = float(np.trace(rho_qec).real)
        recovery_count[n] = 1 if recovered else 0

    return {
        'mode': 'single',
        't_us': tlist_us,
        'fidelity_without_protocol': f_without,
        'fidelity_with_protocol': f_with,
        'trace_with_protocol': traces,
        'recovery_count': recovery_count,
        'single_recovery_time_us': float(recovery_time_us),
        'rates': rates,
    }


def save_csv(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            't_us',
            'bell_fidelity_without_protocol',
            'decoded_bell_fidelity_with_qec',
            'recovery_count_applied',
        ])
        for row in zip(
            result['t_us'],
            result['fidelity_without_protocol'],
            result['fidelity_with_protocol'],
            result['recovery_count'],
        ):
            writer.writerow([f'{row[0]:.12g}', f'{row[1]:.12g}', f'{row[2]:.12g}', int(row[3])])


def spin_text(selected):
    return ', '.join(
        f'#{s.index}: -Apar={s.minus_A_par_kHz:.3g} kHz, Aperp={s.A_perp_kHz:.3g} kHz'
        for s in selected
    )


def plot_overlay(result, selected, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.4, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(result['t_us'], result['fidelity_without_protocol'], label='Without protocol: physical |Psi+>')

    if result['mode'] == 'periodic':
        ax.plot(
            result['t_us'],
            result['fidelity_with_protocol'],
            label=f"With QEC: ideal recovery every {result['recovery_interval_us']:.3g} us",
        )
        recovery_times = result['t_us'][np.r_[False, np.diff(result['recovery_count']) > 0]]
        for rt in recovery_times:
            ax.axvline(rt, linestyle=':', linewidth=0.8, alpha=0.35)
        title = 'Bell-state fidelity with realistic-cadence QEC recovery'
        note = f"DQP spins {spin_text(selected)}\nrecovery interval = {result['recovery_interval_us']:.3g} us"
    else:
        ax.axvline(result['single_recovery_time_us'], linestyle='--', linewidth=1.0, label='single recovery time')
        ax.plot(result['t_us'], result['fidelity_with_protocol'], label='With QEC: exactly one ideal recovery')
        title = 'Bell-state fidelity with one QEC recovery operation'
        note = f"DQP spins {spin_text(selected)}\nsingle recovery at t = {result['single_recovery_time_us']:.3g} us"

    ax.set_xlabel('Time (us)')
    ax.set_ylabel('Bell-state fidelity')
    ax.set_title(title)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.text(
        0.02,
        0.04,
        note + '\nideal instantaneous recovery channel; gate errors not included',
        transform=ax.transAxes,
        fontsize=8.2,
        va='bottom',
        bbox={'boxstyle': 'round', 'alpha': 0.12},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dqp-file', default='dqpspins.json')
    parser.add_argument('--selection', choices=['strongest', 'closest-positive'], default='strongest')
    parser.add_argument('--spin-indices', nargs=2, type=int, metavar=('I', 'J'))
    parser.add_argument('--time-us', type=float, default=20.0)
    parser.add_argument('--plot-dt-us', type=float, default=0.1)
    parser.add_argument('--recovery-interval-us', type=float, default=5.0)
    parser.add_argument('--single-recovery-time-us', type=float, default=None)
    parser.add_argument('--B-T', type=float, default=0.05)
    parser.add_argument('--pump-rate', type=float, default=5.0)
    parser.add_argument('--output-dir', default='outputs')
    return parser.parse_args()


def main():
    args = parse_args()
    spins = load_dqp_spins(args.dqp_file)
    selected = choose_two_spins(spins, args.selection, tuple(args.spin_indices) if args.spin_indices else None)
    params = model_params_from_spins(selected, B_T=args.B_T, pump_rate=args.pump_rate)
    code = make_two_qubit_qec_code(tuple(selected_spin_g_mhz(s) for s in selected))
    tlist = np.arange(0.0, args.time_us + 0.5 * args.plot_dt_us, args.plot_dt_us)
    single_time = args.single_recovery_time_us if args.single_recovery_time_us is not None else 0.5 * args.time_us

    periodic = simulate_periodic_recovery(tlist, params, code, args.recovery_interval_us)
    single = simulate_single_recovery_trace(tlist, params, code, single_time)

    out = Path(args.output_dir)
    outputs = [
        (periodic, out / 'qec_bell_fidelity_periodic_realistic_interval'),
        (single, out / 'qec_bell_fidelity_single_recovery'),
    ]
    for result, stem in outputs:
        plot_overlay(result, selected, stem.with_suffix('.png'))
        plot_overlay(result, selected, stem.with_suffix('.svg'))
        save_csv(result, stem.with_suffix('.csv'))

    print('Simulation completed.')
    print(f'Periodic recovery interval: {args.recovery_interval_us:g} us')
    print(f'Single recovery time: {single_time:g} us')
    print('Final Bell-state fidelities at t = {:.6g} us:'.format(tlist[-1]))
    print(f"  without protocol                   : {periodic['fidelity_without_protocol'][-1]:.6f}")
    print(f"  periodic QEC, decoded              : {periodic['fidelity_with_protocol'][-1]:.6f}")
    print(f"  single-recovery QEC trace, decoded : {single['fidelity_with_protocol'][-1]:.6f}")
    print('Saved:')
    for _, stem in outputs:
        print(f'  {stem}.png')
        print(f'  {stem}.svg')
        print(f'  {stem}.csv')


if __name__ == '__main__':
    main()
