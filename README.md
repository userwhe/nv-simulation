# Ideal two-spin NV QEC readout benchmark

This repository implements the `n = 2` common-fluctuator code of [Layden, Chen, and Cappellaro](https://doi.org/10.1103/PhysRevLett.124.020504) for two `13C` nuclear spins coupled to a five-level NV electron.

It is deliberately an **ideal readout-recovery benchmark**. Each reported time is a stopping-time experiment: the encoded state evolves uninterrupted from `0` to `t`, then one instantaneous, perfect recovery is applied to a copy immediately before logical readout. There is no periodic recovery during evolution. The model does not include gate compilation or duration, measurement and feedback latency, assignment error, reset infidelity, or other control errors. A positive QEC advantage is therefore an ideal-model crossing, not experimental break-even.

## Physical model

The electron basis is `g0`, `g1`, `e0`, `e1`, and a singlet. The nuclear Hamiltonians are

\[
H_{\mathrm{free}}=\omega_L(I_{z,1}+I_{z,2}),
\]

\[
H_{g1}=\sum_{j=1}^{2}
\left[(\omega_L-A_{\parallel,j})I_{z,j}+A_{\perp,j}I_{x,j}\right].
\]

The `g0`, excited, and singlet manifolds use `H_free`; only `g1` uses `H_g1`. Lindblad operators describe optical pumping, radiative decay, intersystem crossing, and singlet decay. The electron starts in `g0` for the logical memory and both physical controls.

All optical rates are collected in the immutable `OpticalParameters` object. The excitation rate is modeled as

\[
W_{\mathrm{pump}}=P_{\mathrm{laser}}\,c_{\mathrm{exc}},
\]

where the default calibration is `c_exc = 1 us^-1/uW`. This explicit modeling convention preserves the former numerical setting (`5 uW` gives `5 us^-1`), but it is not a measured power calibration. Set `excitation_rate_per_us_per_uw` to the calibrated value for a particular optical setup. The default fixed transition rates are:

| Transition | Rate (`us^-1`) |
|---|---:|
| `e0 -> g0` radiative | 66.153846 |
| `e1 -> g1` radiative | 64.285714 |
| `e0 -> singlet` ISC | 10.769231 |
| `e1 -> singlet` ISC | 78.571429 |
| `singlet -> g0` | 4.722222 |
| `singlet -> g1` | 0.833333 |

Every rate has a corresponding command-line option shown by `python run_simulation.py --help`, and every field can be set directly when constructing `OpticalParameters` from Python.

All internal Hamiltonian frequencies use angular frequency in `rad/us`. Input conversion happens once:

\[
A\,[\mathrm{rad/us}]=2\pi\,10^{-3}A\,[\mathrm{kHz}],
\qquad
\omega_L=2\pi(10.705\,\mathrm{MHz/T})B.
\]

## Two-spin code and ideal recovery

For this simplified model, the code-design coupling is explicitly

\[
g_j=-A_{\parallel,j}.
\]

The implementation orders the spins internally so that `|g1| >= |g2|`, removes a common sign so `g1 > 0`, and maps every result back to the user's original spin IDs. Define

\[
a=\sqrt{\frac{g_1-g_2}{2g_1}},\qquad
b=\sqrt{\frac{g_1+g_2}{2g_1}},
\]

\[
|\chi_0\rangle=a|0\rangle+b|1\rangle,\qquad
|\chi_1\rangle=-b|0\rangle+a|1\rangle,
\]

\[
|0_L\rangle=|\chi_0\rangle|0\rangle,\qquad
|1_L\rangle=|\chi_1\rangle|1\rangle.
\]

With error states `|0_E> = |chi1>|0>` and `|1_E> = |chi0>|1>`, the two recovery Kraus operators are

\[
K_0=P_L,\qquad K_1=(U_x\otimes I)P_E,
\]

where `Ux` exchanges `|chi0>` and `|chi1>`. They satisfy `K0†K0 + K1†K1 = I`.

## Laboratory and rotating frames

Evolution always uses the complete laboratory-frame Hamiltonian. This is required when either transverse coupling is nonzero; a static `H_g1 - H_g0` interaction-picture Hamiltonian would be incorrect.

The code, recovery, and logical readout are defined in the nuclear rotating frame using

\[
R(t)=e^{-iH_{\mathrm{free}}t},\qquad
\rho_{\mathrm{rot}}(t)=R^\dagger(t)\rho_{\mathrm{lab}}(t)R(t),
\]

\[
O_{\mathrm{lab}}(t)=R(t)O_{\mathrm{rot}}R^\dagger(t).
\]

The readout recovery uses the absolute stopping time. It is applied only to a copy of the evolved state, so evaluating an earlier sample never changes the trajectory used for later samples and the frame clock never restarts.

## Memory metric and controls

Each channel is evaluated on the six Pauli eigenstates:

\[
F_6=\frac{1}{6}\sum_{\psi\in\{\pm X,\pm Y,\pm Z\}}
\langle\psi|\rho_{\mathrm{out}}^{(\psi)}|\psi\rangle.
\]

This is called **six-state average memory fidelity**. Logical decoding uses `W† rho W` without renormalization, so leakage counts as failure. Both raw fidelity and the exact continuously optimized virtual-`Z`-corrected fidelity are reported.

The two controls store the same six states in each physical spin in turn, with the other spin maximally mixed. They use the same electron state, elapsed optical noise, rotating frame, and virtual-`Z` rule, but no QEC recovery. The benchmark is the better physical memory:

```text
best_single(t) = max(fidelity_spin_a(t), fidelity_spin_b(t))
qec_advantage(t) = fidelity_qec(t) - best_single(t)
```

## Built-in spin register

The stored values are signed `A_parallel` values.

| Spin ID | `A_parallel_kHz` | `A_perp_kHz` |
|---:|---:|---:|
| 1 | -5.616 | 32.847 |
| 2 | -224.217 | 189.318 |
| 3 | -49.837 | 101.007 |
| 4 | -15.734 | 19.295 |
| 5 | -3.832 | 21.140 |
| 6 | 7.015 | 38.250 |
| 7 | -33.962 | 26.000 |

## Run

Install the three dependencies:

```bash
pip install -r requirements.txt
```

Use built-in spins:

```bash
python run_simulation.py \
  --spin-ids 3 7 \
  --duration-us 20 \
  --sample-dt-us 0.5 \
  --b-field-t 0.05 \
  --laser-power-uw 5 \
  --output-dir outputs
```

Or provide both spins directly. Direct inputs are labeled spin `1` and spin `2` in their argument order.

```bash
python run_simulation.py \
  --spin1-a-parallel-khz -49.837 \
  --spin1-a-perp-khz 101.007 \
  --spin2-a-parallel-khz -33.962 \
  --spin2-a-perp-khz 26.0 \
  --duration-us 20 \
  --sample-dt-us 0.5 \
  --b-field-t 0.05 \
  --laser-power-uw 5
```

Generate the individual and combined plots for all 21 built-in pairs:

```bash
python run_simulation.py \
  --all-pairs \
  --duration-us 40 \
  --sample-dt-us 0.5 \
  --b-field-t 0.05 \
  --laser-power-uw 5 \
  --output-dir outputs/all_pairs_5uw
```

A pair run writes `results.csv` and `fidelity_comparison.svg`. An all-pairs run writes those files in 21 pair subdirectories and adds `all_pairs_fidelity.svg`, whose subplots share one fidelity scale. Generated outputs are ignored by Git.

The same protocol is available from Python:

```python
from nv_qec import (
    BUILTIN_SPINS,
    OpticalParameters,
    SimulationConfig,
    simulate_qec_memory,
)

result = simulate_qec_memory(
    SimulationConfig(
        duration_us=20,
        sample_dt_us=0.5,
        b_field_t=0.05,
        optical=OpticalParameters(laser_power_uw=5),
    ),
    BUILTIN_SPINS[3],
    BUILTIN_SPINS[7],
)
```

A dense final-time power sweep can hold the spin pair and delay fixed:

```python
from dataclasses import replace

import numpy as np

from nv_qec import (
    BUILTIN_SPINS,
    OpticalParameters,
    SimulationConfig,
    save_power_sweep_csv,
    save_power_sweep_plot,
    simulate_laser_power_sweep,
)

spin_2 = replace(BUILTIN_SPINS[2], a_perp_khz=0)
spin_7 = replace(BUILTIN_SPINS[7], a_perp_khz=0)
config = SimulationConfig(
    duration_us=20,
    b_field_t=0.05,
    optical=OpticalParameters(),
)
powers_uw = np.geomspace(0.1, 10, 121)
results = simulate_laser_power_sweep(config, spin_2, spin_7, powers_uw)
save_power_sweep_csv(results, "outputs/power_sweep.csv")
save_power_sweep_plot(results, "outputs/power_sweep.svg")
```

## Tests

```bash
python -m unittest discover -s tests -v
```
