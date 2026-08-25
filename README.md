# Ideal two-spin NV QEC benchmark

This repository implements the `n = 2` common-fluctuator code of [Layden, Chen, and Cappellaro](https://doi.org/10.1103/PhysRevLett.124.020504) for two `13C` nuclear spins coupled to a five-level NV electron.

It is deliberately an **ideal-recovery benchmark**. Recovery is instantaneous and perfect. The model does not include gate compilation or duration, measurement and feedback latency, assignment error, reset infidelity, or other control errors. A positive QEC advantage is therefore an ideal-model crossing, not experimental break-even.

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

Recovery operators use the absolute simulation time; the frame clock never restarts. At a scheduled time, recovery changes the continuing encoded trajectory. At every other output time, recovery is applied only to a copy used for stopping-time readout.

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
  --recovery-interval-us 10 \
  --b-field-t 0.05 \
  --pump-rate-per-us 5 \
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
  --recovery-interval-us 10 \
  --b-field-t 0.05
```

The command writes `results.csv` and `fidelity_comparison.svg`. Generated outputs are ignored by Git.

The same protocol is available from Python:

```python
from nv_qec import BUILTIN_SPINS, SimulationConfig, simulate_qec_memory

result = simulate_qec_memory(
    SimulationConfig(
        duration_us=20,
        sample_dt_us=0.5,
        recovery_interval_us=10,
        b_field_t=0.05,
        pump_rate_per_us=5,
    ),
    BUILTIN_SPINS[3],
    BUILTIN_SPINS[7],
)
```

## Tests

```bash
python -m unittest discover -s tests -v
```
