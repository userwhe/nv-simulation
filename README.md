# nv-simulation

NV-center optical-noise simulations for two coupled `13C` nuclear spins and the `n = 2` common-fluctuator error-correction code of Layden, Chen, and Cappellaro.

## What is fixed in this version

### 1. The benchmark is now the best physical nuclear spin

The old script compared the corrected encoding with an unencoded two-spin Bell coherence. That is not the break-even test for a nuclear memory.

The corrected simulation prepares the six Pauli eigenstates

`|0>`, `|1>`, `|+X>`, `|-X>`, `|+Y>`, and `|-Y>`

for each of three channels:

- the encoded-and-corrected logical qubit;
- physical spin 1, with the other nucleus maximally mixed;
- physical spin 2, with the other nucleus maximally mixed.

Their mean state fidelity is the qubit average gate fidelity because the six states form a projective state 2-design. A calibrated final virtual-Z correction is optimized independently for each memory. The reported break-even quantity is

```text
QEC six-state average fidelity - max(single-spin-1 fidelity, single-spin-2 fidelity)
```

Leakage is not renormalized away.

### 2. Nonzero `A_perp` is handled in a valid frame

The production simulations now evolve the complete five-level-electron/two-nuclear-spin model in the laboratory frame. Nuclear encoding, recovery, and readout operators are transformed into the laboratory frame at their actual application time.

The former static approximation

```text
H_interaction = H_g1 - H_g0
```

is not used when `A_perp != 0`, because the transverse term must rotate at the nuclear Larmor frequency. The compatibility function `make_nv_two_c13_model(..., interaction_picture=True)` now raises for nonzero `A_perp` instead of silently returning an invalid static Hamiltonian. A regression test verifies exact lab/rotating-frame agreement when `A_perp = 0`.

### 3. The pulsed experiment is a separate function

`simulate_ideal_periodic_process(...)` remains the clean ideal reference: continuous optical evolution with instantaneous ideal recovery maps.

`simulate_pulsed_experimental_process(...)` is separate and explicitly segments:

1. laser-on optical noise;
2. pump-off excited/singlet settling;
3. electron reset;
4. finite syndrome-gate duration;
5. electron-ancilla syndrome extraction;
6. measurement dephasing and optional assignment error;
7. measurement plus feedback latency;
8. finite conditional-correction duration;
9. final electron reset and optional nuclear depolarizing error.

The syndrome and conditional correction are implemented as explicit unitaries on the electron `g0/g1` ancilla and the nuclear register. This is a timing-and-error model, not yet a compilation of the laboratory DDRF pulse sequence.

### 4. Pair sweeps are order-independent

The unsafe dense-propagator cache keyed by `id(L)` was removed. Batched Krylov propagation is used directly, so a Python object ID cannot accidentally reuse a propagator from another spin pair. A regression test runs `A -> B -> A` and checks identical results for the two `A` runs.

## User-supplied register

`dqpspins.json` now contains:

| Spin | `-A_parallel` (kHz) | `A_perp` (kHz) |
|---|---:|---:|
| S1 | 5.616 | 32.847 |
| S2 | 224.217 | 189.318 |
| S3 | 49.837 | 101.007 |
| S4 | 15.734 | 19.295 |
| S5 | 3.832 | 21.140 |
| S6 | -7.015 | 38.250 |
| X2 | 33.962 | 26.000 |

Both this named dictionary format and the repository's previous two-number row format are accepted.

## Reproduce one pair

Install dependencies:

```bash
pip install -r requirements.txt
```

Ideal recovery-map reference:

```bash
python nv_5level_two_c13_simulation.py \
  --spin-names X2 S4 \
  --protocol ideal \
  --time-us 20 \
  --plot-dt-us 0.5 \
  --recovery-interval-us 10
```

Explicit pulsed electron-ancilla sequence:

```bash
python nv_5level_two_c13_simulation.py \
  --spin-names X2 S3 \
  --protocol pulsed \
  --cycles 2 \
  --laser-on-us 8 \
  --laser-settle-us 0.25 \
  --syndrome-gate-us 0.75 \
  --measurement-us 0.5 \
  --feedback-latency-us 0.5 \
  --correction-gate-us 0.75
```

The compatibility entry point now selects the pulsed model:

```bash
python realistic_recovery_interval_simulation.py --spin-names X2 S3
```

## Generate every pair

```bash
python simulate_all_pairs.py
```

The command creates one SVG and one CSV for each of the 21 pairs, plus:

- [`outputs/all_pairs/all_pairs_grid.svg`](outputs/all_pairs/all_pairs_grid.svg), containing all 21 pair plots in one reviewable figure;
- [`outputs/all_pairs/all_pairs_ranking.svg`](outputs/all_pairs/all_pairs_ranking.svg);
- [`outputs/all_pairs/all_pairs_summary.csv`](outputs/all_pairs/all_pairs_summary.csv).

The combined grid, ranking, and summary are committed in this revision. The 21 individual SVG/CSV pairs are reproducible generated outputs. Existing pair outputs are reused unless `--force` is supplied, which makes interrupted long sweeps resumable.

## Results committed in this revision

Simulation settings for the committed all-pair sweep:

- laboratory-frame five-level model;
- `B = 0.05 T`;
- optical pump rate `5 / us`;
- total time `20 us`;
- ideal instantaneous recovery at `10 us` and `20 us`;
- output every `0.5 us`;
- independently calibrated final virtual-Z phase for each channel.

At `20 us`, the best pair in this simplified model is **X2-S4**:

```text
QEC average fidelity          0.998883
best constituent spin        0.998704
QEC - best single            +0.000179
```

This is only `0.0179` percentage points, so it is a numerical crossing rather than a robust experimental margin. Under the same settings, **S3-X2 does not beat X2**:

```text
QEC average fidelity          0.990942
best constituent spin        0.995254
QEC - best single            -0.004312
```

The full table, rather than only these examples, should be used when changing pump duration, recovery cadence, field, or error parameters.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover:

- code-basis orthogonality and CPTP recovery completeness;
- six-state identity-channel fidelity;
- explicit electron-ancilla recovery versus the ideal Kraus map;
- rejection of an invalid static interaction picture for nonzero `A_perp`;
- exact lab/rotating-frame agreement at `A_perp = 0`;
- construction of the best-single-spin envelope;
- pair-sweep order independence;
- presence of the X2 candidate in the register.

## Important physical limitation

The five-level optical model remains deliberately simplified. It assigns free nuclear precession to the excited and singlet manifolds and includes state-dependent hyperfine coupling only in the ground `g1` manifold. It does not yet include measured excited-state/singlet hyperfine tensors, NV charge switching, or the exact laboratory gate compilation.

Consequently, the committed plots validate the corrected software benchmark and frame treatment, but they are not a quantitative prediction of the final experiment until the optical nuclear channel is calibrated under the exact laser/reset sequence.

## Why the code coupling may not equal DDRF `-A_parallel`

The QEC encoder is designed from the signed coefficient vector of the *actual correlated error mode* during the optical block. DDRF ground-state spectroscopy supplies useful starting values, but optical excitation samples several electron and charge manifolds. Their dwell times and nuclear Hamiltonians determine the integrated random phase.

The final code should therefore use a measured vector `g_opt = (g1_opt, g2_opt)`, obtained from correlated two-spin phase data under the exact optical sequence. See the discussion accompanying this revision for the recommended covariance/eigenmode calibration.
