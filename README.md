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

The production simulations evolve the complete five-level-electron/two-nuclear-spin model in the laboratory frame. Nuclear encoding, recovery, and readout operators are transformed into the laboratory frame at their actual application time.

The former static approximation

```text
H_interaction = H_g1 - H_g0
```

is not used when `A_perp != 0`, because the transverse term must rotate at the nuclear Larmor frequency. The compatibility function `make_nv_two_c13_model(..., interaction_picture=True)` raises for nonzero `A_perp` instead of silently returning an invalid static Hamiltonian. A regression test verifies exact lab/rotating-frame agreement when `A_perp = 0`.

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

### 5. Configurable excited-state hyperfine tensor

The excited-state `m_s=-1` nuclear Hamiltonian can now use a tensor proportional to the ground-state `m_s=-1` tensor:

```text
A_parallel,e = r * A_parallel,g
A_perp,e     = r * A_perp,g
```

The default is

```text
r = 0.1
```

and applies to both nuclei in the selected pair. The `e0` manifold and singlet retain free nuclear precession in this simplified five-level model.

The public helper is

```python
set_excited_state_hyperfine_from_ground(
    A_par_MHz,
    A_perp_MHz,
    scale=0.1,
)
```

and both ideal and pulsed simulations accept the same parameter through

```python
excited_state_hyperfine_scale=0.1
```

Setting `r=0` exactly recovers the previous model in which the excited manifolds had no state-dependent hyperfine term.

This proportional tensor is a modeling ansatz. It should be replaced by a measured excited-state tensor when one is available.

## User-supplied register

`dqpspins.json` contains:

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

Ideal recovery-map reference with the default excited-state scale:

```bash
python nv_5level_two_c13_simulation.py \
  --spin-names X2 S4 \
  --protocol ideal \
  --time-us 20 \
  --plot-dt-us 0.5 \
  --recovery-interval-us 10 \
  --excited-hyperfine-scale 0.1
```

Recover the former zero-excited-hyperfine model:

```bash
python nv_5level_two_c13_simulation.py \
  --spin-names X2 S4 \
  --protocol ideal \
  --excited-hyperfine-scale 0
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
  --correction-gate-us 0.75 \
  --excited-hyperfine-scale 0.1
```

The compatibility entry point selects the pulsed model:

```bash
python realistic_recovery_interval_simulation.py \
  --spin-names X2 S3 \
  --excited-hyperfine-scale 0.1
```

## Generate every pair

```bash
python simulate_all_pairs.py --force --excited-hyperfine-scale 0.1
```

The command creates one SVG and one CSV for each of the 21 pairs, plus:

- [`outputs/all_pairs/all_pairs_grid.svg`](outputs/all_pairs/all_pairs_grid.svg), containing all 21 pair plots in one reviewable figure;
- [`outputs/all_pairs/all_pairs_ranking.svg`](outputs/all_pairs/all_pairs_ranking.svg);
- [`outputs/all_pairs/all_pairs_summary.csv`](outputs/all_pairs/all_pairs_summary.csv).

The combined grid, ranking, and summary are committed. The 21 individual SVG/CSV pairs are reproducible generated outputs. Existing pair outputs are reused unless `--force` is supplied, which makes interrupted long sweeps resumable.

## Results committed in this revision

Simulation settings for the committed all-pair sweep:

- laboratory-frame five-level model;
- excited-state `m_s=-1` tensor scale `r = 0.1`;
- `B = 0.05 T`;
- optical pump rate `5 / us`;
- total time `20 us`;
- ideal instantaneous recovery at `10 us` and `20 us`;
- output every `0.5 us`;
- independently calibrated final virtual-Z phase for each channel.

At `20 us`, the best pair in this simplified model remains **X2-S4**:

```text
QEC average fidelity          0.998877
best constituent spin        0.998697
QEC - best single            +0.000180
```

The more precise difference is `+0.0001796268`, or `+0.01796` percentage points. It is a numerical crossing rather than a robust experimental margin.

Under the same settings, **S3-X2 does not beat X2**:

```text
QEC average fidelity          0.990899
best constituent spin        0.995228
QEC - best single            -0.004328
```

Changing the excited-state scale from `0` to `0.1` changes the numerical curves slightly but does not change the pair-level conclusion at this operating point: X2-S4 is the only pair crossing the best-single benchmark.

The full table, rather than only these examples, should be used when changing pump duration, recovery cadence, field, excited-state scale, or error parameters.

## Tests

```bash
python -m unittest discover -s tests -v
```

The 13 regression tests cover:

- code-basis orthogonality and CPTP recovery completeness;
- six-state identity-channel fidelity;
- explicit electron-ancilla recovery versus the ideal Kraus map;
- rejection of an invalid static interaction picture for nonzero `A_perp`;
- exact lab/rotating-frame agreement at `A_perp = 0`;
- construction of the best-single-spin envelope;
- pair-sweep order independence;
- presence of the X2 candidate in the register;
- default `r=0.1` excited-state scaling;
- exact proportionality of the excited `m_s=-1` tensor;
- recovery of the former model when `r=0`;
- propagation of the scale through pair-model parameters;
- rejection of non-finite scale values.

## Important physical limitation

The five-level optical model remains deliberately simplified. The ground `g1` manifold uses the supplied ground-state hyperfine tensor, and the excited `e1` manifold now uses the proportional ansatz `A_e = r A_g`. The `e0` and singlet manifolds retain free nuclear precession.

The model still does not include an independently measured excited-state tensor, singlet hyperfine interaction, NV charge switching, or the exact laboratory gate compilation. A single scalar `r` assumes that `A_parallel` and `A_perp` scale by the same factor and that this factor is the same for both nuclei. Those assumptions may not hold in the physical NV.

Consequently, the committed plots validate the software implementation and show the sensitivity to this particular excited-state ansatz, but they are not a quantitative prediction of the final experiment until the optical nuclear channel is calibrated under the exact laser/reset sequence.

## Why the code coupling may not equal DDRF `-A_parallel`

The QEC encoder is designed from the signed coefficient vector of the *actual correlated error mode* during the optical block. DDRF ground-state spectroscopy supplies useful starting values, but optical excitation samples several electron and charge manifolds. Their dwell times and nuclear Hamiltonians determine the integrated random phase.

The final code should therefore use a measured vector `g_opt = (g1_opt, g2_opt)`, obtained from correlated two-spin phase data under the exact optical sequence. See [`docs/optical_coupling_calibration.md`](docs/optical_coupling_calibration.md) and [`calibrate_optical_mode.py`](calibrate_optical_mode.py).
