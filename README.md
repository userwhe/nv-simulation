# nv-simulation

NV-center dynamics simulations with a five-level electron model and two coupled `13C` nuclear spins.

This repository includes a two-qubit common-fluctuator dephasing QEC simulation based on the `n = 2` hardware-efficient code of Layden, Chen, and Cappellaro.

## Realistic recovery interval update

The recovery cadence should not be treated as the plotting time step. The recovery protocol uses several operations, so this update records a more realistic cadence-level run with:

- 20 us total storage time;
- DQP spins #2 and #3, selected by strongest `|-A_parallel|`;
- periodic ideal QEC recovery every 10 us;
- a separate single-recovery trace with exactly one recovery at t = 10 us;
- 0.5 us compact plot/data sampling for the committed CSV/SVG artifacts.

The recovery channel is still modeled as ideal and instantaneous. Finite gate duration, gate infidelity, measurement latency, and feedback latency are not included.

## Reproduce the 10 us run

```bash
python run_qec_realistic_recovery_10us.py
```

Equivalent explicit command:

```bash
python realistic_recovery_interval_simulation.py \
  --recovery-interval-us 10 \
  --plot-dt-us 0.5 \
  --single-recovery-time-us 10
```

## Published outputs

- `outputs/qec_bell_fidelity_periodic_10us.svg`
- `outputs/qec_bell_fidelity_periodic_10us.csv`
- `outputs/qec_bell_fidelity_single_recovery_at_10us.svg`
- `outputs/qec_bell_fidelity_single_recovery_at_10us.csv`

Final Bell-state fidelities at 20 us:

| Case | Final Bell fidelity |
|---|---:|
| Without protocol | 0.615048 |
| Periodic QEC, recovery every 10 us | 0.840730 |
| Single QEC recovery at 10 us | 0.773930 |

## Dependencies

```bash
pip install -r requirements.txt
```
