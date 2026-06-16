# nv-simulation

NV-center dynamics simulations with a five-level electron model and two coupled `13C` nuclear spins.

This repository includes a two-qubit common-fluctuator dephasing QEC simulation based on the `n = 2` code of Layden, Chen, and Cappellaro. The scripts can:

- parse DQP spin pairs stored as `(-A_parallel, A_perp)` in kHz;
- select two spins automatically or by explicit index;
- convert the selected hyperfine parameters to the five-level Lindblad model;
- run the physical Bell-state evolution without QEC;
- encode the Bell state into the two-qubit QEC code, apply recovery, decode, and report the Bell-state fidelity;
- save overlay plots and CSV data.

## Default DQP spin pair

The default selected spins are the two strongest by `|-A_parallel|`:

| DQP spin | `-A_parallel` (kHz) | `A_perp` (kHz) |
|---:|---:|---:|
| 2 | 224.2991967686721 | 199.07207268788648 |
| 3 | 50.10705093918639 | 108.7263051655999 |

## Gate-budget-aware recovery-interval run

The original quick overlay used the same `dt` for plotting and recovery. A 0.2 us recovery cadence is too aggressive for a protocol that requires several one- and two-qubit gates, so the added script separates plot sampling from recovery cadence.

```bash
python realistic_recovery_interval_simulation.py \
  --time-us 20 \
  --plot-dt-us 0.1 \
  --recovery-interval-us 5.0
```

This writes:

- `outputs/qec_bell_fidelity_periodic_realistic_interval.png`
- `outputs/qec_bell_fidelity_periodic_realistic_interval.svg`
- `outputs/qec_bell_fidelity_periodic_realistic_interval.csv`

The recovery operation is still modeled as an ideal instantaneous CPTP map. The longer interval is a cadence-level correction; finite gate Hamiltonians, gate errors, and measurement latency are not yet included.

## One-recovery-operation plot

The same script also generates a trace with exactly one recovery operation. By default it applies the single recovery at half of the total time window. You can set it explicitly:

```bash
python realistic_recovery_interval_simulation.py \
  --time-us 20 \
  --plot-dt-us 0.1 \
  --single-recovery-time-us 10.0
```

This writes:

- `outputs/qec_bell_fidelity_single_recovery.png`
- `outputs/qec_bell_fidelity_single_recovery.svg`
- `outputs/qec_bell_fidelity_single_recovery.csv`

## Original compact run

```bash
python nv_5level_two_c13_simulation.py
```

Useful options:

```bash
# Pick explicit 1-based DQP spin indices
python nv_5level_two_c13_simulation.py --spin-indices 1 5

# Choose the positive pair closest to the Bell/DFS limit
python nv_5level_two_c13_simulation.py --selection closest-positive --time-us 100 --dt-us 0.5
```

## Dependencies

```bash
pip install -r requirements.txt
```
