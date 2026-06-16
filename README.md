# nv-simulation

NV-center dynamics simulations with a five-level electron model and two coupled `13C` nuclear spins.

This version includes a two-qubit common-fluctuator dephasing QEC simulation based on the `n = 2` hardware-efficient code of Layden, Chen, and Cappellaro.

## What changed in this run

The recovery cadence is no longer tied to the plot sampling interval. Earlier output used a 0.2 us recovery interval, which is too aggressive for a recovery circuit made from several one- and two-qubit operations. The script now separates:

- `--plot-dt-us`: output/plot sampling step only;
- `--recovery-interval-us`: periodic QEC recovery cadence;
- `--single-recovery-time-us`: time of one fixed recovery operation for the single-recovery plot.

The default periodic recovery interval is 10 us and the default plot sampling step is 0.5 us. The recovery map is still modeled as ideal and instantaneous; finite gate duration, gate infidelity, measurement latency, and feedback latency are not included.

## Default run

```bash
python nv_5level_two_c13_simulation.py
```

This runs a 20 us storage window, samples every 0.5 us, applies periodic ideal QEC recovery every 10 us, and also generates a separate trace with exactly one recovery at 10 us.

Default selected DQP spins, chosen by strongest `|-A_parallel|`:

| DQP spin | `-A_parallel` (kHz) | `A_perp` (kHz) |
|---:|---:|---:|
| 2 | 224.2991967686721 | 199.07207268788648 |
| 3 | 50.10705093918639 | 108.7263051655999 |

## Useful options

```bash
# Finer plot sampling, while keeping the same 10 us recovery cadence
python nv_5level_two_c13_simulation.py --recovery-interval-us 10 --plot-dt-us 0.25

# Less conservative 5 us periodic recovery cadence
python nv_5level_two_c13_simulation.py --recovery-interval-us 5

# Move the single recovery to 15 us
python nv_5level_two_c13_simulation.py --single-recovery-time-us 15

# Pick explicit 1-based DQP spin indices
python nv_5level_two_c13_simulation.py --spin-indices 1 5
```

## Outputs

The script writes:

- `qec_bell_fidelity_periodic_realistic_interval.png/.svg/.csv`
- `qec_bell_fidelity_single_recovery.png/.svg/.csv`

## Dependencies

```bash
pip install -r requirements.txt
```
