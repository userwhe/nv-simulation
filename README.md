# nv-simulation

NV-center dynamics simulations with a five-level electron model and two coupled `13C` nuclear spins.

This version adds a two-qubit common-fluctuator dephasing QEC simulation based on the `n = 2` code of Layden, Chen, and Cappellaro. The script can:

- parse DQP spin pairs stored as `(-A_parallel, A_perp)` in kHz;
- select two spins automatically or by explicit index;
- convert the selected hyperfine parameters to the five-level Lindblad model;
- run the physical Bell-state evolution without QEC;
- encode the Bell state into the two-qubit QEC code, apply recovery after each interval, decode, and report the Bell-state fidelity;
- save an overlay plot and CSV data.

## Default run

```bash
python nv_5level_two_c13_simulation.py
```

By default, the script chooses the two strongest DQP spins by `|-A_parallel|`, uses a 20 us simulation window, and applies a QEC recovery every 0.2 us.

The default selected spins are:

| DQP spin | `-A_parallel` (kHz) | `A_perp` (kHz) |
|---:|---:|---:|
| 2 | 224.2991967686721 | 199.07207268788648 |
| 3 | 50.10705093918639 | 108.7263051655999 |

The generated overlay from this default run is in `outputs/qec_bell_fidelity_overlay.svg`; the numeric data are in `outputs/qec_bell_fidelity_overlay.csv`.

## Useful options

```bash
# Pick explicit 1-based DQP spin indices
python nv_5level_two_c13_simulation.py --spin-indices 1 5

# Choose the positive pair closest to the Bell/DFS limit
python nv_5level_two_c13_simulation.py --selection closest-positive --time-us 100 --dt-us 0.5

# Skip checks for a faster plotting run
python nv_5level_two_c13_simulation.py --skip-checks
```

## Dependencies

```bash
pip install -r requirements.txt
```
