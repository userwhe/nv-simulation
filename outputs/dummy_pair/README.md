# Dummy-pair sanity check

Input physical hyperfine values:

| Spin | A_parallel (kHz) | A_perp (kHz) | JSON -A_parallel (kHz) |
|---|---:|---:|---:|
| Dm50 | -50 | 10 | +50 |
| Dp80 | +80 | 9 | -80 |

The code orders Dp80 first because |80|>|50|, then globally flips the coupling signs, giving the signed code ratio g2/g1=-50/80=-0.625.

Simulation settings: B=0.05 T, pump rate=5/us, total time=20 us, ideal recovery at 10 and 20 us, 0.5 us sampling, and phase-calibrated six-state average fidelity.

## Results

| Excited scale | Final QEC | Final best single | Final QEC-best | Maximum QEC-best | Time of maximum (us) |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.9985816130 | 0.9915035369 | +0.0070780762 | +0.0070780762 | 20.000 |
| 0.5 | 0.9985303785 | 0.9913207990 | +0.0072095795 | +0.0072095795 | 20.000 |

## Plots

- [r=0.1 full fidelity plot](r0p1/Dp80_Dm50_ideal_process.svg)
- [r=0.5 full fidelity plot](r0p5/Dp80_Dm50_ideal_process.svg)
- [QEC-minus-best comparison](excited_scale_comparison.svg)

These are results of the repository's simplified five-level optical model, not an experimental prediction.
