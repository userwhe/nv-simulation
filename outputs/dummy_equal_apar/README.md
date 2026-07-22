# Equal-Aparallel dummy-pair sanity check

Input physical hyperfine values:

| Spin | A_parallel (kHz) | A_perp (kHz) | JSON -A_parallel (kHz) |
|---|---:|---:|---:|
| E80p10 | +80 | 10 | -80 |
| E80p0 | +80 | 0 | -80 |

Both signed longitudinal code couplings are equal. After the code's global sign normalization, g1=g2=+80 kHz and g2/g1=+1. The n=2 code therefore reduces to the exact longitudinal decoherence-free subspace; the unequal A_perp values deliberately break the full Hamiltonian symmetry.

Simulation settings: B=0.05 T, pump rate=5/us, total time=20 us, ideal recovery at 10 and 20 us, 0.5 us sampling, and phase-calibrated six-state average fidelity.

## Results

| Excited scale | Final encoded/DFS | Final E80p10 | Final E80p0 | Final best single | Final encoded-best | Maximum encoded-best | Time of maximum (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.9999286037 | 0.9792696305 | 0.9793648952 | 0.9793648952 | +0.0205637085 | +0.0205637085 | 20.000 |
| 0.5 | 0.9999270655 | 0.9788687661 | 0.9789652894 | 0.9789652894 | +0.0209617761 | +0.0209617761 | 20.000 |

## Plots

- [r=0.1 full fidelity plot](r0p1/E80p10_E80p0_ideal_process.svg)
- [r=0.5 full fidelity plot](r0p5/E80p10_E80p0_ideal_process.svg)
- [Encoded/DFS-minus-best comparison](excited_scale_comparison.svg)

Because |g1|=|g2| with the same sign, this is primarily a DFS symmetry test rather than a distinct active-QEC test. The A_perp mismatch probes how noncommuting transverse dynamics spoil that symmetry.

These are results of the repository's simplified five-level optical model, not an experimental prediction.
