# All-pair results with excited-state hyperfine scale r = 0.5

These artifacts use the proportional excited-state ansatz

```text
A_parallel,e = 0.5 * A_parallel,g
A_perp,e     = 0.5 * A_perp,g
```

for the excited `m_s=-1` manifold. The `e0` and singlet manifolds retain free nuclear precession.

## Simulation settings

- laboratory-frame five-level model;
- `B = 0.05 T`;
- optical pump rate `5 / us`;
- total evolution time `20 us`;
- ideal instantaneous recovery at `10 us` and `20 us`;
- output every `0.5 us`;
- six-state average fidelity with an independently calibrated final virtual-Z phase for each channel.

## Artifacts

- [All 21 pair curves](all_pairs_grid.svg)
- [Ranking against the best constituent spin](all_pairs_ranking.svg)
- [Full numerical summary](all_pairs_summary.csv)

The directory also contains one SVG and one CSV for each of the 21 spin pairs.

## Main result

At `20 us`, **X2-S4** remains the only pair that beats the better constituent spin:

```text
QEC average fidelity          0.9988545432
best constituent spin        0.9986703791
QEC - best single            +0.0001841641
```

The advantage is `+0.01842` percentage points, so it remains a small numerical crossing rather than a robust experimental margin.

For **S3-X2**:

```text
QEC average fidelity          0.9907252544
best constituent spin        0.9951219253
QEC - best single            -0.0043966709
```

Compared with the `r=0.1` sweep, increasing the scale to `r=0.5` changes the curves only modestly at this operating point. The X2-S4 crossing increases from approximately `+0.01796` to `+0.01842` percentage points, while S3-X2 moves slightly farther below the better physical spin.

## Reproduce

```bash
python simulate_all_pairs.py \
  --force \
  --excited-hyperfine-scale 0.5 \
  --output-dir outputs/all_pairs_r0p5
```

The proportional excited-state tensor is a sensitivity-analysis ansatz, not a substitute for a measured excited-state hyperfine tensor or an experimentally reconstructed optical nuclear channel.
