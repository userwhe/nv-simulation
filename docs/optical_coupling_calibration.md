# Calibrating the optical common-noise coupling vector

The `n = 2` code must be compiled from the **signed coupling ratio of the stochastic phase produced by the exact optical block**, not automatically from the dark/DDRF value of `-A_parallel`.

## 1. What the code assumes

The target model is

```text
U(theta) = exp[-i theta (g1 Z1 + g2 Z2)].
```

The random scalar `theta` changes from shot to shot, while the direction

```text
g_opt = (g1, g2)
```

is fixed. Only the signed ratio is needed for the code basis; the absolute scale is needed to choose the optical exposure and recovery cadence.

For `|g1| >= |g2|`, remove an irrelevant common sign so that `g1 > 0`, define

```text
r = g2 / g1,
a = sqrt((1-r)/2),
b = sqrt((1+r)/2),
```

and use

```text
chi0 = a|0> + b|1>
chi1 = -b|0> + a|1>
|0_L> = |chi0>|0>
|1_L> = |chi1>|1>.
```

The sign of `r` matters. A negative value means the two nuclear phases are anticorrelated. A simultaneous sign reversal of both components is irrelevant.

## 2. Why DDRF `A_parallel` is only a proxy

DDRF spectroscopy characterizes nuclear frequencies for selected ground-state electron manifolds. During an optical excitation/reset block, the NV can follow stochastic trajectories through ground, excited, singlet, and possibly different charge states. Each state can change the nuclear precession frequency and axis.

For one optical realization, the phase of nucleus `j` is an integrated quantity,

```text
phi_j = integral dt delta_omega_j[electron/charge trajectory(t)].
```

Therefore the ratio of random optical phases can differ from the ratio of the dark ground-state hyperfine values. The deterministic mean phase is not the quantity used to compile the code; it should be removed by calibration or a virtual-Z correction. The code uses the direction of the **shot-to-shot fluctuation**.

## 3. Covariance/eigenmode calibration

Run the exact optical block repeatedly with the same field, laser power, wavelength, pulse duration, charge preparation, and reset sequence planned for QEC. Estimate the random phase increments

```text
phi = (phi1, phi2)^T.
```

Subtract their mean and form

```text
C = E[(phi - mean(phi))(phi - mean(phi))^T].
```

For one dominant common fluctuator,

```text
C approximately equals sigma_theta^2 g_opt g_opt^T.
```

The principal eigenvector of `C` is consequently the signed optical coupling mode, up to an irrelevant overall sign and scale.

Useful diagnostics are

```text
rho = C12 / sqrt(C11 C22)
common_mode_fraction = lambda1 / (lambda1 + lambda2)
```

where `lambda1 >= lambda2` are the covariance eigenvalues. `|rho|` and the common-mode fraction close to one support the one-dimensional common-fluctuator model. A substantial second eigenvalue means that an independent phase-noise mode remains and is not corrected by this two-qubit code.

## 4. Obtaining the covariance without single-shot phase estimates

Single-spin Ramsey data give `C11` and `C22`. Two-spin sum- and difference-phase coherences give

```text
V_plus  = Var(phi1 + phi2) = C11 + C22 + 2 C12
V_minus = Var(phi1 - phi2) = C11 + C22 - 2 C12
C12 = (V_plus - V_minus) / 4.
```

For approximately Gaussian phase noise, a normalized coherence `L_k` for phase combination `k^T phi` satisfies

```text
Var(k^T phi) = -2 log(|L_k|).
```

It is generally more robust to repeat `N` identical optical blocks, fit the variance or `-2 log` visibility versus `N`, and use the fitted slopes. This separates the optical contribution from static SPAM contrast.

## 5. Why a ratio error is serious

Suppose the code is compiled with `r_hat`, but the actual optical mode is `(g1, g2)`. Within the nominal logical subspace,

```text
W^dagger (g1 Z1 + g2 Z2) W = (g2 - g1 r_hat) Z_L.
```

Thus a mismatch leaves a first-order logical phase term

```text
g1 (r_actual - r_hat) Z_L
```

up to sign convention. The quadratic physical dephasing that the code was intended to remove then reappears at leading order. This is why the optical ratio should be measured, not merely inferred from dark spectroscopy.

## 6. Measurements still needed beyond the covariance

The covariance method only characterizes commuting phase noise. Also measure:

- nuclear population changes after repeated optical blocks;
- `X/Y` error components, for example with the six Pauli input states;
- leakage and charge-state-conditioned dynamics;
- drift of `g_opt` over the full data-acquisition period.

Bootstrap the phase data to obtain uncertainty in `r`, propagate it to `a` and `b`, and include that uncertainty in the predicted break-even margin.

## 7. Utility script

For shot-resolved phase samples in radians, run

```bash
python calibrate_optical_mode.py phases.csv \
  --columns phi_S3 phi_X2 \
  --output optical_mode.json
```

The output contains the covariance matrix, correlation, eigenvalues, common-mode fraction, signed principal mode, ordered coupling ratio, and `a,b` code coefficients. Use the ordered signed mode as the input to `make_two_qubit_qec_code(...)`; preserve the corresponding physical-spin order.
