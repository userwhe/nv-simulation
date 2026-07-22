from __future__ import annotations

import unittest

import numpy as np

from nv_5level_two_c13_simulation import (
    DEFAULT_DQP_SPINS,
    apply_explicit_ancilla_recovery,
    apply_recovery_to_full_state,
    build_nv_two_c13_lab_model,
    density,
    hermitize,
    ket,
    lindblad_liouvillian,
    load_dqp_spins,
    make_nv_two_c13_model,
    make_two_qubit_qec_code,
    memory_test_states,
    model_params_from_spins,
    nuclear_free_unitary,
    partial_trace_electron,
    phase_calibrated_six_state_fidelity,
    propagate_state_batch,
    set_excited_state_hyperfine_from_ground,
    simulate_ideal_periodic_process,
)


class QECAlgebraTests(unittest.TestCase):
    def test_code_basis_and_recovery_are_valid(self) -> None:
        code = make_two_qubit_qec_code((0.049837, 0.033962))
        basis = np.column_stack([code.zero_L, code.one_L, code.zero_E, code.one_E])
        self.assertLess(np.max(np.abs(basis.conj().T @ basis - np.eye(4))), 1e-12)
        completeness = sum(K.conj().T @ K for K in code.recovery_kraus)
        self.assertLess(np.max(np.abs(completeness - np.eye(4))), 1e-12)

    def test_six_state_identity_fidelity_is_one(self) -> None:
        _, targets = memory_test_states()
        outputs = [density(state) for state in targets]
        raw, calibrated, _ = phase_calibrated_six_state_fidelity(outputs, targets)
        self.assertAlmostEqual(raw, 1.0, places=12)
        self.assertAlmostEqual(calibrated, 1.0, places=12)

    def test_explicit_ancilla_recovery_matches_kraus_map(self) -> None:
        code = make_two_qubit_qec_code((0.049837, 0.033962))
        model = build_nv_two_c13_lab_model()
        rng = np.random.default_rng(1234)
        vector = rng.normal(size=4) + 1j * rng.normal(size=4)
        vector /= np.linalg.norm(vector)
        rho_n = density(vector)
        rho = np.kron(density(ket(model.de, 0)), rho_n)

        explicit = apply_explicit_ancilla_recovery(rho, code, model, t_us=0.0)
        ideal = apply_recovery_to_full_state(rho, code, model.de)
        self.assertLess(
            np.max(
                np.abs(
                    partial_trace_electron(explicit, model.de, model.dn)
                    - partial_trace_electron(ideal, model.de, model.dn)
                )
            ),
            1e-11,
        )


class ExcitedStateHyperfineTests(unittest.TestCase):
    @staticmethod
    def electron_block(model, electron_index: int) -> np.ndarray:
        tensor = model.H_lab.reshape(model.de, model.dn, model.de, model.dn)
        return tensor[electron_index, :, electron_index, :]

    def test_scaling_helper_uses_default_point_one(self) -> None:
        par, perp = set_excited_state_hyperfine_from_ground((-0.05, 0.02), (0.1, 0.03))
        np.testing.assert_allclose(par, (-0.005, 0.002), atol=0.0, rtol=1e-14)
        np.testing.assert_allclose(perp, (0.01, 0.003), atol=0.0, rtol=1e-14)

    def test_excited_ms_minus_one_hyperfine_is_scaled_ground_tensor(self) -> None:
        scale = 0.23
        model = build_nv_two_c13_lab_model(
            A_par_MHz=(-0.049837, -0.033962),
            A_perp_MHz=(0.101007, 0.026),
            excited_state_hyperfine_scale=scale,
        )
        H_g1 = self.electron_block(model, 1)
        H_e1 = self.electron_block(model, 3)
        ground_hyperfine = H_g1 - model.H_free_n
        excited_hyperfine = H_e1 - model.H_free_n
        self.assertLess(np.max(np.abs(excited_hyperfine - scale * ground_hyperfine)), 2e-14)

    def test_zero_scale_recovers_free_excited_precession(self) -> None:
        model = build_nv_two_c13_lab_model(excited_state_hyperfine_scale=0.0)
        H_e1 = self.electron_block(model, 3)
        self.assertLess(np.max(np.abs(H_e1 - model.H_free_n)), 1e-14)

    def test_default_model_scale_is_point_one_and_propagates_from_spins(self) -> None:
        default_model = build_nv_two_c13_lab_model()
        explicit_model = build_nv_two_c13_lab_model(excited_state_hyperfine_scale=0.1)
        self.assertLess(np.max(np.abs(default_model.H_lab - explicit_model.H_lab)), 1e-14)
        spins = load_dqp_spins(None)[:2]
        params = model_params_from_spins(spins, 0.05, 5.0)
        self.assertAlmostEqual(params["excited_state_hyperfine_scale"], 0.1)

    def test_nonfinite_scale_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_nv_two_c13_lab_model(excited_state_hyperfine_scale=np.nan)


class FrameTests(unittest.TestCase):
    def test_static_interaction_picture_rejects_nonzero_transverse_coupling(self) -> None:
        with self.assertRaises(ValueError):
            make_nv_two_c13_model(A_perp_MHz=(0.01, 0.0), interaction_picture=True)

    def test_lab_and_static_rotating_frame_agree_when_Aperp_zero(self) -> None:
        kwargs = dict(
            B_T=0.05,
            A_par_MHz=(-0.049837, -0.033962),
            A_perp_MHz=(0.0, 0.0),
            W_pump_per_us=2.0,
            excited_state_hyperfine_scale=0.1,
        )
        model = build_nv_two_c13_lab_model(**kwargs)
        H_rot, collapse_rot, _, _ = make_nv_two_c13_model(**kwargs, interaction_picture=True)
        L_lab = lindblad_liouvillian(model.H_lab, model.collapse_ops)
        L_rot = lindblad_liouvillian(H_rot, collapse_rot)

        plus = (ket(2, 0) + ket(2, 1)) / np.sqrt(2)
        rho0 = np.kron(density(ket(model.de, 0)), density(np.kron(plus, plus)))
        time_us = 0.73
        lab = propagate_state_batch(L_lab, [rho0], time_us, model.dim)[0]
        rotating = propagate_state_batch(L_rot, [rho0], time_us, model.dim)[0]
        frame = np.kron(np.eye(model.de), nuclear_free_unitary(model, time_us))
        lab_in_rotating_frame = hermitize(frame.conj().T @ lab @ frame)
        self.assertLess(np.max(np.abs(lab_in_rotating_frame - rotating)), 2e-9)


class ProcessBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spins = load_dqp_spins(None)
        self.by_name = {spin.name: spin for spin in self.spins}

    def run_pair(self, first: str, second: str) -> dict:
        selected = sorted(
            [self.by_name[first], self.by_name[second]],
            key=lambda spin: abs(spin.g_MHz),
            reverse=True,
        )
        code = make_two_qubit_qec_code(tuple(spin.g_MHz for spin in selected))
        params = model_params_from_spins(
            selected,
            B_T=0.05,
            pump_rate=2.0,
            excited_state_hyperfine_scale=0.1,
        )
        return simulate_ideal_periodic_process([0.0, 0.25, 0.50], params, code, 0.50)

    def test_best_single_is_pointwise_maximum(self) -> None:
        result = self.run_pair("S3", "X2")
        expected = np.maximum(result["spin0"], result["spin1"])
        self.assertLess(np.max(np.abs(result["best_single"] - expected)), 1e-14)
        self.assertLess(
            np.max(np.abs(result["qec_minus_best"] - (result["qec"] - expected))),
            1e-14,
        )

    def test_pair_sweep_is_order_independent(self) -> None:
        first = self.run_pair("S3", "X2")
        _ = self.run_pair("S2", "S4")
        repeated = self.run_pair("S3", "X2")
        for key in ("qec", "spin0", "spin1", "best_single", "qec_minus_best"):
            self.assertLess(np.max(np.abs(first[key] - repeated[key])), 1e-12)

    def test_default_register_contains_user_supplied_x2(self) -> None:
        self.assertTrue(any(row["name"] == "X2" for row in DEFAULT_DQP_SPINS))


if __name__ == "__main__":
    unittest.main()
