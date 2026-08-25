from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nv_qec.model import (
    NVModel,
    OpticalParameters,
    build_nv_model,
    lindblad_generator,
    nuclear_free_unitary,
    propagate_states,
    to_nuclear_rotating_frame,
    trace_out_electron,
)
from nv_qec.qec import (
    apply_recovery_in_lab_frame,
    build_common_fluctuator_code,
    code_design_coupling_khz,
    common_fluctuator_error_hamiltonian,
    density_matrix,
    encode_logical_density,
    six_state_average_memory_fidelity,
    six_state_test_states,
)
from nv_qec.simulation import (
    SimulationConfig,
    save_power_sweep_csv,
    save_power_sweep_plot,
    simulate_laser_power_sweep,
    simulate_qec_memory,
)
from nv_qec.spins import BUILTIN_SPINS, SpinParameters


class QECCodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spin_a = BUILTIN_SPINS[3]
        self.spin_b = BUILTIN_SPINS[7]
        self.code = build_common_fluctuator_code(self.spin_a, self.spin_b)

    def test_logical_and_error_basis_is_orthonormal(self) -> None:
        basis = np.column_stack((*self.code.logical_basis, *self.code.error_basis))
        np.testing.assert_allclose(basis.conj().T @ basis, np.eye(4), atol=1.0e-13)

    def test_recovery_kraus_operators_are_cptp(self) -> None:
        completeness = sum(
            kraus.conj().T @ kraus for kraus in self.code.recovery_kraus
        )
        np.testing.assert_allclose(completeness, np.eye(4), atol=1.0e-13)

    def test_common_fluctuator_identities_hold(self) -> None:
        error_hamiltonian = common_fluctuator_error_hamiltonian(
            self.spin_a, self.spin_b
        )
        isometry = self.code.logical_isometry
        np.testing.assert_allclose(
            isometry.conj().T @ error_hamiltonian @ isometry,
            np.zeros((2, 2)),
            atol=1.0e-12,
        )

        couplings = sorted(
            (
                abs(code_design_coupling_khz(self.spin_a)),
                abs(code_design_coupling_khz(self.spin_b)),
            ),
            reverse=True,
        )
        error_strength_squared = couplings[0] ** 2 - couplings[1] ** 2
        np.testing.assert_allclose(
            isometry.conj().T
            @ error_hamiltonian
            @ error_hamiltonian
            @ isometry,
            error_strength_squared * np.eye(2),
            atol=1.0e-10,
        )

    def test_analytic_virtual_z_correction_finds_continuous_optimum(self) -> None:
        targets = six_state_test_states()
        imposed_phase = 0.271234
        phase_rotation = np.diag(
            [
                np.exp(-0.5j * imposed_phase),
                np.exp(0.5j * imposed_phase),
            ]
        )
        outputs = [
            phase_rotation @ density_matrix(target) @ phase_rotation.conj().T
            for target in targets
        ]
        fidelity = six_state_average_memory_fidelity(outputs, targets)
        self.assertLess(fidelity.raw, 1.0)
        self.assertAlmostEqual(fidelity.phase_corrected, 1.0, places=12)
        self.assertAlmostEqual(fidelity.correction_phase_rad, -imposed_phase, places=12)

    def test_leakage_is_counted_without_renormalization(self) -> None:
        targets = six_state_test_states()
        outputs = [0.37 * density_matrix(target) for target in targets]
        fidelity = six_state_average_memory_fidelity(outputs, targets)
        self.assertAlmostEqual(fidelity.raw, 0.37, places=12)
        self.assertAlmostEqual(fidelity.phase_corrected, 0.37, places=12)


class FrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spin_a = BUILTIN_SPINS[3]
        self.spin_b = BUILTIN_SPINS[7]

    @staticmethod
    def _electron_ground_state(model: NVModel) -> np.ndarray:
        electron = np.zeros(model.electron_dimension, dtype=complex)
        electron[0] = 1.0
        return density_matrix(electron)

    def test_zero_transverse_coupling_matches_static_rotating_frame(self) -> None:
        spin_a = SpinParameters(101, -49.837, 0.0)
        spin_b = SpinParameters(102, -33.962, 0.0)
        model = build_nv_model(
            spin_a,
            spin_b,
            b_field_t=0.05,
            optical=OpticalParameters(laser_power_uw=2.0),
        )
        rotating_hamiltonian = model.hamiltonian_lab - np.kron(
            np.eye(model.electron_dimension), model.nuclear_free_hamiltonian
        )
        rotating_model = NVModel(
            hamiltonian_lab=rotating_hamiltonian,
            collapse_operators=model.collapse_operators,
            nuclear_free_hamiltonian=model.nuclear_free_hamiltonian,
            optical=model.optical,
        )

        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
        initial_nuclear = density_matrix(np.kron(plus, plus))
        initial = np.kron(self._electron_ground_state(model), initial_nuclear)
        time_us = 0.73
        lab = propagate_states(
            lindblad_generator(model), [initial], time_us, model.dimension
        )[0]
        rotating = propagate_states(
            lindblad_generator(rotating_model),
            [initial],
            time_us,
            model.dimension,
        )[0]
        frame = np.kron(
            np.eye(model.electron_dimension), nuclear_free_unitary(model, time_us)
        )
        lab_in_rotating_frame = frame.conj().T @ lab @ frame
        np.testing.assert_allclose(lab_in_rotating_frame, rotating, atol=2.0e-9)

    def test_lab_recovery_matches_rotating_recovery_at_nonzero_time(self) -> None:
        model = build_nv_model(
            self.spin_a,
            self.spin_b,
            b_field_t=0.05,
            optical=OpticalParameters(laser_power_uw=2.0),
        )
        code = build_common_fluctuator_code(self.spin_a, self.spin_b)
        rng = np.random.default_rng(20260824)
        vector = rng.normal(size=model.dimension) + 1j * rng.normal(
            size=model.dimension
        )
        vector /= np.linalg.norm(vector)
        state_rotating = density_matrix(vector)
        time_us = 0.43
        frame = np.kron(
            np.eye(model.electron_dimension), nuclear_free_unitary(model, time_us)
        )
        state_lab = frame @ state_rotating @ frame.conj().T

        actual = apply_recovery_in_lab_frame(state_lab, code, model, time_us)
        expected_rotating = np.zeros_like(state_rotating)
        for nuclear_kraus in code.recovery_kraus:
            full_kraus = np.kron(
                np.eye(model.electron_dimension), nuclear_kraus
            )
            expected_rotating += (
                full_kraus @ state_rotating @ full_kraus.conj().T
            )
        expected = frame @ expected_rotating @ frame.conj().T
        np.testing.assert_allclose(actual, expected, atol=2.0e-12)

    def test_split_propagation_uses_the_absolute_frame_clock(self) -> None:
        model = build_nv_model(
            self.spin_a,
            self.spin_b,
            b_field_t=0.05,
            optical=OpticalParameters(laser_power_uw=2.0),
        )
        code = build_common_fluctuator_code(self.spin_a, self.spin_b)
        generator = lindblad_generator(model)
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
        encoded = encode_logical_density(density_matrix(plus), code)
        initial = np.kron(self._electron_ground_state(model), encoded)
        first_interval = 0.17
        second_interval = 0.23

        one_interval = propagate_states(
            generator,
            [initial],
            first_interval + second_interval,
            model.dimension,
        )[0]
        split = propagate_states(
            generator, [initial], first_interval, model.dimension
        )[0]
        split = propagate_states(
            generator, [split], second_interval, model.dimension
        )[0]
        np.testing.assert_allclose(split, one_interval, atol=2.0e-12)

        after_first_recovery = apply_recovery_in_lab_frame(
            propagate_states(
                generator, [initial], first_interval, model.dimension
            )[0],
            code,
            model,
            first_interval,
        )
        before_second_recovery = propagate_states(
            generator,
            [after_first_recovery],
            second_interval,
            model.dimension,
        )[0]
        correct = apply_recovery_in_lab_frame(
            before_second_recovery,
            code,
            model,
            first_interval + second_interval,
        )
        restarted_clock = apply_recovery_in_lab_frame(
            before_second_recovery,
            code,
            model,
            second_interval,
        )
        self.assertGreater(np.max(np.abs(correct - restarted_clock)), 1.0e-3)

        split_nuclear = trace_out_electron(split, model)
        one_interval_nuclear = trace_out_electron(one_interval, model)
        total_time = first_interval + second_interval
        np.testing.assert_allclose(
            to_nuclear_rotating_frame(split_nuclear, model, total_time),
            to_nuclear_rotating_frame(one_interval_nuclear, model, total_time),
            atol=2.0e-12,
        )


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spin_a = BUILTIN_SPINS[3]
        self.spin_b = BUILTIN_SPINS[7]

    def test_noise_disabled_preserves_all_raw_fidelities(self) -> None:
        result = simulate_qec_memory(
            SimulationConfig(
                duration_us=0.2,
                sample_dt_us=0.1,
                b_field_t=0.05,
                optical=OpticalParameters(laser_power_uw=0.0),
            ),
            self.spin_a,
            self.spin_b,
        )
        for values in (result.qec_raw, result.spin_a_raw, result.spin_b_raw):
            np.testing.assert_allclose(values, np.ones_like(values), atol=2.0e-10)

    def test_reversing_inputs_preserves_physics_and_labels(self) -> None:
        config = SimulationConfig(
            duration_us=0.08,
            sample_dt_us=0.04,
            b_field_t=0.05,
            optical=OpticalParameters(laser_power_uw=2.0),
        )
        forward = simulate_qec_memory(config, self.spin_a, self.spin_b)
        reversed_result = simulate_qec_memory(config, self.spin_b, self.spin_a)
        self.assertEqual((forward.spin_a.id, forward.spin_b.id), (3, 7))
        self.assertEqual(
            (reversed_result.spin_a.id, reversed_result.spin_b.id), (7, 3)
        )
        np.testing.assert_allclose(
            forward.qec_phase_corrected,
            reversed_result.qec_phase_corrected,
            atol=2.0e-11,
        )
        np.testing.assert_allclose(
            forward.spin_a_phase_corrected,
            reversed_result.spin_b_phase_corrected,
            atol=2.0e-11,
        )
        np.testing.assert_allclose(
            forward.spin_b_phase_corrected,
            reversed_result.spin_a_phase_corrected,
            atol=2.0e-11,
        )

    def test_readout_recovery_is_final_only_and_does_not_mutate_trajectory(self) -> None:
        sampled = simulate_qec_memory(
            SimulationConfig(
                duration_us=0.08,
                sample_dt_us=0.04,
                b_field_t=0.05,
                optical=OpticalParameters(laser_power_uw=2.0),
            ),
            self.spin_a,
            self.spin_b,
        )
        coarse = simulate_qec_memory(
            SimulationConfig(
                duration_us=0.08,
                sample_dt_us=0.08,
                b_field_t=0.05,
                optical=OpticalParameters(laser_power_uw=2.0),
            ),
            self.spin_a,
            self.spin_b,
        )

        # Reading at 0.04 us must not alter the state continued to 0.08 us.
        self.assertAlmostEqual(
            sampled.qec_phase_corrected[-1],
            coarse.qec_phase_corrected[-1],
            places=11,
        )

    def test_nonintegral_sampling_includes_exact_final_duration(self) -> None:
        result = simulate_qec_memory(
            SimulationConfig(
                duration_us=0.11,
                sample_dt_us=0.04,
                b_field_t=0.05,
                optical=OpticalParameters(laser_power_uw=0.0),
            ),
            self.spin_a,
            self.spin_b,
        )
        np.testing.assert_allclose(result.t_us, [0.0, 0.04, 0.08, 0.11])
        self.assertEqual(result.t_us[-1], result.config.duration_us)

    def test_laser_power_sweep_uses_zero_perpendicular_spins(self) -> None:
        spin_a = SpinParameters(2, BUILTIN_SPINS[2].a_parallel_khz, 0.0)
        spin_b = SpinParameters(7, BUILTIN_SPINS[7].a_parallel_khz, 0.0)
        config = SimulationConfig(
            duration_us=0.04,
            sample_dt_us=0.02,
            b_field_t=0.05,
            optical=OpticalParameters(laser_power_uw=1.0),
        )
        results = simulate_laser_power_sweep(config, spin_a, spin_b, (0.1, 1.0))

        self.assertEqual(
            [result.config.optical.laser_power_uw for result in results],
            [0.1, 1.0],
        )
        for result in results:
            np.testing.assert_allclose(result.t_us, [0.0, config.duration_us])
            self.assertEqual(result.spin_a.a_perp_khz, 0.0)
            self.assertEqual(result.spin_b.a_perp_khz, 0.0)

        direct = simulate_qec_memory(config, spin_a, spin_b)
        self.assertAlmostEqual(
            results[-1].qec_phase_corrected[-1],
            direct.qec_phase_corrected[-1],
            places=11,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            csv_path = save_power_sweep_csv(results, directory / "sweep.csv")
            svg_path = save_power_sweep_plot(results, directory / "sweep.svg")
            self.assertTrue(csv_path.is_file())
            self.assertTrue(svg_path.is_file())
            self.assertIn("laser_power_uw", csv_path.read_text(encoding="utf-8"))
            svg_text = svg_path.read_text(encoding="utf-8")
            self.assertIn("Laser-power sweep", svg_text)
            self.assertIn("A⊥=0", svg_text)


class CommandLineTests(unittest.TestCase):
    def test_invalid_inputs_report_useful_errors(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        cases = (
            (["--spin-ids", "3", "3"], "two distinct IDs"),
            (["--spin-ids", "3", "99"], "unknown spin ID"),
            (
                ["--spin1-a-parallel-khz", "-49.837"],
                "all four direct spin parameters are required",
            ),
            (
                ["--spin-ids", "3", "7", "--duration-us", "0"],
                "duration_us must be finite and positive",
            ),
            (
                ["--spin-ids", "3", "7", "--sample-dt-us", "0"],
                "sample_dt_us must be finite and positive",
            ),
            (
                ["--spin-ids", "3", "7", "--laser-power-uw", "-1"],
                "laser_power_uw must be finite and nonnegative",
            ),
            (
                ["--all-pairs", "--spin-ids", "3", "7"],
                "cannot be combined with a spin-pair selection",
            ),
        )
        for arguments, expected_error in cases:
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    [sys.executable, str(repository / "run_simulation.py"), *arguments],
                    cwd=repository,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(expected_error, completed.stderr)

    def test_smoke_run_writes_expected_csv_and_svg(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            output_directory = temporary_path / "outputs"
            environment = os.environ.copy()
            environment["MPLBACKEND"] = "Agg"
            environment["MPLCONFIGDIR"] = str(temporary_path / "matplotlib")
            environment["XDG_CACHE_HOME"] = str(temporary_path / "cache")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(repository / "run_simulation.py"),
                    "--spin-ids",
                    "3",
                    "7",
                    "--duration-us",
                    "0.02",
                    "--sample-dt-us",
                    "0.02",
                    "--laser-power-uw",
                    "0",
                    "--output-dir",
                    str(output_directory),
                ],
                cwd=repository,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            csv_path = output_directory / "results.csv"
            svg_path = output_directory / "fidelity_comparison.svg"
            self.assertTrue(csv_path.is_file())
            self.assertTrue(svg_path.is_file())
            with csv_path.open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle))
            self.assertEqual(
                header,
                [
                    "t_us",
                    "qec_raw",
                    "qec_phase_corrected",
                    "spin_3_raw",
                    "spin_3_phase_corrected",
                    "spin_7_raw",
                    "spin_7_phase_corrected",
                    "best_single",
                    "qec_advantage",
                ],
            )
            self.assertIn("QEC fidelity", completed.stdout)
            self.assertIn("Benchmark: ideal recovery at readout only", completed.stdout)
            svg_text = svg_path.read_text(encoding="utf-8")
            self.assertIn("A∥=-49.837", svg_text)
            self.assertIn("A⊥=26", svg_text)
            self.assertNotIn("Best single-spin envelope", svg_text)


class OpticalParametersTests(unittest.TestCase):
    def test_power_calibration_and_rates_are_explicit(self) -> None:
        optical = OpticalParameters(
            laser_power_uw=4.0,
            excitation_rate_per_us_per_uw=0.25,
            radiative_e0_rate_per_us=2.0,
            radiative_e1_rate_per_us=3.0,
            isc_e0_rate_per_us=4.0,
            isc_e1_rate_per_us=5.0,
            singlet_to_g0_rate_per_us=6.0,
            singlet_to_g1_rate_per_us=7.0,
        )
        self.assertEqual(optical.pump_rate_per_us, 1.0)
        self.assertEqual(
            optical.rates_per_us,
            {
                "optical_pump": 1.0,
                "radiative_e0": 2.0,
                "radiative_e1": 3.0,
                "intersystem_crossing_e0": 4.0,
                "intersystem_crossing_e1": 5.0,
                "singlet_to_g0": 6.0,
                "singlet_to_g1": 7.0,
            },
        )
        model = build_nv_model(
            BUILTIN_SPINS[3],
            BUILTIN_SPINS[7],
            b_field_t=0.05,
            optical=optical,
        )
        self.assertIs(model.optical, optical)

    def test_negative_rate_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "isc_e1_rate_per_us must be finite and nonnegative",
        ):
            OpticalParameters(isc_e1_rate_per_us=-1.0)


if __name__ == "__main__":
    unittest.main()
