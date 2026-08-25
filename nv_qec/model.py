"""Five-level NV electron and two-nuclear-spin Lindblad dynamics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .spins import SpinParameters


ELECTRON_DIMENSION = 5
NUCLEAR_DIMENSION = 4

_CARBON13_GYROMAGNETIC_RATIO_MHZ_PER_T = 10.705
_RAD_PER_US_PER_MHZ = 2.0 * np.pi
_RAD_PER_US_PER_KHZ = 1.0e-3 * _RAD_PER_US_PER_MHZ

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


@dataclass(frozen=True)
class OpticalParameters:
    """Five-level optical rates and the laser-power calibration."""

    laser_power_uw: float = 5.0
    excitation_rate_per_us_per_uw: float = 1.0
    radiative_e0_rate_per_us: float = (1.0 - 0.14) / 0.013
    radiative_e1_rate_per_us: float = (1.0 - 0.55) / 0.007
    isc_e0_rate_per_us: float = 0.14 / 0.013
    isc_e1_rate_per_us: float = 0.55 / 0.007
    singlet_to_g0_rate_per_us: float = 0.85 / 0.180
    singlet_to_g1_rate_per_us: float = (1.0 - 0.85) / 0.180

    def __post_init__(self) -> None:
        for field_name in (
            "laser_power_uw",
            "excitation_rate_per_us_per_uw",
            "radiative_e0_rate_per_us",
            "radiative_e1_rate_per_us",
            "isc_e0_rate_per_us",
            "isc_e1_rate_per_us",
            "singlet_to_g0_rate_per_us",
            "singlet_to_g1_rate_per_us",
        ):
            value = getattr(self, field_name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative")

    @property
    def pump_rate_per_us(self) -> float:
        return self.laser_power_uw * self.excitation_rate_per_us_per_uw

    @property
    def rates_per_us(self) -> dict[str, float]:
        return {
            "optical_pump": self.pump_rate_per_us,
            "radiative_e0": self.radiative_e0_rate_per_us,
            "radiative_e1": self.radiative_e1_rate_per_us,
            "intersystem_crossing_e0": self.isc_e0_rate_per_us,
            "intersystem_crossing_e1": self.isc_e1_rate_per_us,
            "singlet_to_g0": self.singlet_to_g0_rate_per_us,
            "singlet_to_g1": self.singlet_to_g1_rate_per_us,
        }


@dataclass(frozen=True)
class NVModel:
    """Lab-frame generator inputs in the internal rad/us convention."""

    hamiltonian_lab: np.ndarray
    collapse_operators: tuple[np.ndarray, ...]
    nuclear_free_hamiltonian: np.ndarray
    optical: OpticalParameters
    electron_dimension: int = ELECTRON_DIMENSION
    nuclear_dimension: int = NUCLEAR_DIMENSION

    @property
    def dimension(self) -> int:
        return self.electron_dimension * self.nuclear_dimension


def _basis_state(dimension: int, index: int) -> np.ndarray:
    state = np.zeros(dimension, dtype=complex)
    state[index] = 1.0
    return state


def _transition(dimension: int, final: int, initial: int) -> np.ndarray:
    return np.outer(
        _basis_state(dimension, final),
        _basis_state(dimension, initial).conj(),
    )


def hermitian_part(matrix: np.ndarray) -> np.ndarray:
    """Remove roundoff-scale anti-Hermitian drift."""

    return 0.5 * (matrix + matrix.conj().T)


def _convert_inputs_to_angular_frequency(
    spin_a: SpinParameters,
    spin_b: SpinParameters,
    b_field_t: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Central input boundary for MHz, kHz, and 2π conversions."""

    larmor = (
        _RAD_PER_US_PER_MHZ
        * _CARBON13_GYROMAGNETIC_RATIO_MHZ_PER_T
        * float(b_field_t)
    )
    a_parallel = _RAD_PER_US_PER_KHZ * np.array(
        [spin_a.a_parallel_khz, spin_b.a_parallel_khz], dtype=float
    )
    a_perp = _RAD_PER_US_PER_KHZ * np.array(
        [spin_a.a_perp_khz, spin_b.a_perp_khz], dtype=float
    )
    return larmor, a_parallel, a_perp


def build_nv_model(
    spin_a: SpinParameters,
    spin_b: SpinParameters,
    *,
    b_field_t: float,
    optical: OpticalParameters,
) -> NVModel:
    """Build the simplified five-electron-level/two-spin lab-frame model."""

    if spin_a.id == spin_b.id:
        raise ValueError("spin ids must be distinct")
    if not np.isfinite(b_field_t):
        raise ValueError("b_field_t must be finite")
    if not isinstance(optical, OpticalParameters):
        raise TypeError("optical must be an OpticalParameters instance")

    larmor, a_parallel, a_perp = _convert_inputs_to_angular_frequency(
        spin_a, spin_b, b_field_t
    )
    nuclear_ix = (
        np.kron(_X / 2.0, _I2),
        np.kron(_I2, _X / 2.0),
    )
    nuclear_iz = (
        np.kron(_Z / 2.0, _I2),
        np.kron(_I2, _Z / 2.0),
    )

    free_hamiltonian = larmor * (nuclear_iz[0] + nuclear_iz[1])
    coupled_hamiltonian = sum(
        (larmor - a_parallel[index]) * nuclear_iz[index]
        + a_perp[index] * nuclear_ix[index]
        for index in range(2)
    )

    g0, g1, e0, e1, singlet = range(ELECTRON_DIMENSION)
    hamiltonian = np.zeros(
        (ELECTRON_DIMENSION * NUCLEAR_DIMENSION,) * 2,
        dtype=complex,
    )
    # Excited and singlet hyperfine tensors are outside this simplified model.
    for electron_state, nuclear_hamiltonian in (
        (g0, free_hamiltonian),
        (g1, coupled_hamiltonian),
        (e0, free_hamiltonian),
        (e1, free_hamiltonian),
        (singlet, free_hamiltonian),
    ):
        projector = _transition(ELECTRON_DIMENSION, electron_state, electron_state)
        hamiltonian += np.kron(projector, nuclear_hamiltonian)

    rates = optical.rates_per_us

    nuclear_identity = np.eye(NUCLEAR_DIMENSION, dtype=complex)

    def collapse(rate: float, final: int, initial: int) -> np.ndarray:
        electron_transition = _transition(ELECTRON_DIMENSION, final, initial)
        return np.sqrt(rate) * np.kron(electron_transition, nuclear_identity)

    collapse_operators = (
        collapse(rates["optical_pump"], e0, g0),
        collapse(rates["optical_pump"], e1, g1),
        collapse(rates["radiative_e0"], g0, e0),
        collapse(rates["radiative_e1"], g1, e1),
        collapse(rates["intersystem_crossing_e0"], singlet, e0),
        collapse(rates["intersystem_crossing_e1"], singlet, e1),
        collapse(rates["singlet_to_g0"], g0, singlet),
        collapse(rates["singlet_to_g1"], g1, singlet),
    )
    return NVModel(
        hamiltonian_lab=hamiltonian,
        collapse_operators=collapse_operators,
        nuclear_free_hamiltonian=free_hamiltonian,
        optical=optical,
    )


def lindblad_generator(model: NVModel) -> sp.csr_matrix:
    """Return L such that d vec(rho)/dt = L vec(rho), using column stacking."""

    hamiltonian = sp.csr_matrix(model.hamiltonian_lab)
    identity = sp.identity(model.dimension, format="csr", dtype=complex)
    generator = -1j * (
        sp.kron(identity, hamiltonian, format="csr")
        - sp.kron(hamiltonian.T, identity, format="csr")
    )
    for collapse_operator in model.collapse_operators:
        collapse = sp.csr_matrix(collapse_operator)
        rate_operator = collapse.getH() @ collapse
        generator += sp.kron(collapse.conjugate(), collapse, format="csr")
        generator -= 0.5 * sp.kron(identity, rate_operator, format="csr")
        generator -= 0.5 * sp.kron(rate_operator.T, identity, format="csr")
    return generator.tocsr()


def propagate_states(
    generator: sp.csr_matrix,
    states: Sequence[np.ndarray],
    duration_us: float,
    dimension: int,
) -> list[np.ndarray]:
    """Propagate a density-matrix batch for one lab-frame interval."""

    if not np.isfinite(duration_us) or duration_us < -1.0e-12:
        raise ValueError("propagation duration must be finite and nonnegative")
    if not states:
        return []
    if duration_us <= 1.0e-15:
        return [hermitian_part(np.asarray(state, dtype=complex)) for state in states]

    vectors = np.column_stack(
        [np.asarray(state, dtype=complex).reshape(-1, order="F") for state in states]
    )
    evolved = spla.expm_multiply(generator * float(duration_us), vectors)
    return [
        hermitian_part(evolved[:, index].reshape((dimension, dimension), order="F"))
        for index in range(evolved.shape[1])
    ]


def nuclear_free_unitary(model: NVModel, absolute_time_us: float) -> np.ndarray:
    """R(t) = exp[-i H_free t] at the absolute simulation time."""

    return la.expm(-1j * model.nuclear_free_hamiltonian * float(absolute_time_us))


def to_nuclear_rotating_frame(
    nuclear_state_lab: np.ndarray,
    model: NVModel,
    absolute_time_us: float,
) -> np.ndarray:
    """Return R(t)† rho_lab R(t)."""

    frame = nuclear_free_unitary(model, absolute_time_us)
    return hermitian_part(frame.conj().T @ nuclear_state_lab @ frame)


def rotating_nuclear_operator_to_lab(
    operator_rotating: np.ndarray,
    model: NVModel,
    absolute_time_us: float,
) -> np.ndarray:
    """Return R(t) O_rot R(t)†."""

    frame = nuclear_free_unitary(model, absolute_time_us)
    return frame @ operator_rotating @ frame.conj().T


def trace_out_electron(full_state: np.ndarray, model: NVModel) -> np.ndarray:
    reshaped = full_state.reshape(
        model.electron_dimension,
        model.nuclear_dimension,
        model.electron_dimension,
        model.nuclear_dimension,
    )
    return hermitian_part(np.trace(reshaped, axis1=0, axis2=2))


def reduced_nuclear_spin(nuclear_state: np.ndarray, spin_index: int) -> np.ndarray:
    """Trace out the other spin; spin_index follows the caller's input order."""

    reshaped = nuclear_state.reshape(2, 2, 2, 2)
    if spin_index == 0:
        return hermitian_part(np.trace(reshaped, axis1=1, axis2=3))
    if spin_index == 1:
        return hermitian_part(np.trace(reshaped, axis1=0, axis2=2))
    raise ValueError("spin_index must be 0 or 1")


__all__ = [
    "ELECTRON_DIMENSION",
    "NUCLEAR_DIMENSION",
    "NVModel",
    "OpticalParameters",
    "build_nv_model",
    "hermitian_part",
    "lindblad_generator",
    "nuclear_free_unitary",
    "propagate_states",
    "reduced_nuclear_spin",
    "rotating_nuclear_operator_to_lab",
    "to_nuclear_rotating_frame",
    "trace_out_electron",
]
