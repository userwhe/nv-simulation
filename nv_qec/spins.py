"""Signed hyperfine parameters for the seven-spin register."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SpinParameters:
    """Hyperfine parameters for one nuclear spin, stored in kHz."""

    id: int
    a_parallel_khz: float
    a_perp_khz: float

    def __post_init__(self) -> None:
        if not isinstance(self.id, int) or isinstance(self.id, bool):
            raise ValueError("spin id must be an integer")
        if not math.isfinite(self.a_parallel_khz) or not math.isfinite(self.a_perp_khz):
            raise ValueError("spin couplings must be finite")


BUILTIN_SPINS: dict[int, SpinParameters] = {
    1: SpinParameters(1, -5.616, 32.847),
    2: SpinParameters(2, -224.217, 189.318),
    3: SpinParameters(3, -49.837, 101.007),
    4: SpinParameters(4, -15.734, 19.295),
    5: SpinParameters(5, -3.832, 21.140),
    6: SpinParameters(6, 7.015, 38.250),
    7: SpinParameters(7, -33.962, 26.000),
}


__all__ = ["BUILTIN_SPINS", "SpinParameters"]
