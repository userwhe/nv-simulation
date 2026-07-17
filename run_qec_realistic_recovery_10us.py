"""Convenience run approximating a 10 us optical-noise/recovery cycle."""

from __future__ import annotations

import sys

from nv_5level_two_c13_simulation import run_cli


DEFAULT_ARGS = [
    "--protocol",
    "pulsed",
    "--cycles",
    "2",
    "--laser-on-us",
    "8",
    "--laser-settle-us",
    "0.25",
    "--syndrome-gate-us",
    "0.75",
    "--measurement-us",
    "0.5",
    "--feedback-latency-us",
    "0.5",
    "--correction-gate-us",
    "0.75",
]


if __name__ == "__main__":
    run_cli([*DEFAULT_ARGS, *sys.argv[1:]], default_protocol="pulsed")
