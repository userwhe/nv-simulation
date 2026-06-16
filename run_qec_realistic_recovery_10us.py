"""Convenience entry point for the 10 us QEC recovery-cadence run.

This wrapper keeps the main realistic-recovery simulator configurable while
making the requested run reproducible with:

    python run_qec_realistic_recovery_10us.py

It applies ideal QEC recoveries every 10 us, uses a 0.5 us plot/data sampling
step, and also produces the exactly-one-recovery trace at t = 10 us.
"""

from __future__ import annotations

import sys

from realistic_recovery_interval_simulation import main

DEFAULT_ARGS = [
    "--recovery-interval-us", "10",
    "--plot-dt-us", "0.5",
    "--single-recovery-time-us", "10",
]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    main()
