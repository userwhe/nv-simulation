"""Compatibility entry point for the explicit pulsed electron-ancilla model."""

from nv_5level_two_c13_simulation import run_cli


if __name__ == "__main__":
    run_cli(default_protocol="pulsed")
