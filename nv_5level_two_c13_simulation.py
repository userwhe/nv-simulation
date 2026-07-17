"""Compatibility entry point for the corrected NV n=2 QEC simulations.

Implementation is split between :mod:`nv_qec_core` and
:mod:`nv_qec_protocols`; public names are re-exported here so existing imports
continue to work.
"""

from nv_qec_core import *
from nv_qec_protocols import *
from nv_qec_protocols import main

if __name__ == "__main__":
    main()
