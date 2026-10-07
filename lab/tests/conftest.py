"""Tests run in quick mode (tiny budgets, separately cached teachers) unless
LAB_QUICK=0 is set explicitly."""

import os

os.environ.setdefault("LAB_QUICK", "1")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
