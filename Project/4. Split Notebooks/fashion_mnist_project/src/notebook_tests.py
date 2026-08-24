"""
notebook_tests.py – Inline unit test suite for the Fashion-MNIST pipeline.

Contains:
  - SkipTest exception class
  - run_test_suite(): test runner
  - All test_* functions for data, tensors, models, inference, bookkeeping
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class SkipTest(Exception):
    """Raised by a test whose prerequisites are absent."""
    pass


def run_test_suite(tests: Sequence[Callable[[], None]], verbose: bool = True) -> pd.DataFrame:
    """Execute every test function, catching failures."""
    rows = []
    for fn in tests:
        t0 = time.time()
        try:
            fn()
            status, message = "PASS", (fn.__doc__ or "").strip().split("\n")[0]
        except SkipTest as exc:
            status, message = "SKIP", str(exc)
        except AssertionError as exc:
            status, message = "FAIL", f"AssertionError: {exc}"
        except Exception as exc:
            status, message = "ERROR", f"{type(exc).__name__}: {exc}"
        rows.append({"test": fn.__name__, "status": status, "detail": message,
                     "seconds": round(time.time() - t0, 3)})
        if verbose:
            symbol = {"PASS": "PASS ", "SKIP": "SKIP ", "FAIL": "FAIL ", "ERROR": "ERROR"}[status]
            print(f"[{symbol}] {fn.__name__:<46s} {rows[-1]['seconds']:>6.2f}s  {message[:70]}")
    return pd.DataFrame(rows)
