"""
VariableRangeTracker — traces decision_pipeline.py at runtime, records every
numeric scalar variable value, and writes per-variable history + min/max to
VariableRangeData.txt.

Uses sys.settrace so decision_pipeline.py is never modified.

Usage:
    python VariableRangeTracker.py [path_to_arrhythmia.data] [max_users]
"""

from __future__ import annotations

import sys
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_MODULE_FILE = "decision_pipeline.py"

TRACKED_FILES = {
    "decision_pipeline.py",
    "build_decision_tree.py",
    "action_normalRange.py",
    "action_pruning.py",
}

OUTPUT_PATH = Path(__file__).parent / "VariableRangeData.txt"

MAX_VALUES_PER_VAR = 50_000

# ---------------------------------------------------------------------------
# Tracker state
# ---------------------------------------------------------------------------

_var_history: OrderedDict[str, list[float]] = OrderedDict()
_lock = threading.Lock()


def _is_numeric_scalar(val: Any) -> bool:
    if isinstance(val, (int, float)):
        return True
    if isinstance(val, (np.integer, np.floating)):
        return True
    return False


def _to_float(val: Any) -> float:
    return float(val)


def _record(var_key: str, val: Any) -> None:
    if not _is_numeric_scalar(val):
        return
    fval = _to_float(val)
    if np.isnan(fval) or np.isinf(fval):
        return
    with _lock:
        if var_key not in _var_history:
            _var_history[var_key] = [fval]
        else:
            lst = _var_history[var_key]
            if len(lst) < MAX_VALUES_PER_VAR:
                lst.append(fval)


# ---------------------------------------------------------------------------
# sys.settrace callbacks
# ---------------------------------------------------------------------------

def _local_tracer(frame, event, arg):
    if event == "line" or event == "return":
        func_name = frame.f_code.co_name
        for name, val in frame.f_locals.items():
            if name.startswith("_") or name.startswith("__"):
                continue
            var_key = f"{func_name}.{name}"
            _record(var_key, val)
    return _local_tracer


def _global_tracer(frame, event, arg):
    if event != "call":
        return None
    filename = os.path.basename(frame.f_code.co_filename)
    if filename in TRACKED_FILES:
        return _local_tracer
    return None

# ---------------------------------------------------------------------------
# Main — run decision_pipeline through the tracer
# ---------------------------------------------------------------------------

def main():
    pipeline_dir = Path(__file__).parent.parent / "CDS-NI_Algorithms"
    sys.path.insert(0, str(pipeline_dir))

    from build_decision_tree import load_dataset
    from decision_pipeline import run_loocv, print_results

    data_path = sys.argv[1] if len(sys.argv) > 1 else str(pipeline_dir / "data" / "arrhythmia.data")
    max_users = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"[VariableRangeTracker] Loading dataset from {data_path}")
    data, labels = load_dataset(data_path)
    print(f"[VariableRangeTracker] Data shape: {data.shape}, tracing with max_users={max_users}")
    print(f"[VariableRangeTracker] Tracing files: {TRACKED_FILES}")

    sys.settrace(_global_tracer)
    threading.settrace(_global_tracer)

    try:
        output = run_loocv(data, labels, max_users=max_users)
    finally:
        sys.settrace(None)
        threading.settrace(None)

    print_results(output)
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"{'Variable':<55} | {'Min':>15} | {'Max':>15} | Values (up to last {MAX_VALUES_PER_VAR})\n")
        f.write("-" * 55 + "-+-" + "-" * 15 + "-+-" + "-" * 15 + "-+-" + "-" * 40 + "\n")

        for var_key, values in _var_history.items():
            if not values:
                continue
            v_min = min(values)
            v_max = max(values)
            vals_str = ", ".join(f"{v:g}" for v in values)
            f.write(f"{var_key:<55} | {v_min:>15.6g} | {v_max:>15.6g} | [{vals_str}]\n")

    print(f"\n[VariableRangeTracker] Wrote {len(_var_history)} variables to {OUTPUT_PATH}")



if __name__ == "__main__":
    main()
