"""
fpga_uart_validator.py — 10-Fold CV with per-fold FPGA validation

=== TWO OPERATING MODES ===

  SOFTWARE MODE (run once first):
    python fpga_uart_validator.py --mode software

    - Runs 10-fold CV in Python (fixed-point golden model)
    - Prints per-fold AND aggregate accuracy/sensitivity/specificity
    - Exports 10 sets of .mem files (one per fold) for FPGA synthesis
    - Saves golden predictions per fold so FPGA mode can compare later

    After this completes, you have:
      fpga_cv_output/
        fold_0/   <- .mem files + golden_results.csv + test_vectors.mem
        fold_1/
        ...
        fold_9/

  FPGA MODE (run once per fold, after synthesizing with that fold's .mem files):
    python fpga_uart_validator.py --mode fpga --fold 0 --port COM3

    - Loads fold 0's saved golden predictions (from software mode)
    - Sends ONLY fold 0's test users to the FPGA via UART
    - Compares FPGA response against the saved golden predictions
    - Reports match rate and accuracy

  FULL WORKFLOW:
    1. Run software mode once (generates everything)
    2. For fold 0: copy fold_0/*.mem into Vivado, synthesize, program FPGA
    3. Run: python fpga_uart_validator.py --mode fpga --fold 0 --port COM3
    4. Repeat steps 2-3 for folds 1-9
    5. Compare per-fold results between Python and FPGA
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "FixedPoint_decision_pipeline.py"))

from CDS_NI_Algorithms.build_decision_tree import (
    DecisionTree, TreeNode, HEALTHY_CLASS, N_FEATURES,
    load_dataset, build_decision_tree,
)
from CDS_NI_Algorithms.action_normalRange import (
    Algorithm2Output, run_algorithm2, DEFAULT_N_BINS,
)
from CDS_NI_Algorithms.action_pruning import Algorithm3Output, run_algorithm3
from decision_pipeline_fixedPoint import (
    to_fixed, run_algorithm4, HealthDecision, PredictionRecord,
    Algorithm4Output, print_results,
)

# Import the parameter export functions for .mem file generation
from parameter_export import (
    export_model_parameters, export_test_vectors,
    export_golden_predictions, export_af_trace,
)


# ===========================================================================
# Constants
# ===========================================================================
HEADER_BYTE = 0xAA
N_FEAT = N_FEATURES  # 279

FPGA_DEC_HEALTHY   = 0b00
FPGA_DEC_UNHEALTHY = 0b01
FPGA_DEC_SCREENING = 0b10

DECISION_FROM_CODE = {
    FPGA_DEC_HEALTHY:   HealthDecision.HEALTHY,
    FPGA_DEC_UNHEALTHY: HealthDecision.UNHEALTHY,
    FPGA_DEC_SCREENING: HealthDecision.SCREENING,
    0b11:               HealthDecision.UNKNOWN,
}

DECISION_TO_CODE = {
    HealthDecision.HEALTHY:   0,
    HealthDecision.UNHEALTHY: 1,
    HealthDecision.SCREENING: 2,
    HealthDecision.UNKNOWN:   3,
}


# ===========================================================================
# Fold Splitting — deterministic, must match parameter_export.py
# ===========================================================================

def compute_fold_splits(
    n_total: int, rng_seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """Compute the 10-fold train/test index splits.

    Uses the exact same logic as parameter_export.export_all_folds()
    and decision_pipeline_fixedPoint.ten_fold_cv().

    Returns a list of 10 tuples: (train_indices, test_indices).
    """
    random.seed(rng_seed)
    indices = list(range(n_total))
    random.shuffle(indices)
    fold_size = (n_total + 9) // 10

    folds = []
    for fold in range(10):
        start_idx = fold * fold_size
        end_idx = min(start_idx + fold_size, n_total)
        test_indices = indices[start_idx:end_idx]
        train_indices = [idx for idx in indices if idx not in test_indices]
        folds.append((train_indices, test_indices))

    return folds


# ===========================================================================
# Per-Fold Statistics
# ===========================================================================

def compute_fold_stats(records: List[PredictionRecord]) -> Dict:
    """Compute accuracy/sensitivity/specificity for one fold's records."""
    n = len(records)
    if n == 0:
        return {'n': 0, 'accuracy': 0, 'sensitivity': 0, 'specificity': 0,
                'false_alarm_rate': 0, 'n_screening': 0}

    n_healthy_total = sum(1 for r in records if r.true_is_healthy)
    n_diseased_total = sum(1 for r in records if r.true_is_diseased)

    n_healthy_correct = sum(
        1 for r in records
        if r.true_is_healthy and r.decision != HealthDecision.UNHEALTHY
    )
    n_diseased_correct = sum(
        1 for r in records
        if r.true_is_diseased and r.decision == HealthDecision.UNHEALTHY
    )
    n_false_alarm = sum(
        1 for r in records
        if r.true_is_healthy and r.decision == HealthDecision.UNHEALTHY
    )
    n_screening = sum(1 for r in records if r.decision == HealthDecision.SCREENING)

    n_correct = n_healthy_correct + n_diseased_correct

    return {
        'n': n,
        'n_correct': n_correct,
        'accuracy': n_correct / n if n else 0,
        'sensitivity': n_diseased_correct / n_diseased_total if n_diseased_total else 0,
        'specificity': n_healthy_correct / n_healthy_total if n_healthy_total else 0,
        'false_alarm_rate': n_false_alarm / n_healthy_total if n_healthy_total else 0,
        'n_screening': n_screening,
        'n_healthy_total': n_healthy_total,
        'n_diseased_total': n_diseased_total,
        'n_healthy_correct': n_healthy_correct,
        'n_diseased_correct': n_diseased_correct,
    }


# ===========================================================================
# Golden Results Save/Load
# ===========================================================================

def save_golden_results(
    records: List[PredictionRecord],
    test_indices: List[int],
    filepath: str,
) -> None:
    """Save golden model predictions to a CSV file for later FPGA comparison.

    Format: user_idx, true_label, decision_code, alarm_class, is_correct, af_value
    """
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'user_idx', 'true_label', 'decision_code',
            'alarm_class', 'is_correct', 'af_value',
        ])
        for record, user_idx in zip(records, test_indices):
            # Get final AF from last trace step
            if record.af_trace:
                final_af = record.af_trace[-1].AF_real
            else:
                final_af = 0

            writer.writerow([
                user_idx,
                record.true_label,
                DECISION_TO_CODE[record.decision],
                record.alarm_class if record.alarm_class is not None else -1,
                1 if record.is_correct else 0,
                final_af,
            ])


def load_golden_results(filepath: str) -> List[Dict]:
    """Load saved golden predictions from CSV."""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'user_idx': int(row['user_idx']),
                'true_label': int(row['true_label']),
                'decision_code': int(row['decision_code']),
                'alarm_class': int(row['alarm_class']),
                'is_correct': bool(int(row['is_correct'])),
                'af_value': int(row['af_value']),
            })
    return results


# ===========================================================================
# UART Functions (same as before, unchanged)
# ===========================================================================

def features_to_uart_bytes(data: np.ndarray, user_idx: int) -> bytes:
    """Convert one user's feature vector into 558 UART payload bytes."""
    payload = bytearray()
    for feat_j in range(N_FEAT):
        val = float(data[user_idx, feat_j])
        if np.isnan(val):
            fixed_val = 0
        else:
            fixed_val = to_fixed(val, 11, 4)
        if fixed_val < 0:
            fixed_val = fixed_val + (1 << 16)
        fixed_val = fixed_val & 0xFFFF
        payload.append((fixed_val >> 8) & 0xFF)
        payload.append(fixed_val & 0xFF)
    return bytes(payload)


def decode_fpga_response(response: bytes) -> Tuple[int, int, int]:
    """Decode 5-byte FPGA response into (decision, alarm_class, af_value)."""
    decision_byte = response[0]
    decision    = decision_byte & 0x03
    alarm_class = (decision_byte >> 2) & 0x0F
    af_unsigned = (response[1] << 24) | (response[2] << 16) | \
                  (response[3] << 8)  |  response[4]
    if af_unsigned >= (1 << 31):
        af_value = af_unsigned - (1 << 32)
    else:
        af_value = af_unsigned
    return decision, alarm_class, af_value


def send_user_to_fpga(ser, data: np.ndarray, user_idx: int,
                       timeout: float = 5.0) -> Optional[Tuple[int, int, int]]:
    """Send one user to FPGA, return (decision, alarm_class, af_value) or None."""
    ser.reset_input_buffer()
    ser.write(bytes([HEADER_BYTE]))
    ser.write(features_to_uart_bytes(data, user_idx))
    ser.flush()

    start = time.time()
    response = b''
    while len(response) < 5 and (time.time() - start) < timeout:
        chunk = ser.read(5 - len(response))
        if chunk:
            response += chunk
    if len(response) < 5:
        return None
    return decode_fpga_response(response)


# ===========================================================================
# SOFTWARE MODE — 10-Fold CV + Export
# ===========================================================================

def run_software_mode(
    data: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    max_users: Optional[int] = None,
    rng_seed: int = 42,
) -> None:
    """Run 10-fold CV, print per-fold results, export .mem files + golden predictions."""

    n_total = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    folds = compute_fold_splits(n_total, rng_seed)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"10-FOLD CROSS VALIDATION  (Fixed-Point Golden Model)")
    print(f"  Users: {n_total},  Fold size: ~{(n_total + 9) // 10}")
    print(f"  Output: {output_dir}/fold_0/ .. fold_9/")
    print(f"{'='*70}\n")

    all_records: List[PredictionRecord] = []
    fold_stats_list: List[Dict] = []

    for fold_idx, (train_indices, test_indices) in enumerate(folds):
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        print(f"--- Fold {fold_idx}/9: train={len(train_indices)}, "
              f"test={len(test_indices)} ---")

        # ---- Train Algorithms 1-3 on the TRAINING partition ----
        train_data = data[train_indices]
        train_labels = labels[train_indices]

        tree_i = build_decision_tree(train_data, train_labels)

        root_id = tree_i.root.node_id
        nodes_filter = [root_id]
        level2_by_feat: Dict[int, List] = defaultdict(list)
        for n in tree_i.nodes_by_level.get(2, []):
            if not n.is_leaf:
                level2_by_feat[n.branching_feat_k].append(n)
        for feat_k, children in level2_by_feat.items():
            if len(children) >= 2:
                nodes_filter.extend(c.node_id for c in children)

        print(f"  Tree: {tree_i.count_nodes()} nodes, {len(nodes_filter)} active")

        alg2_i = run_algorithm2(tree_i, train_data, train_labels,
                                DEFAULT_N_BINS, nodes_filter)
        print(f"  Alg2: {alg2_i.n_perceptor_entries} perceptor, "
              f"{alg2_i.n_executive_entries} executive")

        alg3_i = run_algorithm3(alg2_i, tree_i, train_data, train_labels,
                                nodes_filter, reset_per_h=False)
        print(f"  Alg3: {len(alg3_i.refined_actions)} retained, "
              f"{len(alg3_i.removed_actions)} removed")

        # ---- Export .mem files for this fold ----
        print(f"  Exporting .mem files to {fold_dir}/")
        export_model_parameters(tree_i, alg2_i, alg3_i, nodes_filter, fold_dir)

        export_test_vectors(data, labels, test_indices,
                            os.path.join(fold_dir, "test_vectors.mem"))

        # ---- Run golden model on TEST partition ----
        print(f"  Running golden model on {len(test_indices)} test users...")
        fold_records: List[PredictionRecord] = []
        for user_idx in test_indices:
            record = run_algorithm4(
                user_idx, data, labels, tree_i, alg2_i, alg3_i,
                rng_seed=rng_seed,
            )
            fold_records.append(record)

        # ---- Export golden predictions (.mem format for Verilog testbench) ----
        golden_records = export_golden_predictions(
            data, labels, test_indices, tree_i, alg2_i, alg3_i,
            os.path.join(fold_dir, "expected_output.mem"),
            rng_seed=rng_seed,
        )

        export_af_trace(golden_records, test_indices,
                        os.path.join(fold_dir, "af_trace.mem"))

        # ---- Save golden results as CSV (for FPGA mode comparison) ----
        save_golden_results(
            fold_records, test_indices,
            os.path.join(fold_dir, "golden_results.csv"),
        )

        # ---- Compute per-fold stats ----
        stats = compute_fold_stats(fold_records)
        fold_stats_list.append(stats)
        all_records.extend(fold_records)

        print(f"  Fold {fold_idx} results: accuracy={stats['accuracy']*100:.1f}%  "
              f"sensitivity={stats['sensitivity']*100:.1f}%  "
              f"specificity={stats['specificity']*100:.1f}%\n")

    # ================================================================
    # Print per-fold summary table
    # ================================================================
    print(f"\n{'='*70}")
    print("PER-FOLD RESULTS")
    print(f"{'='*70}")
    print(f"  {'Fold':>4}  {'Users':>5}  {'Correct':>7}  {'Acc':>7}  "
          f"{'Sens':>7}  {'Spec':>7}  {'FA Rate':>7}  {'Screen':>6}")
    print(f"  {'-'*60}")

    for i, stats in enumerate(fold_stats_list):
        print(f"  {i:4d}  {stats['n']:5d}  {stats['n_correct']:7d}  "
              f"{stats['accuracy']*100:6.1f}%  "
              f"{stats['sensitivity']*100:6.1f}%  "
              f"{stats['specificity']*100:6.1f}%  "
              f"{stats['false_alarm_rate']*100:6.1f}%  "
              f"{stats['n_screening']:6d}")

    # ================================================================
    # Print aggregate results
    # ================================================================
    agg = compute_fold_stats(all_records)

    print(f"  {'-'*60}")
    print(f"  {'AVG':>4}  {agg['n']:5d}  {agg['n_correct']:7d}  "
          f"{agg['accuracy']*100:6.1f}%  "
          f"{agg['sensitivity']*100:6.1f}%  "
          f"{agg['specificity']*100:6.1f}%  "
          f"{agg['false_alarm_rate']*100:6.1f}%  "
          f"{agg['n_screening']:6d}")
    print(f"{'='*70}")

    # Per-class breakdown (aggregate)
    diseased = [r for r in all_records if r.true_is_diseased]
    if diseased:
        by_class: Dict[int, List[PredictionRecord]] = defaultdict(list)
        for r in diseased:
            by_class[r.true_label].append(r)
        print(f"\n  Per-class detection (aggregate across all folds):")
        print(f"  {'class':>6} {'total':>6} {'detected':>9} {'rate':>7}")
        print(f"  {'-'*32}")
        for cls in sorted(by_class.keys()):
            recs = by_class[cls]
            detected = sum(1 for r in recs if r.decision == HealthDecision.UNHEALTHY)
            pct = detected / len(recs) * 100
            print(f"  {cls:6d} {len(recs):6d} {detected:9d} {pct:6.1f}%")

    print(f"\n{'='*70}")
    print(f"  All fold data exported to: {output_dir}/")
    print(f"  Each fold_N/ directory contains:")
    print(f"    - 6 x .mem files       (load into FPGA BRAMs for that fold)")
    print(f"    - test_vectors.mem     (which users to test)")
    print(f"    - expected_output.mem  (golden predictions in hex, for Verilog TB)")
    print(f"    - golden_results.csv   (golden predictions, for FPGA UART mode)")
    print(f"    - af_trace.mem         (step-by-step AF trace, for RTL debug)")
    print(f"{'='*70}\n")

    # Save a summary JSON for easy reference
    summary = {
        'rng_seed': rng_seed,
        'n_total': n_total,
        'n_folds': 10,
        'aggregate': {
            'accuracy': agg['accuracy'],
            'sensitivity': agg['sensitivity'],
            'specificity': agg['specificity'],
            'false_alarm_rate': agg['false_alarm_rate'],
        },
        'per_fold': [
            {
                'fold': i,
                'n_test': s['n'],
                'accuracy': s['accuracy'],
                'sensitivity': s['sensitivity'],
                'specificity': s['specificity'],
            }
            for i, s in enumerate(fold_stats_list)
        ],
    }
    summary_path = os.path.join(output_dir, "cv_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}")


# ===========================================================================
# FPGA MODE — Validate one fold against saved golden predictions
# ===========================================================================

def run_fpga_mode(
    data: np.ndarray,
    labels: np.ndarray,
    fold_idx: int,
    output_dir: str,
    port: str,
    baud: int = 115200,
    rng_seed: int = 42,
) -> None:
    """Send one fold's test users to the FPGA and compare against golden predictions.

    Prerequisites:
      1. Software mode was run first (generated golden_results.csv per fold)
      2. fold_N's .mem files were loaded into FPGA (synthesized with those BRAMs)
      3. FPGA is programmed and connected via USB-UART
    """
    try:
        import serial
    except ImportError:
        print("\nERROR: pyserial is required. Install with: pip install pyserial")
        return

    # ---- Load saved golden results for this fold ----
    golden_path = os.path.join(output_dir, f"fold_{fold_idx}", "golden_results.csv")
    if not os.path.exists(golden_path):
        print(f"\nERROR: Golden results not found at: {golden_path}")
        print(f"  You must run software mode first:")
        print(f"    python fpga_uart_validator.py --mode software")
        return

    golden_results = load_golden_results(golden_path)
    n_test = len(golden_results)

    print(f"\n{'='*70}")
    print(f"FPGA VALIDATION — Fold {fold_idx}")
    print(f"{'='*70}")
    print(f"  Serial port:      {port}")
    print(f"  Baud rate:        {baud}")
    print(f"  Test users:       {n_test}")
    print(f"  Golden results:   {golden_path}")
    print(f"  .mem files from:  {output_dir}/fold_{fold_idx}/")
    print(f"{'='*70}")

    # ---- Open serial port ----
    print(f"\n  Opening {port} at {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=2.0)
    except Exception as e:
        print(f"\n  ERROR: Could not open {port}: {e}")
        print(f"  Check that the FPGA is connected and no other program has the port.")
        return

    time.sleep(1.0)  # let FPGA reset settle

    # ---- Send each test user and compare ----
    print(f"  Sending {n_test} test users to FPGA...\n")

    n_match = 0
    n_mismatch = 0
    n_timeout = 0
    n_correct = 0  # correct vs ground truth

    for i, golden in enumerate(golden_results):
        user_idx = golden['user_idx']
        golden_dec = golden['decision_code']
        golden_alarm = golden['alarm_class']
        golden_correct = golden['is_correct']

        # Send to FPGA
        fpga_result = send_user_to_fpga(ser, data, user_idx)

        if fpga_result is None:
            print(f"  User {user_idx:3d}: TIMEOUT")
            n_timeout += 1
            continue

        fpga_dec, fpga_alarm, fpga_af = fpga_result

        # Compare decision
        decisions_match = (fpga_dec == golden_dec)
        if decisions_match:
            n_match += 1
        else:
            n_mismatch += 1
            golden_dec_name = [k.value for k, v in DECISION_TO_CODE.items() if v == golden_dec][0]
            fpga_dec_name = DECISION_FROM_CODE.get(fpga_dec, HealthDecision.UNKNOWN).value
            print(f"  User {user_idx:3d}: MISMATCH  "
                  f"golden={golden_dec_name}  fpga={fpga_dec_name}  "
                  f"true_label={golden['true_label']}")

        # Check correctness vs ground truth
        fpga_decision = DECISION_FROM_CODE.get(fpga_dec, HealthDecision.UNKNOWN)
        true_label = golden['true_label']
        if true_label == HEALTHY_CLASS:
            is_correct = (fpga_decision != HealthDecision.UNHEALTHY)
        else:
            is_correct = (fpga_decision == HealthDecision.UNHEALTHY)
        if is_correct:
            n_correct += 1

        # Progress
        if (i + 1) % 10 == 0 or i == n_test - 1:
            n_responded = n_match + n_mismatch
            match_pct = n_match / n_responded * 100 if n_responded else 0
            print(f"  Progress: {i+1:3d}/{n_test}  "
                  f"match={n_match}/{n_responded} ({match_pct:.1f}%)  "
                  f"timeouts={n_timeout}")

    ser.close()

    # ---- Report ----
    n_responded = n_match + n_mismatch
    match_rate = n_match / n_responded * 100 if n_responded else 0
    accuracy = n_correct / n_responded * 100 if n_responded else 0

    # Load Python golden accuracy for comparison
    golden_correct_count = sum(1 for g in golden_results if g['is_correct'])
    golden_accuracy = golden_correct_count / n_test * 100

    print(f"\n{'='*70}")
    print(f"FPGA VALIDATION RESULTS — Fold {fold_idx}")
    print(f"{'='*70}")
    print(f"  Test users:             {n_test}")
    print(f"  FPGA responses:         {n_responded}")
    print(f"  Timeouts:               {n_timeout}")
    print(f"")
    print(f"  Match (FPGA vs Python): {n_match}/{n_responded} ({match_rate:.1f}%)")
    print(f"  Mismatches:             {n_mismatch}")
    print(f"")
    print(f"  FPGA accuracy:          {accuracy:.1f}%")
    print(f"  Python accuracy:        {golden_accuracy:.1f}%")

    if n_mismatch == 0 and n_responded > 0:
        print(f"\n  PASS: All {n_responded} predictions match the golden model!")
    elif n_mismatch > 0:
        print(f"\n  FAIL: {n_mismatch} mismatches — FPGA implementation has bugs.")
    print(f"{'='*70}\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CDS Algorithm 4 — 10-Fold CV + Per-Fold FPGA Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Step-by-step workflow:

  1. Run software mode (once):
     python fpga_uart_validator.py --mode software

  2. For each fold 0-9:
     a. Copy fpga_cv_output/fold_N/*.mem into your Vivado project
     b. Re-synthesize + program the FPGA
     c. Run: python fpga_uart_validator.py --mode fpga --fold N --port COM3

  3. Compare per-fold accuracy between Python and FPGA
        """,
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Path to arrhythmia.data")
    parser.add_argument("--mode", choices=["software", "fpga"], default="software",
                        help="'software' = Python 10-fold CV + export; "
                             "'fpga' = validate one fold on FPGA hardware")
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold to validate in FPGA mode (0-9, default: 0)")
    parser.add_argument("--port", type=str, default="COM3",
                        help="Serial port for FPGA mode (default: COM3)")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Baud rate (default: 115200)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: fpga_cv_output)")
    parser.add_argument("--max-users", type=int, default=None,
                        help="Limit number of users (for quick debug runs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default: 42)")
    args = parser.parse_args()

    # ---- Load dataset ----
    data_path = args.data or str(
        PROJECT_ROOT / "CDS_NI_Algorithms" / "data" / "arrhythmia.data"
    )
    print(f"Loading dataset: {data_path}")
    data, labels = load_dataset(data_path)
    print(f"  {data.shape[0]} users, {data.shape[1]} features")

    output_dir = args.output or str(PROJECT_ROOT / "fpga_cv_output")

    # ---- Run selected mode ----
    if args.mode == "software":
        run_software_mode(
            data, labels, output_dir,
            max_users=args.max_users,
            rng_seed=args.seed,
        )

    elif args.mode == "fpga":
        if args.fold < 0 or args.fold > 9:
            print("ERROR: --fold must be 0-9")
            return
        run_fpga_mode(
            data, labels,
            fold_idx=args.fold,
            output_dir=output_dir,
            port=args.port,
            baud=args.baud,
            rng_seed=args.seed,
        )


if __name__ == "__main__":
    main()
