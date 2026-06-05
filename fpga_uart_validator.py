from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from codecarbon import EmissionsTracker


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
from FixedPoint_pipeline.decision_pipeline_fixedPoint import (
    to_fixed, run_algorithm4, HealthDecision, PredictionRecord,
    Algorithm4Output, print_results,
)

# Import the parameter export functions for .mem file generation
from FixedPoint_pipeline.parameter_export import (
    export_model_parameters, export_test_vectors,
    export_golden_predictions, export_af_trace,
)

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


def save_golden_results(
    records: List[PredictionRecord],
    test_indices: List[int],
    filepath: str,
    sw_latencies_ms: Optional[List[float]] = None,
    fold_energy: Optional[Dict] = None,
) -> None:
    """Save golden model predictions to a CSV file for later FPGA comparison.

    Format: user_idx, true_label, decision_code, alarm_class, is_correct, af_value, sw_latency_ms,
            fold_energy_kwh, fold_power_watts, fold_duration_s, fold_cpu_energy_kwh, fold_ram_energy_kwh
    """
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'user_idx', 'true_label', 'decision_code',
            'alarm_class', 'is_correct', 'af_value', 'sw_latency_ms',
            'fold_energy_kwh', 'fold_power_watts', 'fold_duration_s',
            'fold_cpu_energy_kwh', 'fold_ram_energy_kwh',
        ])
        for i, (record, user_idx) in enumerate(zip(records, test_indices)):
            # Get final AF from last trace step
            if record.af_trace:
                final_af = record.af_trace[-1].AF_real
            else:
                final_af = 0

            lat = sw_latencies_ms[i] if sw_latencies_ms else 0.0

            e = fold_energy or {}
            writer.writerow([
                user_idx,
                record.true_label,
                DECISION_TO_CODE[record.decision],
                record.alarm_class if record.alarm_class is not None else -1,
                1 if record.is_correct else 0,
                final_af,
                f"{lat:.3f}",
                f"{e.get('energy_kwh', 0):.10f}",
                f"{e.get('power_watts', 0):.4f}",
                f"{e.get('duration_s', 0):.3f}",
                f"{e.get('cpu_energy_kwh', 0):.10f}",
                f"{e.get('ram_energy_kwh', 0):.10f}",
            ])


def load_golden_results(filepath: str) -> List[Dict]:
    """Load saved golden predictions from CSV."""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                'user_idx': int(row['user_idx']),
                'true_label': int(row['true_label']),
                'decision_code': int(row['decision_code']),
                'alarm_class': int(row['alarm_class']),
                'is_correct': bool(int(row['is_correct'])),
                'af_value': int(row['af_value']),
            }
            if 'sw_latency_ms' in row:
                entry['sw_latency_ms'] = float(row['sw_latency_ms'])
            else:
                entry['sw_latency_ms'] = 0.0
            for col in ('fold_energy_kwh', 'fold_power_watts', 'fold_duration_s',
                        'fold_cpu_energy_kwh', 'fold_ram_energy_kwh'):
                if col in row:
                    entry[col] = float(row[col])
            results.append(entry)
    return results

def features_to_uart_bytes(data: np.ndarray, user_idx: int) -> bytes:
    """Convert one user's feature vector into 558 UART payload bytes."""
    payload = bytearray()
    for feat_j in range(N_FEAT):
        val = float(data[user_idx, feat_j])
        if np.isnan(val):
            fixed_val = 0x7FFF  # NaN sentinel — FPGA skips actions for this feature
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
                       timeout: float = 5.0) -> Optional[Tuple[int, int, int, float]]:
    """Send one user to FPGA, return (decision, alarm_class, af_value, latency_ms) or None."""
    ser.reset_input_buffer()
    t_start = time.perf_counter()
    ser.write(bytes([HEADER_BYTE]))
    ser.write(features_to_uart_bytes(data, user_idx))
    ser.flush()

    start = time.time()
    response = b''
    while len(response) < 5 and (time.time() - start) < timeout:
        chunk = ser.read(5 - len(response))
        if chunk:
            response += chunk
    t_end = time.perf_counter()
    if len(response) < 5:
        return None
    dec, alarm, af = decode_fpga_response(response)
    latency_ms = (t_end - t_start) * 1000.0
    return dec, alarm, af, latency_ms


N_LANES = 4  # number of parallel patient lanes on FPGA


def send_batch_to_fpga(
    ser, data: np.ndarray, user_indices: List[int],
    timeout: float = 15.0,
) -> Optional[List[Tuple[int, int, int, float]]]:
    """Send 4 users to FPGA in one batch, return list of 4 result tuples or None.

    Protocol: send 4 x [0xAA + 558 bytes] back-to-back.
    FPGA processes all 4 in parallel, returns 4 x 5-byte responses.

    Each result tuple: (decision, alarm_class, af_value, per_user_latency_ms).
    """
    assert len(user_indices) == N_LANES

    ser.reset_input_buffer()
    t_start = time.perf_counter()

    # Send all 4 patients back-to-back
    for user_idx in user_indices:
        ser.write(bytes([HEADER_BYTE]))
        ser.write(features_to_uart_bytes(data, user_idx))
    ser.flush()

    # Receive 4 x 5 = 20 response bytes
    n_expected = N_LANES * 5
    start = time.time()
    response = b''
    while len(response) < n_expected and (time.time() - start) < timeout:
        chunk = ser.read(n_expected - len(response))
        if chunk:
            response += chunk
    t_end = time.perf_counter()

    if len(response) < n_expected:
        return None

    batch_latency_ms = (t_end - t_start) * 1000.0
    per_user_ms = batch_latency_ms / N_LANES

    results = []
    for i in range(N_LANES):
        dec, alarm, af = decode_fpga_response(response[i*5 : i*5 + 5])
        results.append((dec, alarm, af, per_user_ms))

    return results


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
    fold_energy_list: List[Dict] = []

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

        # ---- Run golden model on TEST partition (Alg 4 only — energy tracked) ----
        # Energy measurement covers ONLY Algorithm 4 inference, not Alg 1-3
        # training. This matches what the FPGA executes for fair comparison.
        tracker = EmissionsTracker(
            project_name=f"fold_{fold_idx}",
            output_dir=fold_dir,
            output_file=f"fold_{fold_idx}_emissions.csv",
            log_level="error",
            save_to_file=True,
        )
        tracker.start()
        fold_t0 = time.perf_counter()

        print(f"  Running golden model on {len(test_indices)} test users...")
        fold_records: List[PredictionRecord] = []
        fold_sw_latencies: List[float] = []
        for user_idx in test_indices:
            t0 = time.perf_counter()
            record = run_algorithm4(
                user_idx, data, labels, tree_i, alg2_i, alg3_i,
                rng_seed=rng_seed,
            )
            t1 = time.perf_counter()
            fold_records.append(record)
            fold_sw_latencies.append((t1 - t0) * 1000.0)

        fold_t1 = time.perf_counter()
        fold_emissions = tracker.stop()
        fold_duration_s = fold_t1 - fold_t0

        fold_energy_kwh = fold_emissions if fold_emissions is not None else 0.0
        fold_power_watts = (fold_energy_kwh * 1000.0 * 3600.0) / fold_duration_s if fold_duration_s > 0 else 0.0

        cpu_energy_kwh = tracker._total_cpu_energy.kWh if hasattr(tracker, '_total_cpu_energy') else 0.0
        ram_energy_kwh = tracker._total_ram_energy.kWh if hasattr(tracker, '_total_ram_energy') else 0.0

        fold_energy = {
            'energy_kwh': fold_energy_kwh,
            'power_watts': fold_power_watts,
            'duration_s': fold_duration_s,
            'cpu_energy_kwh': cpu_energy_kwh,
            'ram_energy_kwh': ram_energy_kwh,
        }
        fold_energy_list.append(fold_energy)

        golden_records = export_golden_predictions(
            data, labels, test_indices, tree_i, alg2_i, alg3_i,
            os.path.join(fold_dir, "expected_output.mem"),
            rng_seed=rng_seed,
        )

        export_af_trace(golden_records, test_indices,
                        os.path.join(fold_dir, "af_trace.mem"))

        save_golden_results(
            fold_records, test_indices,
            os.path.join(fold_dir, "golden_results.csv"),
            sw_latencies_ms=fold_sw_latencies,
            fold_energy=fold_energy,
        )

        stats = compute_fold_stats(fold_records)
        fold_stats_list.append(stats)
        all_records.extend(fold_records)

        print(f"  Fold {fold_idx} results: accuracy={stats['accuracy']*100:.1f}%  "
              f"sensitivity={stats['sensitivity']*100:.1f}%  "
              f"specificity={stats['specificity']*100:.1f}%")
        print(f"  Energy: {fold_energy_kwh*1e6:.4f} uWh  "
              f"({fold_power_watts:.2f} W avg over {fold_duration_s:.1f}s)\n")

    print(f"\n{'='*90}")
    print("PER-FOLD RESULTS")
    print(f"{'='*90}")
    print(f"  {'Fold':>4}  {'Users':>5}  {'Correct':>7}  {'Acc':>7}  "
          f"{'Sens':>7}  {'Spec':>7}  {'FA Rate':>7}  {'Screen':>6}  "
          f"{'Energy(uWh)':>11}  {'Power(W)':>8}")
    print(f"  {'-'*82}")

    for i, stats in enumerate(fold_stats_list):
        e = fold_energy_list[i]
        print(f"  {i:4d}  {stats['n']:5d}  {stats['n_correct']:7d}  "
              f"{stats['accuracy']*100:6.1f}%  "
              f"{stats['sensitivity']*100:6.1f}%  "
              f"{stats['specificity']*100:6.1f}%  "
              f"{stats['false_alarm_rate']*100:6.1f}%  "
              f"{stats['n_screening']:6d}  "
              f"{e['energy_kwh']*1e6:11.4f}  "
              f"{e['power_watts']:8.2f}")

    agg = compute_fold_stats(all_records)
    total_energy_kwh = sum(e['energy_kwh'] for e in fold_energy_list)
    avg_power_watts = sum(e['power_watts'] for e in fold_energy_list) / len(fold_energy_list) if fold_energy_list else 0

    print(f"  {'-'*82}")
    print(f"  {'TOT':>4}  {agg['n']:5d}  {agg['n_correct']:7d}  "
          f"{agg['accuracy']*100:6.1f}%  "
          f"{agg['sensitivity']*100:6.1f}%  "
          f"{agg['specificity']*100:6.1f}%  "
          f"{agg['false_alarm_rate']*100:6.1f}%  "
          f"{agg['n_screening']:6d}  "
          f"{total_energy_kwh*1e6:11.4f}  "
          f"{avg_power_watts:8.2f}")
    print(f"{'='*90}")

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
            'total_energy_kwh': total_energy_kwh,
            'avg_power_watts': avg_power_watts,
        },
        'per_fold': [
            {
                'fold': i,
                'n_test': s['n'],
                'accuracy': s['accuracy'],
                'sensitivity': s['sensitivity'],
                'specificity': s['specificity'],
                'energy_kwh': fold_energy_list[i]['energy_kwh'],
                'power_watts': fold_energy_list[i]['power_watts'],
                'duration_s': fold_energy_list[i]['duration_s'],
                'cpu_energy_kwh': fold_energy_list[i]['cpu_energy_kwh'],
                'ram_energy_kwh': fold_energy_list[i]['ram_energy_kwh'],
            }
            for i, s in enumerate(fold_stats_list)
        ],
    }
    summary_path = os.path.join(output_dir, "cv_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}")



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

    Binary decision only: HEALTHY vs UNHEALTHY (SCREENING counted as not-unhealthy).

    Collects 5 metrics:
      1. Bit-Exact Match Rate       — % FPGA decisions matching Python golden model
      2. AF Value Deviation          — mean/max |FPGA_AF - Python_AF| in Q s2.30
      3. Per-User Latency            — min/median/max/stdev of UART round-trip (ms)
      4. Throughput                   — users/sec for FPGA vs Python
      5. Binary Confusion Matrix     — 2x2 (Healthy/Unhealthy) Python vs FPGA

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

    golden_path = os.path.join(output_dir, f"fold_{fold_idx}", "golden_results.csv")
    if not os.path.exists(golden_path):
        print(f"\nERROR: Golden results not found at: {golden_path}")
        print(f"  You must run software mode first:")
        print(f"    python fpga_uart_validator.py --mode software")
        return

    golden_results = load_golden_results(golden_path)
    n_test = len(golden_results)

    print(f"\n{'='*70}")
    print(f"FPGA VALIDATION — Fold {fold_idx}  (5-Metric Analysis)")
    print(f"{'='*70}")
    print(f"  Serial port:      {port}")
    print(f"  Baud rate:        {baud}")
    print(f"  Test users:       {n_test}")
    print(f"  Golden results:   {golden_path}")
    print(f"  .mem files from:  {output_dir}/fold_{fold_idx}/")
    print(f"  Decision mode:    Binary (Healthy vs Unhealthy)")
    print(f"{'='*70}")

    print(f"\n  Opening {port} at {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=2.0)
    except Exception as e:
        print(f"\n  ERROR: Could not open {port}: {e}")
        print(f"  Check that the FPGA is connected and no other program has the port.")
        return

    time.sleep(1.0)  # let FPGA reset settle

    # 0=HEALTHY, 1=UNHEALTHY, 2=SCREENING  -->  binary: 1=UNHEALTHY, else=HEALTHY
    def to_binary(dec_code: int) -> int:
        return 1 if dec_code == 1 else 0  # only UNHEALTHY counts as positive

    n_batches = (n_test + N_LANES - 1) // N_LANES
    print(f"  Sending {n_test} test users in {n_batches} batches "
          f"of {N_LANES}...\n")

    n_match = 0
    n_mismatch = 0
    n_timeout = 0

    fpga_latencies_ms: List[float] = []
    af_deviations: List[int] = []

    confusion = [[0, 0], [0, 0]]
    py_confusion = [[0, 0], [0, 0]]

    processed = 0

    for batch_start in range(0, n_test, N_LANES):
        batch_end = min(batch_start + N_LANES, n_test)
        batch_goldens = golden_results[batch_start:batch_end]

        # Build list of user indices; pad to N_LANES if short
        batch_user_indices = [g['user_idx'] for g in batch_goldens]
        while len(batch_user_indices) < N_LANES:
            batch_user_indices.append(batch_user_indices[-1])

        batch_results = send_batch_to_fpga(ser, data, batch_user_indices)

        if batch_results is None:
            print(f"  Batch @user {batch_user_indices[0]}: TIMEOUT")
            n_timeout += len(batch_goldens)
            processed += len(batch_goldens)
            continue

        # Only process real users (ignore padding results)
        for lane_idx, golden in enumerate(batch_goldens):
            user_idx   = golden['user_idx']
            golden_dec = golden['decision_code']
            golden_af  = golden['af_value']
            true_label = golden['true_label']

            fpga_dec, _, fpga_af, latency_ms = batch_results[lane_idx]
            fpga_latencies_ms.append(latency_ms)

            py_bin    = to_binary(golden_dec)
            fp_bin    = to_binary(fpga_dec)
            truth_bin = 0 if true_label == HEALTHY_CLASS else 1

            if fp_bin == py_bin:
                n_match += 1
            else:
                n_mismatch += 1
                py_label = "UNHEALTHY" if py_bin else "HEALTHY"
                fp_label = "UNHEALTHY" if fp_bin else "HEALTHY"
                print(f"  User {user_idx:3d}: MISMATCH  "
                      f"python={py_label}  fpga={fp_label}  "
                      f"true={'UNHEALTHY' if truth_bin else 'HEALTHY'}")

            af_deviations.append(abs(fpga_af - golden_af))
            confusion[truth_bin][fp_bin] += 1
            py_confusion[truth_bin][py_bin] += 1

        processed += len(batch_goldens)
        if processed % 8 == 0 or processed >= n_test:
            n_responded = n_match + n_mismatch
            match_pct = n_match / n_responded * 100 if n_responded else 0
            print(f"  Progress: {processed:3d}/{n_test}  "
                  f"match={n_match}/{n_responded} ({match_pct:.1f}%)  "
                  f"timeouts={n_timeout}  ({N_LANES}/batch)")

    ser.close()

    n_responded = n_match + n_mismatch
    if n_responded == 0:
        print("\n  ERROR: No FPGA responses received. Cannot compute metrics.")
        return

    match_rate = n_match / n_responded * 100

    print(f"\n{'='*70}")
    print(f"FPGA vs PYTHON — 5-METRIC COMPARISON REPORT  (Fold {fold_idx})")
    print(f"{'='*70}")
    print(f"  Test users: {n_test}   Responded: {n_responded}   "
          f"Timeouts: {n_timeout}")

    print(f"\n  ---- METRIC 1: Bit-Exact Match Rate (Binary: H vs U) ----")
    print(f"  Matching:    {n_match}/{n_responded}  ({match_rate:.1f}%)")
    print(f"  Mismatches:  {n_mismatch}")
    if n_mismatch == 0:
        print(f"  PASS")
    else:
        print(f"  FAIL — FPGA diverges from golden model on {n_mismatch} users")


    print(f"\n  ---- METRIC 2: AF Value Deviation (Q s2.30) ----")
    if af_deviations:
        mean_dev = sum(af_deviations) / len(af_deviations)
        max_dev = max(af_deviations)
        scale = 1 << 30
        print(f"  Mean |FPGA_AF - Py_AF|:  {mean_dev:.0f}  "
              f"({mean_dev / scale:.8f} real)")
        print(f"  Max  |FPGA_AF - Py_AF|:  {max_dev}  "
              f"({max_dev / scale:.8f} real)")
        n_zero = sum(1 for d in af_deviations if d == 0)
        print(f"  Exact AF matches:        {n_zero}/{len(af_deviations)}  "
              f"({n_zero / len(af_deviations) * 100:.1f}%)")

    print(f"\n  ---- METRIC 3: Per-User Latency (ms) ----")
    fpga_processing = 0.0
    if fpga_latencies_ms:
        lat_min = min(fpga_latencies_ms)
        lat_max = max(fpga_latencies_ms)
        lat_med = statistics.median(fpga_latencies_ms)
        lat_mean = statistics.mean(fpga_latencies_ms)
        lat_std = statistics.stdev(fpga_latencies_ms) if len(fpga_latencies_ms) > 1 else 0

        uart_overhead_ms = (559 * 10 / baud + 5 * 10 / baud) * 1000
        fpga_processing = max(0, lat_med - uart_overhead_ms)

        print(f"  FPGA round-trip:    min={lat_min:.1f}  median={lat_med:.1f}  "
              f"max={lat_max:.1f}  stdev={lat_std:.1f}")
        print(f"  UART wire time:     ~{uart_overhead_ms:.1f} ms  "
              f"(559 TX + 5 RX bytes @ {baud} baud)")
        print(f"  Est. FPGA compute:  ~{fpga_processing:.1f} ms  "
              f"(median - wire time)")

    sw_latencies = [g['sw_latency_ms'] for g in golden_results
                    if g['sw_latency_ms'] > 0]
    if sw_latencies:
        sw_med = statistics.median(sw_latencies)
        sw_mean = statistics.mean(sw_latencies)
        sw_min = min(sw_latencies)
        sw_max = max(sw_latencies)
        print(f"  Python inference:   min={sw_min:.1f}  median={sw_med:.1f}  "
              f"max={sw_max:.1f} ms")

    print(f"\n  ---- METRIC 4: Throughput ----")
    if fpga_latencies_ms:
        fpga_throughput = 1000.0 / lat_mean
        print(f"  FPGA:    {fpga_throughput:.1f} users/sec  "
              f"(mean {lat_mean:.1f} ms)")
    if sw_latencies:
        sw_throughput = 1000.0 / sw_mean
        print(f"  Python:  {sw_throughput:.1f} users/sec  "
              f"(mean {sw_mean:.1f} ms)")
        if fpga_latencies_ms:
            ratio = sw_mean / lat_mean
            if ratio > 1:
                print(f"  FPGA is {ratio:.1f}x FASTER (wall-clock, incl. UART)")
            else:
                print(f"  Python is {1/ratio:.1f}x faster (UART overhead dominates)")
                print(f"  Note: FPGA compute alone ~{fpga_processing:.1f} ms vs "
                      f"Python {sw_med:.1f} ms")

    print(f"\n  ---- METRIC 5: Binary Confusion Matrix (H vs U) ----")

    # Helper to print one confusion matrix and derive stats
    def print_binary_confusion(label: str, cm: List[List[int]]) -> None:
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        total = tn + fp + fn + tp

        accuracy = (tp + tn) / total * 100 if total else 0
        sensitivity = tp / (tp + fn) * 100 if (tp + fn) else 0
        specificity = tn / (tn + fp) * 100 if (tn + fp) else 0
        fa_rate = fp / (fp + tn) * 100 if (fp + tn) else 0

        print(f"\n  {label}:")
        print(f"  {'':>14}  Predicted H  Predicted U")
        print(f"  {'True Healthy':>14}  {tn:>10}   {fp:>10}   (n={tn+fp})")
        print(f"  {'True Unhealthy':>14}  {fn:>10}   {tp:>10}   (n={fn+tp})")
        print(f"  Accuracy:     {accuracy:.1f}%  ({tp+tn}/{total})")
        print(f"  Sensitivity:  {sensitivity:.1f}%  ({tp}/{tp+fn} unhealthy detected)")
        print(f"  Specificity:  {specificity:.1f}%  ({tn}/{tn+fp} healthy correct)")
        print(f"  False Alarm:  {fa_rate:.1f}%  ({fp}/{fp+tn} healthy misclassified)")

    print_binary_confusion("FPGA", confusion)
    print_binary_confusion("Python (golden model)", py_confusion)

    print(f"\n  {'='*50}")
    if n_mismatch == 0 and n_responded > 0:
        print(f"  PASS: All {n_responded} binary decisions match!")
    elif n_mismatch > 0:
        print(f"  FAIL: {n_mismatch}/{n_responded} binary mismatches.")
    print(f"  {'='*50}")
    print(f"{'='*70}\n")


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

    data_path = args.data or str(
        PROJECT_ROOT / "CDS_NI_Algorithms" / "data" / "arrhythmia.data"
    )
    print(f"Loading dataset: {data_path}")
    data, labels = load_dataset(data_path)
    print(f"  {data.shape[0]} users, {data.shape[1]} features")

    output_dir = args.output or str(PROJECT_ROOT / "fpga_cv_output")

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
