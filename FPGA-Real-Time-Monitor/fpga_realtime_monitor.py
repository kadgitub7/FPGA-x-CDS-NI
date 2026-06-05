"""
Unlike the validation pipeline (which uses 10-fold CV on a fixed dataset),
this script trains the model on ALL 452 users, exports .mem files for the
FPGA, and then provides a live interactive diagnosis interface.

  python fpga_realtime_monitor.py --setup

  - Trains Algorithms 1-3 on the full dataset (all 452 users)
  - Exports .mem files to fpga_mem/ for FPGA synthesis
  - After this, load fpga_mem/*.mem into your Vivado project,
    synthesize, and program the FPGA

    For real time diagnosis, run:

  python fpga_realtime_monitor.py --run --port COM4

  - Opens a live interactive session
  - Type 'diagnose me' to diagnose from a CSV file
  - Type 'diagnose <user_id>' to diagnose a specific user from the dataset
  - Type 'load <path.csv>' to set the patient CSV file
  - Type 'list' to show all users in the dataset
  - Type 'quit' to exit

  If you want to load a new patient, you need to add the following to the csv:
  The patient CSV file should have 279 feature columns matching the
  arrhythmia dataset format. Column names are optional — if absent,
  columns are read positionally. Missing values use '?' or NaN.

  A template CSV with column headers is generated during --setup.

  This is the main project which is used to predict and acts as a real product which can theoretically be used
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent
CDS_NI_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(CDS_NI_ROOT))
sys.path.insert(0, str(CDS_NI_ROOT / "FixedPoint_pipeline"))

from CDS_NI_Algorithms.build_decision_tree import (
    DecisionTree, TreeNode, HEALTHY_CLASS, N_FEATURES,
    FEATURE_NAMES, load_dataset, build_decision_tree,
)
from CDS_NI_Algorithms.action_normalRange import (
    Algorithm2Output, run_algorithm2, DEFAULT_N_BINS,
)
from CDS_NI_Algorithms.action_pruning import Algorithm3Output, run_algorithm3
from FixedPoint_pipeline.decision_pipeline_fixedPoint import (
    to_fixed, run_algorithm4, HealthDecision, PredictionRecord,
)
from FixedPoint_pipeline.parameter_export import export_model_parameters


HEADER_BYTE = 0xAA
N_FEAT = N_FEATURES  # 279

DISEASE_CLASSES: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16)

DISEASE_NAMES: Dict[int, str] = {
    1:  "Normal (Healthy)",
    2:  "Ischemic changes (Coronary Artery Disease)",
    3:  "Old Anterior Myocardial Infarction",
    4:  "Old Inferior Myocardial Infarction",
    5:  "Sinus Tachycardia",
    6:  "Sinus Bradycardia",
    7:  "Ventricular Premature Contraction (PVC)",
    8:  "Supraventricular Premature Contraction",
    9:  "Left Bundle Branch Block",
    10: "Right Bundle Branch Block",
    14: "Left Ventricular Hypertrophy",
    15: "Atrial Fibrillation or Flutter",
    16: "Others",
}

FPGA_DEC_HEALTHY   = 0b00
FPGA_DEC_UNHEALTHY = 0b01
FPGA_DEC_SCREENING = 0b10

DECISION_NAMES = {
    0: "HEALTHY",
    1: "UNHEALTHY",
    2: "SCREENING",
    3: "UNKNOWN",
}

def get_ordered_feature_names() -> List[str]:
    return [FEATURE_NAMES.get(i, f"feature_{i}") for i in range(N_FEAT)]

def features_to_uart_bytes(features: np.ndarray) -> bytes:
    """Convert a 279-element feature vector to 558 UART payload bytes."""
    payload = bytearray()
    for feat_j in range(N_FEAT):
        val = float(features[feat_j])
        if np.isnan(val):
            fixed_val = 0x7FFF
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


N_LANES = 4  # parallel patient pipelines on FPGA


def send_batch_to_fpga(
    ser,
    feature_list: List[np.ndarray],
    timeout: float = 15.0,
) -> Optional[List[Tuple[int, int, int]]]:
    """Send 1-4 patients to the FPGA, return list of (decision, alarm_class, af_value).

    Always pads to N_LANES (4) internally — the FPGA expects exactly 4 patients.
    Padding uses the last real patient's data (results for padding slots are discarded
    by the caller).

    Returns results for all N_LANES slots, or None on timeout.
    """
    n_real = len(feature_list)
    assert 1 <= n_real <= N_LANES

    # Pad to exactly N_LANES
    padded = list(feature_list)
    while len(padded) < N_LANES:
        padded.append(padded[-1])

    ser.reset_input_buffer()

    # Send N_LANES patients back-to-back, each with 0xAA header
    for features in padded:
        ser.write(bytes([HEADER_BYTE]))
        ser.write(features_to_uart_bytes(features))
    ser.flush()

    # Receive N_LANES x 5 = 20 response bytes
    n_expected = N_LANES * 5
    start = time.time()
    response = b''
    while len(response) < n_expected and (time.time() - start) < timeout:
        chunk = ser.read(n_expected - len(response))
        if chunk:
            response += chunk

    if len(response) < n_expected:
        return None

    results = []
    for i in range(N_LANES):
        results.append(decode_fpga_response(response[i*5 : i*5 + 5]))

    return results

def load_patient_csv(csv_path: str) -> np.ndarray:
    """Load patient features from a CSV file.

    Expects 279 feature columns. If 280 columns exist, the last is treated
    as a label and ignored for diagnosis. Missing values ('?', '', NaN)
    are stored as NaN.

    Returns a (1, 279) numpy array.
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")

    has_header = False
    try:
        float(rows[0][0].replace('?', 'nan'))
    except ValueError:
        has_header = True

    data_rows = rows[1:] if has_header else rows

    if len(data_rows) == 0:
        raise ValueError(f"No data rows in CSV: {csv_path}")

    row = data_rows[0]
    n_cols = len(row)

    if n_cols < N_FEAT:
        raise ValueError(
            f"CSV has {n_cols} columns, need at least {N_FEAT}. "
            f"Check that all 279 features are present."
        )

    features = np.full(N_FEAT, np.nan)
    for j in range(N_FEAT):
        val = row[j].strip()
        if val in ('?', '', 'nan', 'NaN', 'NA'):
            features[j] = np.nan
        else:
            features[j] = float(val)

    return features


def load_patient_csv_multi(csv_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load multiple patients from a CSV file.

    Returns (features array [n_patients, 279], labels array or None).
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")

    has_header = False
    try:
        float(rows[0][0].replace('?', 'nan'))
    except ValueError:
        has_header = True

    data_rows = rows[1:] if has_header else rows
    if len(data_rows) == 0:
        raise ValueError(f"No data rows in CSV: {csv_path}")

    n_cols = len(data_rows[0])
    has_label = (n_cols >= 280)

    features = np.full((len(data_rows), N_FEAT), np.nan)
    labels = np.zeros(len(data_rows), dtype=int) if has_label else None

    for i, row in enumerate(data_rows):
        for j in range(min(N_FEAT, len(row))):
            val = row[j].strip()
            if val not in ('?', '', 'nan', 'NaN', 'NA'):
                features[i, j] = float(val)
        if has_label and len(row) > N_FEAT:
            try:
                labels[i] = int(row[N_FEAT])
            except (ValueError, IndexError):
                labels[i] = 0

    return features, labels

def run_setup(data: np.ndarray, labels: np.ndarray, output_dir: str) -> None:
    """Train on ALL users and export .mem files for FPGA synthesis."""

    print(f"\n{'='*70}")
    print(f"FPGA REAL-TIME MONITOR — SETUP")
    print(f"  Training on ALL {data.shape[0]} users (no train/test split)")
    print(f"  Output: {output_dir}/")
    print(f"{'='*70}\n")

    # Algorithm 1: Build decision tree
    print("Training Algorithm 1 (Decision Tree)...")
    tree = build_decision_tree(data, labels)

    root_id = tree.root.node_id
    nodes_filter = [root_id]
    level2_by_feat: Dict[int, list] = defaultdict(list)
    for n in tree.nodes_by_level.get(2, []):
        if not n.is_leaf:
            level2_by_feat[n.branching_feat_k].append(n)
    for feat_k, children in level2_by_feat.items():
        if len(children) >= 2:
            nodes_filter.extend(c.node_id for c in children)

    print(f"  Tree: {tree.count_nodes()} nodes, {len(nodes_filter)} active")

    # Algorithm 2: Perceptor/Executive library
    print("Training Algorithm 2 (Perceptor/Executive Library)...")
    alg2 = run_algorithm2(tree, data, labels, DEFAULT_N_BINS, nodes_filter)
    print(f"  Alg2: {alg2.n_perceptor_entries} perceptor, "
          f"{alg2.n_executive_entries} executive")

    # Algorithm 3: Action pruning
    print("Training Algorithm 3 (Action Pruning)...")
    alg3 = run_algorithm3(alg2, tree, data, labels, nodes_filter, reset_per_h=False)
    print(f"  Alg3: {len(alg3.refined_actions)} retained, "
          f"{len(alg3.removed_actions)} removed")

    # Export .mem files
    print(f"\nExporting .mem files to {output_dir}/")
    export_model_parameters(tree, alg2, alg3, nodes_filter, output_dir)

    # Generate CSV template with feature headers
    template_path = os.path.join(PROJECT_ROOT, "patient_template.csv")
    feature_names = get_ordered_feature_names()
    with open(template_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(feature_names)
        writer.writerow(['?' for _ in range(N_FEAT)])

    print(f"\n{'='*70}")
    print(f"SETUP COMPLETE")
    print(f"{'='*70}")
    print(f"  .mem files:        {output_dir}/")
    print(f"  Patient template:  {template_path}")
    print(f"")
    print(f"  Next steps:")
    print(f"    1. Copy {output_dir}/*.mem into your Vivado project")
    print(f"    2. Synthesize and program the FPGA")
    print(f"    3. Run: python fpga_realtime_monitor.py --run --port COM4")
    print(f"{'='*70}\n")

def print_diagnosis(decision: int, alarm_class: int, af_value: int,
                    patient_label: str = "Patient") -> None:
    """Pretty-print a diagnosis result from the FPGA."""
    dec_name = DECISION_NAMES.get(decision, "UNKNOWN")

    af_float = af_value / (1 << 30)

    print(f"\n  {'='*50}")
    print(f"  DIAGNOSIS RESULT — {patient_label}")
    print(f"  {'='*50}")

    if decision == FPGA_DEC_HEALTHY:
        print(f"  Status:     HEALTHY")
        print(f"  Confidence: AF = {af_float:.6f}")
        print(f"  Summary:    No cardiac arrhythmia detected.")
    elif decision == FPGA_DEC_UNHEALTHY:
        disease_class = DISEASE_CLASSES[alarm_class] if alarm_class < len(DISEASE_CLASSES) else -1
        disease_name = DISEASE_NAMES.get(disease_class, f"Unknown (class {disease_class})")
        print(f"  Status:     UNHEALTHY")
        print(f"  Condition:  {disease_name}")
        print(f"  Disease ID: {disease_class}")
        print(f"  Confidence: AF = {af_float:.6f}")
        print(f"  Summary:    Cardiac arrhythmia detected. Consult a cardiologist.")
    elif decision == FPGA_DEC_SCREENING:
        print(f"  Status:     SCREENING RECOMMENDED")
        print(f"  Confidence: AF = {af_float:.6f}")
        print(f"  Summary:    Inconclusive. Further screening is recommended.")
    else:
        print(f"  Status:     UNKNOWN ({dec_name})")
        print(f"  AF value:   {af_float:.6f}")

    print(f"  {'='*50}\n")


def run_realtime(
    data: np.ndarray,
    labels: np.ndarray,
    port: str,
    baud: int = 115200,
) -> None:
    """Interactive real-time diagnosis loop (4-lane parallel FPGA)."""

    try:
        import serial
    except ImportError:
        print("\nERROR: pyserial is required. Install with: pip install pyserial")
        return

    print(f"\n{'='*70}")
    print(f"FPGA REAL-TIME CARDIAC DIAGNOSIS SYSTEM  (4-Lane Parallel)")
    print(f"{'='*70}")
    print(f"  Serial port:  {port}")
    print(f"  Baud rate:    {baud}")
    print(f"  Parallel:     {N_LANES} patients per batch")
    print(f"  Dataset:      {data.shape[0]} users loaded for reference")
    print(f"{'='*70}")
    print(f"")
    print(f"  Commands:")
    print(f"    diagnose me            — Diagnose from loaded patient CSV")
    print(f"    diagnose <id>          — Diagnose 1 user from the dataset")
    print(f"    diagnose <id1> <id2>.. — Diagnose 1-4 users in parallel")
    print(f"    diagnose all           — Diagnose ALL users (batches of 4)")
    print(f"    load <path.csv>        — Load patient CSV for 'diagnose me'")
    print(f"    list                   — Show all users in the dataset")
    print(f"    features               — Show the 279 feature names")
    print(f"    quit                   — Exit")
    print(f"")

    print(f"  Connecting to FPGA on {port}...")
    try:
        ser = serial.Serial(port, baud, timeout=2.0)
    except Exception as e:
        print(f"\n  ERROR: Could not open {port}: {e}")
        print(f"  Check that the FPGA is connected and no other program has the port.")
        return

    time.sleep(1.0)
    print(f"  Connected!\n")

    patient_csv_path: Optional[str] = None

    while True:
        try:
            cmd = input("CDS> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not cmd:
            continue

        cmd_lower = cmd.lower()

        if cmd_lower in ('quit', 'exit', 'q'):
            print("  Goodbye!")
            break

        elif cmd_lower.startswith('load '):
            path = cmd[5:].strip().strip('"').strip("'")
            if os.path.isfile(path):
                patient_csv_path = path
                print(f"  Loaded patient file: {path}")
                try:
                    feats, lbls = load_patient_csv_multi(path)
                    n = feats.shape[0]
                    has_labels = lbls is not None
                    print(f"  Found {n} patient(s)" +
                          (f" with labels" if has_labels else " (no labels)"))
                except Exception as e:
                    print(f"  Warning: Could not preview file: {e}")
            else:
                print(f"  ERROR: File not found: {path}")

        elif cmd_lower == 'list':
            print(f"\n  Dataset: {data.shape[0]} users")
            print(f"  {'ID':>4}  {'Label':>5}  {'Status':>10}  {'Sex':>4}  {'Age':>4}")
            print(f"  {'-'*35}")
            for i in range(data.shape[0]):
                label = int(labels[i])
                status = "Healthy" if label == HEALTHY_CLASS else f"Disease {label}"
                sex = "M" if data[i, 1] == 0 else "F"
                age = f"{data[i, 0]:.0f}" if not np.isnan(data[i, 0]) else "?"
                print(f"  {i:4d}  {label:5d}  {status:>10}  {sex:>4}  {age:>4}")
            print()

        elif cmd_lower == 'features':
            print(f"\n  279 ECG Feature Names:")
            print(f"  {'Idx':>4}  {'Name':<25}")
            print(f"  {'-'*30}")
            for i in range(N_FEAT):
                print(f"  {i:4d}  {FEATURE_NAMES.get(i, f'feature_{i}'):<25}")
            print()

        elif cmd_lower.startswith('diagnose'):
            target = cmd[8:].strip()   # preserve original case for paths
            target_lower = target.lower()

            if target_lower == 'me':
                if patient_csv_path is None:
                    print("  No patient file loaded. Use: load <path.csv>")
                    continue
                try:
                    feats, _ = load_patient_csv_multi(patient_csv_path)
                except Exception as e:
                    print(f"  ERROR reading CSV: {e}")
                    continue

                n_patients = feats.shape[0]
                print(f"  Sending {n_patients} patient(s) to FPGA "
                      f"(batches of {N_LANES})...")

                for batch_start in range(0, n_patients, N_LANES):
                    batch_end = min(batch_start + N_LANES, n_patients)
                    batch_features = [feats[j] for j in range(batch_start, batch_end)]
                    n_real = len(batch_features)

                    results = send_batch_to_fpga(ser, batch_features)
                    if results is None:
                        print(f"  ERROR: FPGA timeout on batch "
                              f"{batch_start}-{batch_end-1}")
                        continue

                    for k in range(n_real):
                        decision, alarm_class, af_value = results[k]
                        label = f"Patient {batch_start + k + 1}/{n_patients}"
                        print_diagnosis(decision, alarm_class, af_value, label)

            elif target_lower == 'all':
                n_total = data.shape[0]
                n_batches = (n_total + N_LANES - 1) // N_LANES
                print(f"\n  Diagnosing ALL {n_total} users "
                      f"({n_batches} batches of {N_LANES})...")

                n_healthy_correct = 0
                n_diseased_correct = 0
                n_healthy = 0
                n_diseased = 0
                n_timeout = 0
                processed = 0

                for batch_start in range(0, n_total, N_LANES):
                    batch_end = min(batch_start + N_LANES, n_total)
                    batch_features = [data[j] for j in range(batch_start, batch_end)]
                    n_real = len(batch_features)

                    results = send_batch_to_fpga(ser, batch_features)

                    if results is None:
                        n_timeout += n_real
                        processed += n_real
                        continue

                    for k in range(n_real):
                        user_idx = batch_start + k
                        decision, alarm_class, af_value = results[k]
                        true_label = int(labels[user_idx])

                        if true_label == HEALTHY_CLASS:
                            n_healthy += 1
                            if decision != FPGA_DEC_UNHEALTHY:
                                n_healthy_correct += 1
                        else:
                            n_diseased += 1
                            if decision == FPGA_DEC_UNHEALTHY:
                                n_diseased_correct += 1

                    processed += n_real
                    if processed % 48 == 0 or processed >= n_total:
                        total_correct = n_healthy_correct + n_diseased_correct
                        total_responded = n_healthy + n_diseased
                        acc = (total_correct / total_responded * 100
                               if total_responded else 0)
                        print(f"  Progress: {processed}/{n_total}  "
                              f"accuracy={acc:.1f}%  timeouts={n_timeout}")

                total_correct = n_healthy_correct + n_diseased_correct
                total_responded = n_healthy + n_diseased
                acc = total_correct / total_responded * 100 if total_responded else 0
                sens = n_diseased_correct / n_diseased * 100 if n_diseased else 0
                spec = n_healthy_correct / n_healthy * 100 if n_healthy else 0

                print(f"\n  {'='*50}")
                print(f"  FULL DATASET RESULTS ({total_responded} users)")
                print(f"  {'='*50}")
                print(f"  Accuracy:    {acc:.1f}%  ({total_correct}/{total_responded})")
                print(f"  Sensitivity: {sens:.1f}%  ({n_diseased_correct}/{n_diseased})")
                print(f"  Specificity: {spec:.1f}%  ({n_healthy_correct}/{n_healthy})")
                print(f"  Timeouts:    {n_timeout}")
                print(f"  {'='*50}\n")

            else:
                # Parse space-separated user IDs
                parts = target_lower.split()
                user_ids = []
                bad_input = False
                for p in parts:
                    try:
                        uid = int(p)
                        if uid < 0 or uid >= data.shape[0]:
                            print(f"  ERROR: User ID {uid} out of range "
                                  f"(0-{data.shape[0]-1})")
                            bad_input = True
                            break
                        user_ids.append(uid)
                    except ValueError:
                        print(f"  Usage: diagnose me | diagnose <id> [<id2> ...] "
                              f"| diagnose all")
                        bad_input = True
                        break

                if bad_input or not user_ids:
                    continue

                if len(user_ids) > N_LANES:
                    print(f"  ERROR: Max {N_LANES} users per batch. "
                          f"Got {len(user_ids)}.")
                    continue

                n_users = len(user_ids)
                batch_features = [data[uid] for uid in user_ids]

                print(f"  Sending {n_users} user(s) to FPGA "
                      f"(padded to {N_LANES} lanes)...")

                results = send_batch_to_fpga(ser, batch_features)
                if results is None:
                    print("  ERROR: FPGA did not respond (timeout)")
                    continue

                for k, uid in enumerate(user_ids):
                    true_label = int(labels[uid])
                    true_status = ("Healthy" if true_label == HEALTHY_CLASS
                                   else DISEASE_NAMES.get(true_label,
                                                          f"Disease {true_label}"))
                    decision, alarm_class, af_value = results[k]
                    print_diagnosis(decision, alarm_class, af_value,
                                    f"User {uid} (True: {true_status})")

        elif cmd_lower in ('help', '?'):
            print(f"  Commands:")
            print(f"    diagnose me            — Diagnose from loaded patient CSV")
            print(f"    diagnose <id>          — Diagnose 1 user from the dataset")
            print(f"    diagnose <id1> <id2>.. — Diagnose up to {N_LANES} users in parallel")
            print(f"    diagnose all           — Diagnose ALL dataset users")
            print(f"    load <path.csv>        — Load a patient CSV file")
            print(f"    list                   — List all dataset users")
            print(f"    features               — Show feature names")
            print(f"    quit                   — Exit")

        else:
            print(f"  Unknown command: '{cmd}'. Type 'help' for commands.")

    ser.close()
    print("  Serial port closed.")

def main():
    parser = argparse.ArgumentParser(
        description="FPGA Real-Time Cardiac Diagnosis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:

  1. Setup (train model, export .mem files):
     python fpga_realtime_monitor.py --setup

  2. Load .mem files into Vivado, synthesize, program FPGA

  3. Run real-time diagnosis:
     python fpga_realtime_monitor.py --run --port COM4

  4. At the CDS> prompt:
     - Type 'diagnose me' after loading a patient CSV
     - Type 'diagnose 42' to test user #42 from the dataset
     - Type 'diagnose all' to run all 452 users through the FPGA
        """,
    )
    parser.add_argument("--setup", action="store_true",
                        help="Train model on full dataset and export .mem files")
    parser.add_argument("--run", action="store_true",
                        help="Start real-time diagnosis mode")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to arrhythmia.data")
    parser.add_argument("--port", type=str, default="COM4",
                        help="Serial port for FPGA (default: COM4)")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Baud rate (default: 115200)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for .mem files (default: fpga_mem/)")
    args = parser.parse_args()

    if not args.setup and not args.run:
        parser.print_help()
        print("\n  ERROR: Specify --setup or --run")
        return

    # Load dataset
    data_path = args.data or str(
        CDS_NI_ROOT / "CDS_NI_Algorithms" / "data" / "arrhythmia.data"
    )
    print(f"Loading dataset: {data_path}")
    data, labels = load_dataset(data_path)
    print(f"  {data.shape[0]} users, {data.shape[1]} features")

    output_dir = args.output or str(PROJECT_ROOT / "fpga_mem")

    if args.setup:
        run_setup(data, labels, output_dir)

    if args.run:
        run_realtime(data, labels, args.port, args.baud)


if __name__ == "__main__":
    main()
