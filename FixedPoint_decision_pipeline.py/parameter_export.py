"""
FPGA Parameter Export — Convert trained CDS model to Verilog-loadable .mem files

Exports all pretrained data that Algorithm 4 needs at inference time,
split into 6 BRAM-aligned .mem files matching the Verilog module interfaces:

  1. tree_topology.mem  — 3 words/node (feat_idx, branch_low, branch_high)
  2. action_hdr.mem     — header section: (node*12+disease) -> (count, start_addr)
  3. action_data.mem    — data section: flat (feature_idx, r_j_h) pairs
  4. prob_phf.mem       — P(h,f) per (node, disease), Q s1.15
  5. prob_pgt1.mem      — 1/P(h>1,f) reciprocal per node, Q s3.13
  6. healthy_ranges.mem — direct-addressed 32-bit: {b_min, b_max} at addr {node, feat}

Output format: Verilog $readmemh compatible (.mem files with hex values).

All float values are converted to fixed-point integers before export:
  - Sensor/branch values: Q s11.4 (16-bit signed)
  - Probabilities/weights: Q s1.15 (16-bit signed)
  - Reciprocals:           Q s3.13 (16-bit signed)
  - Threshold:             Q s2.30 (32-bit signed)
"""

from __future__ import annotations

import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- Import project modules ---
sys.path.insert(0, str(Path(__file__).parent.parent))
from CDS_NI_Algorithms.build_decision_tree import (
    DecisionTree, TreeNode, HEALTHY_CLASS, N_FEATURES,
    load_dataset, build_decision_tree,
)
from CDS_NI_Algorithms.action_normalRange import (
    Algorithm2Output, run_algorithm2, DEFAULT_N_BINS,
)
from CDS_NI_Algorithms.action_pruning import Algorithm3Output, run_algorithm3

# --- Import fixed-point functions and Algorithm 4 from the golden model ---
from decision_pipeline_fixedPoint import (
    to_fixed, fixed_divide,
    run_algorithm4, HealthDecision, PredictionRecord,
)


# =====================================================================
# Constants
# =====================================================================

# Disease classes in the arrhythmia dataset (non-healthy)
# Mapped to contiguous offsets 0-11 for efficient BRAM addressing
DISEASE_CLASSES: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16)
N_DISEASES: int = len(DISEASE_CLASSES)

# Map each disease class to a contiguous index
DISEASE_TO_OFFSET: Dict[int, int] = {cls: i for i, cls in enumerate(DISEASE_CLASSES)}

# Maximum actions per (node, disease) pair — pad shorter lists with sentinel
# This is set generously; the export will report the actual max encountered
MAX_ACTIONS_PER_PAIR: int = 32

# Sentinel value indicating "no valid entry" in BRAM
SENTINEL_16: int = 0xFFFF    # for 16-bit fields
SENTINEL_9: int = 0x1FF      # for 9-bit feature indices (max valid = 278)


# =====================================================================
# Hex formatting
# =====================================================================

def to_hex_16(val: int) -> str:
    """Convert a signed 16-bit fixed-point integer to 4-digit hex string.
    Uses two's complement for negative values."""
    if val < 0:
        val = val + (1 << 16)
    return format(val & 0xFFFF, '04X')


def to_hex_32(val: int) -> str:
    """Convert a signed 32-bit fixed-point integer to 8-digit hex string."""
    if val < 0:
        val = val + (1 << 32)
    return format(val & 0xFFFFFFFF, '08X')


# =====================================================================
# Node indexing
# =====================================================================

def build_node_index(tree: DecisionTree, nodes_filter: List[str]) -> Dict[str, int]:
    """Assign each active tree node a sequential integer index.

    Only nodes in nodes_filter get an index (these are the nodes that
    Algorithm 2/3 trained on).  Root is always index 0.

    Returns: dict mapping node_id string -> integer index
    """
    # Start with root
    index_map: Dict[str, int] = {}
    idx = 0

    # Root first (always index 0)
    root_id = tree.root.node_id
    if root_id in nodes_filter:
        index_map[root_id] = idx
        idx += 1

    # Then level-2 nodes in the order they appear in nodes_filter
    for nid in nodes_filter:
        if nid not in index_map:
            index_map[nid] = idx
            idx += 1

    return index_map


# =====================================================================
# Export 1: Tree Topology
# =====================================================================

def export_tree_topology(
    tree: DecisionTree,
    node_index: Dict[str, int],
    output_path: str,
) -> None:
    """Export tree structure to .mem file (3 words per node).

    Addressed by tree_traversal as: node_counter * 3 + word_offset
    tree_traversal reads tree_data[7:0] for feature index.

        Word 0: branch_feature_idx (16 bits, low 9 significant)
        Word 1: branch_low  in Q s11.4 (16 bits)
        Word 2: branch_high in Q s11.4 (16 bits)

    Total: 3 words × 16 bits = 48 bits per node.
    Address width: ceil(log2(n_nodes * 3)) bits.
    """
    n_nodes = len(node_index)
    idx_to_nid = {idx: nid for nid, idx in node_index.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"// Tree Topology — {n_nodes} nodes, 3 words each\n")
        f.write(f"// Addr = node_index * 3 + word_offset\n")
        f.write(f"// Format per node: 3 x 16-bit words\n")
        f.write(f"//   Word 0: branch_feature_idx (low 9 bits significant)\n")
        f.write(f"//   Word 1: branch_low  (Q s11.4)\n")
        f.write(f"//   Word 2: branch_high (Q s11.4)\n")
        f.write(f"// Total entries: {n_nodes * 3}\n")
        f.write(f"//\n")

        for idx in range(n_nodes):
            nid = idx_to_nid[idx]
            node = tree.all_nodes[nid]

            branch_feat = node.branching_feat_k if node.branch_def is not None else 0

            if node.branch_def is not None:
                branch_low = to_fixed(node.branch_def.low, 11, 4)
                branch_high = to_fixed(node.branch_def.high, 11, 4)
            else:
                branch_low = to_fixed(-1024.0, 11, 4)
                branch_high = to_fixed(1023.0, 11, 4)

            f.write(f"// Node {idx}: {nid}\n")
            f.write(f"{to_hex_16(branch_feat & 0x1FF)}\n")
            f.write(f"{to_hex_16(branch_low)}\n")
            f.write(f"{to_hex_16(branch_high)}\n")

    print(f"  Tree topology: {n_nodes} nodes x 3 words = {n_nodes * 3} entries -> {output_path}")


# =====================================================================
# Export 2: Healthy Ranges (direct-addressed, 32-bit wide)
# =====================================================================

HR_NODE_BITS: int = 8
HR_FEAT_BITS: int = 9
HR_ADDR_DEPTH: int = (1 << (HR_NODE_BITS + HR_FEAT_BITS))  # 131072

HR_SENTINEL_BMIN: int = 0x7FFF
HR_SENTINEL_BMAX: int = 0x8000


def export_healthy_ranges(
    alg2_output: Algorithm2Output,
    node_index: Dict[str, int],
    output_path: str,
) -> None:
    """Export healthy ranges as direct-addressed 32-bit BRAM.

    Address = {node_idx[7:0], feature_idx[8:0]}  (17 bits, 131072 entries)
    Data    = {b_min[15:0], b_max[15:0]}          (32 bits per entry)

    af_engine drives hr_read_addr = {node_idx, latched_feature_idx} and
    model_rom splits the 32-bit output:
        hr_bmin = data[31:16],  hr_bmax = data[15:0]

    Empty entries use sentinel {0x7FFF, 0x8000} (b_min > b_max) so
    rangeComparator sets invalid_range=true -> triggered=false.
    """
    populated: Dict[int, Tuple[int, int]] = {}

    for (nid, feat_idx), model in alg2_output.perceptor_index.items():
        if nid not in node_index:
            continue
        n_idx = node_index[nid]
        b_min = to_fixed(model.healthy_range.b_min_healthy, 9, 4)
        b_max = to_fixed(model.healthy_range.b_max_healthy, 9, 4)
        addr = (n_idx << HR_FEAT_BITS) | (feat_idx & ((1 << HR_FEAT_BITS) - 1))
        populated[addr] = (b_min, b_max)

    n_populated = len(populated)
    sentinel_word = f"{HR_SENTINEL_BMIN:04X}{HR_SENTINEL_BMAX:04X}"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"// Healthy Ranges — direct-addressed, 32-bit wide BRAM\n")
        f.write(f"// Addr = {{node_idx[7:0], feature_idx[8:0]}} (17 bits)\n")
        f.write(f"// Data = {{b_min[15:0], b_max[15:0]}} (32 bits)\n")
        f.write(f"// Depth: {HR_ADDR_DEPTH}, Populated: {n_populated}\n")
        f.write(f"// Sentinel: {sentinel_word} (b_min > b_max -> invalid_range)\n")
        f.write(f"//\n")

        for addr in range(HR_ADDR_DEPTH):
            if addr in populated:
                b_min, b_max = populated[addr]
                f.write(f"{b_min & 0xFFFF:04X}{b_max & 0xFFFF:04X}\n")
            else:
                f.write(f"{sentinel_word}\n")

    print(f"  Healthy ranges: {HR_ADDR_DEPTH} entries ({n_populated} populated) -> {output_path}")


# =====================================================================
# Export 3a: Action Header BRAM
# =====================================================================

def _build_action_groups(
    alg3_output: Algorithm3Output,
    node_index: Dict[str, int],
) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, int]]], int]:
    """Shared helper: collect (node, disease) -> [(feat, weight_fixed)] groups."""
    action_groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    max_actions = 0

    for nid, n_idx in node_index.items():
        for disease_h in DISEASE_CLASSES:
            actions = alg3_output.retained_for_node_disease(nid, disease_h)
            if not actions:
                continue

            fixed_actions = []
            for a in actions:
                if a.action_weight <= 0.0:
                    continue
                fixed_actions.append((a.feature_idx, to_fixed(a.action_weight, 1, 15)))

            if fixed_actions:
                d_off = DISEASE_TO_OFFSET[disease_h]
                action_groups[(n_idx, d_off)] = fixed_actions
                max_actions = max(max_actions, len(fixed_actions))

    return action_groups, max_actions


def export_action_hdr(
    alg3_output: Algorithm3Output,
    node_index: Dict[str, int],
    output_path: str,
) -> None:
    """Export action header BRAM (separate .mem file).

    Addressed by af_engine as: action_hdr_addr = (node_idx * 12 + disease_offset) * 2 + word
    Format: 2 x 16-bit words per slot
        Word 0: action_count
        Word 1: start_address in action_data BRAM (word-pair index)

    Total entries: n_nodes * 12 * 2 words.
    """
    action_groups, max_actions = _build_action_groups(alg3_output, node_index)

    n_nodes = len(node_index)
    total_slots = n_nodes * N_DISEASES

    # Assign data addresses (word-pair index, not raw word offset)
    header_table: Dict[int, Tuple[int, int]] = {}
    data_addr = 0
    for (n_idx, d_off) in sorted(action_groups.keys()):
        count = len(action_groups[(n_idx, d_off)])
        flat = n_idx * N_DISEASES + d_off
        header_table[flat] = (count, data_addr)
        data_addr += count

    total_actions = data_addr
    populated = len(header_table)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"// Action Header BRAM — {total_slots} slots ({n_nodes} nodes x {N_DISEASES} diseases)\n")
        f.write(f"// {populated} active groups, {total_actions} total actions\n")
        f.write(f"// Addr = (node_index * {N_DISEASES} + disease_offset) * 2 + word\n")
        f.write(f"// Format: 2 x 16-bit words per slot\n")
        f.write(f"//   Word 0: action_count\n")
        f.write(f"//   Word 1: start_address in action_data BRAM\n")
        f.write(f"// Total entries: {total_slots * 2}\n")
        f.write(f"//\n")

        for slot in range(total_slots):
            n_idx = slot // N_DISEASES
            d_off = slot % N_DISEASES
            count, start = header_table.get(slot, (0, 0))

            if count > 0:
                f.write(f"// [{slot}] node={n_idx}, disease_off={d_off} "
                        f"(class={DISEASE_CLASSES[d_off]}), "
                        f"count={count}, data@{start}\n")
            f.write(f"{to_hex_16(count)}\n")
            f.write(f"{to_hex_16(start)}\n")

    print(f"  Action header: {total_slots * 2} entries "
          f"({populated} active groups) -> {output_path}")


# =====================================================================
# Export 3b: Action Data BRAM
# =====================================================================

def export_action_data(
    alg3_output: Algorithm3Output,
    node_index: Dict[str, int],
    output_path: str,
) -> None:
    """Export action data BRAM (separate .mem file).

    Addressed by af_engine as: action_data_addr = (start_addr + action_idx) * 2 + word
    Format: 2 x 16-bit words per action
        Word 0: feature_idx (low 9 bits significant)
        Word 1: r_j_h (Q s1.15)
    """
    action_groups, max_actions = _build_action_groups(alg3_output, node_index)

    data_words: List[Tuple[int, int, int, int]] = []
    for (n_idx, d_off) in sorted(action_groups.keys()):
        for feat, weight in action_groups[(n_idx, d_off)]:
            data_words.append((feat, weight, n_idx, d_off))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"// Action Data BRAM — {len(data_words)} actions\n")
        f.write(f"// Max per (node, disease): {max_actions}\n")
        f.write(f"// Addr = (start_addr + action_idx) * 2 + word\n")
        f.write(f"// Format: 2 x 16-bit words per action\n")
        f.write(f"//   Word 0: feature_idx (low 9 bits significant)\n")
        f.write(f"//   Word 1: r_j_h (Q s1.15)\n")
        f.write(f"// Total entries: {len(data_words) * 2}\n")
        f.write(f"//\n")

        for feat, weight, n_idx, d_off in data_words:
            f.write(f"// node={n_idx}, disease={DISEASE_CLASSES[d_off]}, feat={feat}\n")
            f.write(f"{to_hex_16(feat)}\n")
            f.write(f"{to_hex_16(weight)}\n")

    print(f"  Action data: {len(data_words) * 2} entries "
          f"({len(data_words)} actions, max {max_actions}/group) -> {output_path}")


# =====================================================================
# Export 4a: P(h,f) BRAM
# =====================================================================

def export_prob_phf(
    tree: DecisionTree,
    node_index: Dict[str, int],
    output_path: str,
) -> None:
    """Export P(h,f) probability table as its own BRAM .mem file.

    Addressed by af_engine as: prob_phf_addr = node_idx * 12 + disease_offset
    Each entry: Q s1.15 (16 bits)
    Total: n_nodes * 12 entries.
    """
    n_nodes = len(node_index)
    idx_to_nid = {idx: nid for nid, idx in node_index.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"// P(h,f) BRAM — {n_nodes * N_DISEASES} entries\n")
        f.write(f"// Addr = node_index * {N_DISEASES} + disease_offset\n")
        f.write(f"// Each entry: Q s1.15 (16 bits)\n")
        f.write(f"//\n")

        for idx in range(n_nodes):
            nid = idx_to_nid[idx]
            node = tree.all_nodes[nid]

            for d_off, disease_h in enumerate(DISEASE_CLASSES):
                count_h = node.health_dist.get(disease_h, 0)
                n_users = node.n_users

                if n_users == 0:
                    p_h_f = 0
                else:
                    p_h_f = fixed_divide(count_h, n_users, 15)

                addr = idx * N_DISEASES + d_off
                f.write(f"// [{addr:4d}] node={idx} ({nid}), disease={disease_h}: "
                        f"{count_h}/{n_users}\n")
                f.write(f"{to_hex_16(p_h_f)}\n")

    print(f"  prob_phf: {n_nodes * N_DISEASES} entries -> {output_path}")


# =====================================================================
# Export 4b: 1/P(h>1,f) Reciprocal BRAM
# =====================================================================

def export_prob_pgt1(
    tree: DecisionTree,
    node_index: Dict[str, int],
    output_path: str,
) -> None:
    """Export 1/P(h>1,f) reciprocal table as its own BRAM .mem file.

    Addressed by af_engine as: prob_pgt1_addr = node_idx
    Each entry: Q s3.13 (16 bits, max ~3.9999)
    Total: n_nodes entries.

    af_engine feeds this directly to fixedDivide.reciprocal_denominator.
    fixedDivide shifts the 48-bit product right by 13 to produce Q s2.30.
    """
    n_nodes = len(node_index)
    idx_to_nid = {idx: nid for nid, idx in node_index.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"// 1/P(h>1,f) Reciprocal BRAM — {n_nodes} entries\n")
        f.write(f"// Addr = node_index\n")
        f.write(f"// Each entry: Q s3.13 (16 bits, max ~3.9999)\n")
        f.write(f"// Used by fixedDivide.reciprocal_denominator\n")
        f.write(f"//\n")

        for idx in range(n_nodes):
            nid = idx_to_nid[idx]
            node = tree.all_nodes[nid]
            n_users = node.n_users
            n_diseased = node.n_diseased

            if n_users == 0 or n_diseased == 0:
                p_gt1_float = 1.0
            else:
                p_gt1_float = n_diseased / n_users

            recip = to_fixed(1.0 / p_gt1_float, 3, 13)

            f.write(f"// [{idx:4d}] node={idx} ({nid}): 1/({n_diseased}/{n_users})"
                    f" = {1.0 / p_gt1_float:.6f}\n")
            f.write(f"{to_hex_16(recip)}\n")

    print(f"  prob_pgt1: {n_nodes} entries -> {output_path}")


# =====================================================================
# Export Constants
# =====================================================================

def export_constants(
    node_index: Dict[str, int],
    output_path: str,
) -> None:
    """Export Verilog-ready parameter header file.

    Not a .mem file — this is a .vh (Verilog header) that defines
    localparam constants for the FPGA design.
    """
    n_nodes = len(node_index)
    threshold_fixed = to_fixed(0.025, 2, 30)
    one_fixed = 1 << 30

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"// Auto-generated CDS model parameters\n")
        f.write(f"// Do not edit — regenerate with parameter_export.py\n\n")

        f.write(f"// Model dimensions\n")
        f.write(f"localparam N_NODES       = {n_nodes};\n")
        f.write(f"localparam N_FEATURES    = {N_FEATURES};\n")
        f.write(f"localparam N_DISEASES    = {N_DISEASES};\n\n")

        f.write(f"// Disease class mapping (class -> contiguous offset)\n")
        for cls, offset in DISEASE_TO_OFFSET.items():
            f.write(f"// Class {cls:2d} -> offset {offset}\n")
        f.write(f"\n")

        f.write(f"// Fixed-point constants (Q s2.30)\n")
        f.write(f"localparam signed [31:0] ONE_FP       = 32'sh{to_hex_32(one_fixed)};  "
                f"// 1.0\n")
        f.write(f"localparam signed [31:0] THRESHOLD_FP = 32'sh{to_hex_32(threshold_fixed)};  "
                f"// 0.025\n\n")

        f.write(f"// Node index mapping\n")
        for nid, idx in sorted(node_index.items(), key=lambda x: x[1]):
            f.write(f"// Node {idx:3d} = \"{nid}\"\n")

    print(f"  Constants header -> {output_path}")


# =====================================================================
# Master export function
# =====================================================================

def export_model_parameters(
    tree: DecisionTree,
    alg2_output: Algorithm2Output,
    alg3_output: Algorithm3Output,
    nodes_filter: List[str],
    output_dir: str,
) -> Dict[str, int]:
    """Export all trained CDS model parameters to FPGA-loadable .mem files.

    Args:
        tree:         Trained decision tree from Algorithm 1
        alg2_output:  Perceptor/executive library from Algorithm 2
        alg3_output:  Refined action library from Algorithm 3
        nodes_filter: List of node_id strings that were trained
        output_dir:   Directory to write .mem files into

    Returns:
        node_index mapping (node_id -> integer index)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"FPGA PARAMETER EXPORT")
    print(f"{'='*60}")

    # Step 1: Assign numeric indices to nodes
    node_index = build_node_index(tree, nodes_filter)
    print(f"  Nodes indexed: {len(node_index)}")
    for nid, idx in sorted(node_index.items(), key=lambda x: x[1]):
        node = tree.all_nodes[nid]
        print(f"    [{idx:3d}] {nid} (level={node.focus_level}, "
              f"users={node.n_users}, diseased={node.n_diseased})")

    # Step 2: Export each BRAM-aligned .mem file
    export_tree_topology(tree, node_index,
                         os.path.join(output_dir, "tree_topology.mem"))

    export_healthy_ranges(alg2_output, node_index,
                          os.path.join(output_dir, "healthy_ranges.mem"))

    export_action_hdr(alg3_output, node_index,
                      os.path.join(output_dir, "action_hdr.mem"))

    export_action_data(alg3_output, node_index,
                       os.path.join(output_dir, "action_data.mem"))

    export_prob_phf(tree, node_index,
                    os.path.join(output_dir, "prob_phf.mem"))

    export_prob_pgt1(tree, node_index,
                     os.path.join(output_dir, "prob_pgt1.mem"))

    export_constants(node_index,
                     os.path.join(output_dir, "cds_params.vh"))

    print(f"{'='*60}")
    print(f"  All exports written to: {output_dir}/")
    print(f"{'='*60}\n")

    return node_index


# =====================================================================
# Export Test Vectors (stimulus for FPGA testbench)
# =====================================================================

def export_test_vectors(
    data: np.ndarray,
    labels: np.ndarray,
    test_indices: List[int],
    output_path: str,
) -> None:
    """Export test user feature vectors and expected labels to .mem file.

    Each test user occupies N_FEATURES + 1 words:
        Words 0..N_FEATURES-1: feature values in Q s11.4 (16 bits each)
        Word N_FEATURES:       ground truth label (16 bits unsigned)

    First line is the number of test users.
    """
    n_test = len(test_indices)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"// Test Vectors — {n_test} users, {N_FEATURES} features each\n")
        f.write(f"// Format per user: {N_FEATURES} x feature (Q s11.4) + 1 x label\n")
        f.write(f"//\n")
        f.write(f"// User count:\n")
        f.write(f"{to_hex_16(n_test)}\n")

        for user_idx in test_indices:
            label = int(labels[user_idx])
            f.write(f"// User {user_idx} (label={label})\n")

            for feat_j in range(N_FEATURES):
                val = float(data[user_idx, feat_j])
                fixed_val = 0 if np.isnan(val) else to_fixed(val, 11, 4)
                f.write(f"{to_hex_16(fixed_val)}\n")

            f.write(f"{to_hex_16(label)}\n")

    print(f"  Test vectors: {n_test} users -> {output_path}")


# =====================================================================
# Export Golden Predictions (expected FPGA output)
# =====================================================================

# Decision encoding for .mem file (matches what FPGA FSM should output)
DECISION_ENCODING = {
    HealthDecision.HEALTHY: 0x00,
    HealthDecision.UNHEALTHY: 0x01,
    HealthDecision.SCREENING: 0x02,
    HealthDecision.UNKNOWN: 0x03,
}


def export_golden_predictions(
    data: np.ndarray,
    labels: np.ndarray,
    test_indices: List[int],
    tree: 'DecisionTree',
    alg2_output: 'Algorithm2Output',
    alg3_output: 'Algorithm3Output',
    output_path: str,
    rng_seed: int = 42,
) -> List[PredictionRecord]:
    """Run the Python golden model (Algorithm 4) on test users and export predictions.

    For each test user, exports:
        Word 0: [decision(2) | is_correct(1) | padding(5) | alarm_class(8)] = 16 bits
        Word 1: total_actions_applied (16 bits unsigned)
        Word 2: final AF value (Q s2.30, high 16 bits)
        Word 3: final AF value (Q s2.30, low 16 bits)

    Also exports a per-user AF trace summary for detailed RTL debugging.

    Returns the list of PredictionRecords for upstream accuracy reporting.
    """
    n_test = len(test_indices)
    records: List[PredictionRecord] = []

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"// Golden Model Predictions — {n_test} test users\n")
        f.write(f"// Generated by Python fixed-point Algorithm 4\n")
        f.write(f"// Format per user: 4 x 16-bit words\n")
        f.write(f"//   Word 0: [decision(2) | is_correct(1) | pad(5) | alarm_class(8)]\n")
        f.write(f"//   Word 1: total_actions_applied\n")
        f.write(f"//   Word 2: final_AF[31:16]  (Q s2.30 high)\n")
        f.write(f"//   Word 3: final_AF[15:0]   (Q s2.30 low)\n")
        f.write(f"//\n")
        f.write(f"// Decision encoding: HEALTHY=0x00, UNHEALTHY=0x01, SCREENING=0x02\n")
        f.write(f"//\n")
        f.write(f"// User count:\n")
        f.write(f"{to_hex_16(n_test)}\n")

        for user_idx in test_indices:
            record = run_algorithm4(
                user_idx, data, labels, tree, alg2_output, alg3_output, rng_seed
            )
            records.append(record)

            # Encode decision
            decision_code = DECISION_ENCODING.get(record.decision, 0x03)
            correct_bit = 1 if record.is_correct else 0
            alarm_class = record.alarm_class if record.alarm_class is not None else 0xFF

            # Word 0: [decision(2) | is_correct(1) | pad(5) | alarm_class(8)]
            word0 = ((decision_code & 0x3) << 14) | ((correct_bit & 0x1) << 13) | (alarm_class & 0xFF)

            # Word 1: total actions applied
            word1 = min(record.total_actions_applied, 0xFFFF)

            # Word 2-3: final AF as 32-bit Q s2.30
            # Get final AF from last trace step, or 0 if no trace
            if record.af_trace:
                final_af = record.af_trace[-1].AF_real
            else:
                final_af = 0

            # Two's complement for 32-bit
            if final_af < 0:
                final_af_unsigned = final_af + (1 << 32)
            else:
                final_af_unsigned = final_af & 0xFFFFFFFF

            word2 = (final_af_unsigned >> 16) & 0xFFFF  # high 16
            word3 = final_af_unsigned & 0xFFFF           # low 16

            f.write(f"// User {user_idx} (label={record.true_label}): "
                    f"decision={record.decision.value}, correct={record.is_correct}, "
                    f"actions={record.total_actions_applied}, "
                    f"AF=0x{final_af_unsigned:08X}\n")
            f.write(f"{to_hex_16(word0)}\n")
            f.write(f"{to_hex_16(word1)}\n")
            f.write(f"{to_hex_16(word2)}\n")
            f.write(f"{to_hex_16(word3)}\n")

    # Summary stats
    n_correct = sum(1 for r in records if r.is_correct)
    accuracy = n_correct / n_test * 100 if n_test > 0 else 0.0
    print(f"  Golden predictions: {n_test} users, accuracy={accuracy:.1f}% -> {output_path}")

    return records


def export_af_trace(
    records: List[PredictionRecord],
    test_indices: List[int],
    output_path: str,
) -> None:
    """Export detailed AF trace for each test user (for RTL step-by-step debugging).

    Each user's trace is a sequence of entries:
        Word 0: [step_type(2) | feature_idx(9) | disease_offset(4) | pad(1)] = 16 bits
        Word 1: delta_AF high 16 bits (Q s2.30)
        Word 2: delta_AF low 16 bits  (Q s2.30)
        Word 3: cumulative AF high 16 bits (Q s2.30)
        Word 4: cumulative AF low 16 bits  (Q s2.30)

    Users are separated by a sentinel word (0xFFFF).
    First line is total number of test users.
    """
    STEP_TYPE_CODE = {"pac": 0, "rl_sim": 1, "branch_route": 2, "threshold_check": 3}

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"// AF Trace — detailed step-by-step golden model output\n")
        f.write(f"// For RTL debugging: compare each FPGA AF update against this trace\n")
        f.write(f"// Users separated by sentinel 0xFFFF\n")
        f.write(f"//\n")
        f.write(f"// User count:\n")
        f.write(f"{to_hex_16(len(records))}\n")

        total_steps = 0
        for record, user_idx in zip(records, test_indices):
            f.write(f"// --- User {user_idx} (label={record.true_label}, "
                    f"decision={record.decision.value}, steps={len(record.af_trace)}) ---\n")

            for step in record.af_trace:
                step_code = STEP_TYPE_CODE.get(step.step_type, 0)
                feat = step.feature_idx if step.feature_idx >= 0 else 0x1FF
                disease_off = DISEASE_TO_OFFSET.get(step.disease_class, 0xF)

                # Word 0: [step_type(2) | feature_idx(9) | disease_offset(4) | pad(1)]
                word0 = ((step_code & 0x3) << 14) | ((feat & 0x1FF) << 5) | ((disease_off & 0xF) << 1)

                # delta_AF as 32-bit
                daf = step.delta_AF
                if daf < 0:
                    daf_u = daf + (1 << 32)
                else:
                    daf_u = daf & 0xFFFFFFFF

                # cumulative AF as 32-bit
                af = step.AF_real
                if af < 0:
                    af_u = af + (1 << 32)
                else:
                    af_u = af & 0xFFFFFFFF

                f.write(f"{to_hex_16(word0)}\n")
                f.write(f"{to_hex_16((daf_u >> 16) & 0xFFFF)}\n")
                f.write(f"{to_hex_16(daf_u & 0xFFFF)}\n")
                f.write(f"{to_hex_16((af_u >> 16) & 0xFFFF)}\n")
                f.write(f"{to_hex_16(af_u & 0xFFFF)}\n")
                total_steps += 1

            # Sentinel separating users
            f.write(f"FFFF\n")

    print(f"  AF trace: {total_steps} steps across {len(records)} users -> {output_path}")


# =====================================================================
# 10-Fold Cross Validation Export
# =====================================================================

def export_all_folds(
    data: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    rng_seed: int = 42,
    max_users: Optional[int] = None,
) -> None:
    """Train 10 models (one per fold) and export each to its own subdirectory.

    For each fold k (0-9):
      - Trains Algorithms 1-3 on the 90% training partition
      - Exports model .mem files to output_dir/fold_k/
      - Exports the 10% test partition as test_vectors.mem

    The FPGA testbench loads fold_k/ model files, feeds test_vectors.mem
    as stimulus, and collects predictions. Aggregating across all 10 folds
    gives the full cross-validated accuracy.

    Uses the same deterministic split logic as decision_pipeline.ten_fold_cv().
    """
    n_total = data.shape[0] if max_users is None else min(max_users, data.shape[0])

    random.seed(rng_seed)
    indices = list(range(n_total))
    random.shuffle(indices)
    fold_size = (n_total + 9) // 10

    print(f"\n{'='*60}")
    print(f"10-FOLD CROSS VALIDATION EXPORT")
    print(f"  Users: {n_total}, Fold size: ~{fold_size}")
    print(f"  Output: {output_dir}/fold_0/ .. fold_9/")
    print(f"{'='*60}\n")

    for fold in range(10):
        start_idx = fold * fold_size
        end_idx = min(start_idx + fold_size, n_total)
        test_indices = indices[start_idx:end_idx]
        train_indices = [idx for idx in indices if idx not in test_indices]

        train_data = data[train_indices]
        train_labels = labels[train_indices]

        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        print(f"--- Fold {fold}/9: train={len(train_indices)}, test={len(test_indices)} ---")

        # Algorithm 1: Build decision tree on training partition
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

        # Algorithm 2: Train perceptor/executive on training partition
        alg2_i = run_algorithm2(tree_i, train_data, train_labels, DEFAULT_N_BINS, nodes_filter)
        print(f"  Alg2: {alg2_i.n_perceptor_entries} perceptor, {alg2_i.n_executive_entries} executive")

        # Algorithm 3: Refine actions on training partition
        alg3_i = run_algorithm3(alg2_i, tree_i, train_data, train_labels, nodes_filter, reset_per_h=False)
        print(f"  Alg3: {len(alg3_i.refined_actions)} retained, {len(alg3_i.removed_actions)} removed")

        # Export model parameters for this fold
        export_model_parameters(tree_i, alg2_i, alg3_i, nodes_filter, fold_dir)

        # Export test vectors (using original full data so indices map correctly)
        export_test_vectors(data, labels, test_indices,
                           os.path.join(fold_dir, "test_vectors.mem"))

        # Run golden model (Algorithm 4) on test users and export expected results
        print(f"  Running golden model on {len(test_indices)} test users...")
        golden_records = export_golden_predictions(
            data, labels, test_indices, tree_i, alg2_i, alg3_i,
            os.path.join(fold_dir, "expected_output.mem"),
            rng_seed=rng_seed,
        )

        # Export detailed AF trace for RTL step-by-step verification
        export_af_trace(golden_records, test_indices,
                        os.path.join(fold_dir, "af_trace.mem"))

        # Accumulate cross-fold stats
        fold_correct = sum(1 for r in golden_records if r.is_correct)
        fold_acc = fold_correct / len(test_indices) * 100
        print(f"  Fold {fold}/9 accuracy: {fold_acc:.1f}% "
              f"({fold_correct}/{len(test_indices)})\n")

    # Write a manifest summarizing all folds
    manifest_path = os.path.join(output_dir, "cv_manifest.txt")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(f"// 10-Fold Cross Validation Manifest\n")
        f.write(f"// Generated with rng_seed={rng_seed}, n_users={n_total}\n")
        f.write(f"// Each fold_k/ directory contains:\n")
        f.write(f"//   tree_topology.mem   - 3 words/node (feat, low, high)\n")
        f.write(f"//   healthy_ranges.mem  - direct-addressed 32-bit {{bmin,bmax}}\n")
        f.write(f"//   action_hdr.mem      - header: (count, start_addr) per slot\n")
        f.write(f"//   action_data.mem     - data: (feature_idx, r_j_h) per action\n")
        f.write(f"//   prob_phf.mem        - P(h,f) per (node, disease)\n")
        f.write(f"//   prob_pgt1.mem       - 1/P(h>1,f) reciprocal per node\n")
        f.write(f"//   cds_params.vh       - Verilog constants header\n")
        f.write(f"//   test_vectors.mem    - test user stimulus (features + ground truth)\n")
        f.write(f"//   expected_output.mem - golden model predictions (FPGA must match)\n")
        f.write(f"//   af_trace.mem        - step-by-step AF trace (RTL debug)\n")
        f.write(f"//\n")
        for fold in range(10):
            s = fold * fold_size
            e = min(s + fold_size, n_total)
            f.write(f"fold_{fold}: test_users={e - s}\n")

    print(f"{'='*60}")
    print(f"  All 10 folds exported to: {output_dir}/")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*60}\n")


# =====================================================================
# Standalone entry point
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export trained CDS model to FPGA-loadable .mem files"
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Path to arrhythmia.data (default: auto-detect)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: FixedPointAnalysis/fpga_mem)")
    parser.add_argument("--mode", choices=["full", "cv"], default="full",
                        help="'full' = train on all data, export one model; "
                             "'cv' = 10-fold cross validation, export 10 models + test vectors")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for fold splits (default: 42)")
    parser.add_argument("--max-users", type=int, default=None,
                        help="Limit number of users (for faster debug runs)")
    args = parser.parse_args()

    data_path = args.data or str(
        Path(__file__).parent.parent / "CDS_NI_Algorithms" / "data" / "arrhythmia.data"
    )
    data, labels = load_dataset(data_path)

    output_dir = args.output or str(
        Path(__file__).parent.parent / "FixedPointAnalysis" / "fpga_mem"
    )

    print(f"Dataset: {data.shape[0]} users, {data.shape[1]} features")
    print(f"Output:  {output_dir}")
    print(f"Mode:    {args.mode}")

    if args.mode == "cv":
        export_all_folds(data, labels, output_dir,
                         rng_seed=args.seed, max_users=args.max_users)
    else:
        print(f"\nTraining Algorithms 1-3 on full dataset...")

        tree = build_decision_tree(data, labels)

        root_id = tree.root.node_id
        nodes_filter = [root_id]
        level2_by_feat: Dict[int, List] = defaultdict(list)
        for n in tree.nodes_by_level.get(2, []):
            if not n.is_leaf:
                level2_by_feat[n.branching_feat_k].append(n)
        for feat_k, children in level2_by_feat.items():
            if len(children) >= 2:
                nodes_filter.extend(c.node_id for c in children)

        print(f"  Tree: {tree.count_nodes()} nodes, {len(nodes_filter)} active")

        alg2 = run_algorithm2(tree, data, labels, DEFAULT_N_BINS, nodes_filter)
        print(f"  Alg2: {alg2.n_perceptor_entries} perceptor, {alg2.n_executive_entries} executive")

        alg3 = run_algorithm3(alg2, tree, data, labels, nodes_filter, reset_per_h=False)
        print(f"  Alg3: {len(alg3.refined_actions)} retained, {len(alg3.removed_actions)} removed")

        export_model_parameters(tree, alg2, alg3, nodes_filter, output_dir)
