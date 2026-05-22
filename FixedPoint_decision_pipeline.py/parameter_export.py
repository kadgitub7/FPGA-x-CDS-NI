"""
FPGA Parameter Export — Convert trained CDS model to Verilog-loadable .mem files

Exports all pretrained data that Algorithm 4 needs at inference time:
  1. Tree topology     — node structure, branch bounds for routing
  2. Healthy ranges    — (node, feature) -> b_min, b_max for Eq. 5 checks
  3. Action library    — (node, disease) -> sorted list of (feature, r_j_h) for RL
  4. Probability tables — (node, disease) -> P(h,f), and per-node P(h>1,f)

Output format: Verilog $readmemh compatible (.mem files with hex values).

All float values are converted to fixed-point integers before export:
  - Sensor/branch values: Q s11.4 (16-bit signed)
  - Probabilities/weights: Q s1.15 (16-bit signed)
  - Threshold:             Q s2.30 (32-bit signed)
"""

from __future__ import annotations

import os
import sys
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

# --- Import fixed-point functions from the golden model ---
from decision_pipeline_fixedPoint import to_fixed, fixed_divide


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
    """Export tree structure to .mem file.

    For each node (ordered by index), writes one line with fields packed as:
        Word 0: [level(4) | is_leaf(1) | branch_feature(9) | padding(2)] = 16 bits
        Word 1: branch_low  in Q s11.4 (16 bits)
        Word 2: branch_high in Q s11.4 (16 bits)
        Word 3: n_users     (16 bits unsigned)
        Word 4: n_diseased  (16 bits unsigned)

    Total: 5 words × 16 bits = 80 bits per node.
    """
    n_nodes = len(node_index)

    # Build reverse map: index -> node_id
    idx_to_nid = {idx: nid for nid, idx in node_index.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"// Tree Topology — {n_nodes} nodes\n")
        f.write(f"// Format per node: 5 x 16-bit words\n")
        f.write(f"//   Word 0: [level(4) | is_leaf(1) | branch_feat(9) | pad(2)]\n")
        f.write(f"//   Word 1: branch_low  (Q s11.4)\n")
        f.write(f"//   Word 2: branch_high (Q s11.4)\n")
        f.write(f"//   Word 3: n_users     (unsigned)\n")
        f.write(f"//   Word 4: n_diseased  (unsigned)\n")
        f.write(f"//\n")

        for idx in range(n_nodes):
            nid = idx_to_nid[idx]
            node = tree.all_nodes[nid]

            level = node.focus_level
            is_leaf = 1 if node.is_leaf else 0
            branch_feat = node.branching_feat_k if node.branch_def is not None else 0

            # Pack word 0: level(4 bits) | is_leaf(1 bit) | branch_feat(9 bits) | pad(2 bits)
            word0 = ((level & 0xF) << 12) | ((is_leaf & 0x1) << 11) | ((branch_feat & 0x1FF) << 2)

            # Branch bounds — convert to Q s11.4
            if node.branch_def is not None:
                branch_low = to_fixed(node.branch_def.low, 11, 4)
                branch_high = to_fixed(node.branch_def.high, 11, 4)
            else:
                # Root has no branch bounds
                branch_low = to_fixed(-1024.0, 11, 4)   # minimum representable
                branch_high = to_fixed(1023.0, 11, 4)    # maximum representable

            n_users = min(node.n_users, 0xFFFF)       # clamp to 16 bits
            n_diseased = min(node.n_diseased, 0xFFFF)

            f.write(f"// Node {idx}: {nid} (level={level}, users={n_users})\n")
            f.write(f"{to_hex_16(word0)}\n")
            f.write(f"{to_hex_16(branch_low)}\n")
            f.write(f"{to_hex_16(branch_high)}\n")
            f.write(f"{to_hex_16(n_users)}\n")
            f.write(f"{to_hex_16(n_diseased)}\n")

    print(f"  Tree topology: {n_nodes} nodes -> {output_path}")


# =====================================================================
# Export 2: Healthy Ranges
# =====================================================================

def export_healthy_ranges(
    alg2_output: Algorithm2Output,
    node_index: Dict[str, int],
    output_path: str,
) -> None:
    """Export healthy ranges to .mem file.

    Uses sparse storage: only (node, feature) pairs that have a trained
    perceptor model get an entry.  Each entry is:
        Word 0: [node_index(8) | feature_idx(9)] packed in 17 bits -> 32-bit word
        Word 1: b_min in Q s9.4 (16 bits)
        Word 2: b_max in Q s9.4 (16 bits)

    First line of the file is the total entry count (so the FPGA knows
    how many to read).
    """
    entries = []

    for (nid, feat_idx), model in alg2_output.perceptor_index.items():
        if nid not in node_index:
            continue
        n_idx = node_index[nid]
        b_min = to_fixed(model.healthy_range.b_min_healthy, 9, 4)
        b_max = to_fixed(model.healthy_range.b_max_healthy, 9, 4)
        entries.append((n_idx, feat_idx, b_min, b_max))

    # Sort by node index, then feature index for predictable ordering
    entries.sort(key=lambda e: (e[0], e[1]))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"// Healthy Ranges — {len(entries)} entries (sparse)\n")
        f.write(f"// Format per entry: 3 x 16-bit words\n")
        f.write(f"//   Word 0: [node_index(8) | feature_idx(8)] (lookup key)\n")
        f.write(f"//   Word 1: b_min (Q s9.4)\n")
        f.write(f"//   Word 2: b_max (Q s9.4)\n")
        f.write(f"//\n")
        f.write(f"// Entry count:\n")
        f.write(f"{to_hex_16(len(entries))}\n")

        for n_idx, feat_idx, b_min, b_max in entries:
            # Pack node_index (high byte) and feature_idx (low byte) into 16 bits
            key_word = ((n_idx & 0xFF) << 8) | (feat_idx & 0xFF)
            f.write(f"// node={n_idx}, feat={feat_idx}\n")
            f.write(f"{to_hex_16(key_word)}\n")
            f.write(f"{to_hex_16(b_min)}\n")
            f.write(f"{to_hex_16(b_max)}\n")

    print(f"  Healthy ranges: {len(entries)} entries -> {output_path}")


# =====================================================================
# Export 3: Action Library
# =====================================================================

def export_action_library(
    alg3_output: Algorithm3Output,
    node_index: Dict[str, int],
    output_path: str,
) -> None:
    """Export refined action library to .mem file.

    Structure:
      HEADER SECTION — one entry per (node, disease) pair:
        Word 0: [node_index(8) | disease_offset(4) | action_count(4)] = 16 bits
        Word 1: start_address in the DATA section (16 bits)

      DATA SECTION — flat list of (feature_idx, r_j_h) pairs:
        Word 0: feature_idx (16 bits, 9 significant)
        Word 1: r_j_h in Q s1.15 (16 bits)

    The FPGA reads the header to find where a (node, disease) pair's
    actions start and how many there are, then reads sequentially from
    the data section.
    """
    # Collect all (node, disease) -> sorted action list
    action_groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    max_actions_seen = 0

    for nid, n_idx in node_index.items():
        for disease_h in DISEASE_CLASSES:
            actions = alg3_output.retained_for_node_disease(nid, disease_h)
            if not actions:
                continue

            fixed_actions = []
            for a in actions:
                if a.action_weight <= 0.0:
                    continue
                feat = a.feature_idx
                weight = to_fixed(a.action_weight, 1, 15)
                fixed_actions.append((feat, weight))

            if fixed_actions:
                d_off = DISEASE_TO_OFFSET[disease_h]
                action_groups[(n_idx, d_off)] = fixed_actions
                max_actions_seen = max(max_actions_seen, len(fixed_actions))

    # Build header and data sections
    headers = []
    data_words = []
    data_addr = 0  # current write position in data section

    # Sort by (node_index, disease_offset) for predictable layout
    for (n_idx, d_off) in sorted(action_groups.keys()):
        actions = action_groups[(n_idx, d_off)]
        count = len(actions)

        headers.append((n_idx, d_off, count, data_addr))

        for feat, weight in actions:
            data_words.append((feat, weight, n_idx, d_off))  # last two for comments
            data_addr += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"// Action Library — {len(headers)} groups, {len(data_words)} total actions\n")
        f.write(f"// Max actions per (node, disease): {max_actions_seen}\n")
        f.write(f"//\n")
        f.write(f"// === HEADER SECTION ===\n")
        f.write(f"// Format: 2 x 16-bit words per group\n")
        f.write(f"//   Word 0: [node_index(8) | disease_offset(4) | count(4)]\n")
        f.write(f"//   Word 1: start_address in data section\n")
        f.write(f"//\n")

        # Write header count first
        f.write(f"// Number of header entries:\n")
        f.write(f"{to_hex_16(len(headers))}\n")

        for n_idx, d_off, count, start_addr in headers:
            # Pack: node(8) | disease(4) | count(4)
            header_word = ((n_idx & 0xFF) << 8) | ((d_off & 0xF) << 4) | (count & 0xF)
            f.write(f"// node={n_idx}, disease_off={d_off} (class={DISEASE_CLASSES[d_off]}), "
                    f"count={count}, data@{start_addr}\n")
            f.write(f"{to_hex_16(header_word)}\n")
            f.write(f"{to_hex_16(start_addr)}\n")

        f.write(f"//\n")
        f.write(f"// === DATA SECTION ===\n")
        f.write(f"// Format: 2 x 16-bit words per action\n")
        f.write(f"//   Word 0: feature_idx\n")
        f.write(f"//   Word 1: r_j_h (Q s1.15)\n")
        f.write(f"//\n")

        for feat, weight, n_idx, d_off in data_words:
            f.write(f"// node={n_idx}, disease={DISEASE_CLASSES[d_off]}, feat={feat}\n")
            f.write(f"{to_hex_16(feat)}\n")
            f.write(f"{to_hex_16(weight)}\n")

    print(f"  Action library: {len(headers)} groups, {len(data_words)} actions -> {output_path}")
    print(f"    Max actions per (node,disease): {max_actions_seen}")


# =====================================================================
# Export 4: Probability Tables
# =====================================================================

def export_probability_tables(
    tree: DecisionTree,
    node_index: Dict[str, int],
    output_path: str,
) -> None:
    """Export precomputed probability tables to .mem file.

    Two sub-tables:

    TABLE A — P(h, f) for each (node, disease) pair:
        Indexed by: node_index * N_DISEASES + disease_offset
        Value: P(h, f) = health_dist[h] / n_users, stored as Q s1.15

    TABLE B — P(h>1, f) for each node:
        Indexed by: node_index
        Value: P(h>1, f) = n_diseased / n_users, stored as Q s1.15

    Precomputing avoids division on the FPGA entirely.
    """
    n_nodes = len(node_index)
    idx_to_nid = {idx: nid for nid, idx in node_index.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"// Probability Tables — {n_nodes} nodes, {N_DISEASES} disease classes\n")
        f.write(f"//\n")

        # --- Table A: P(h, f) ---
        f.write(f"// === TABLE A: P(h, f) ===\n")
        f.write(f"// {n_nodes * N_DISEASES} entries (node_index * {N_DISEASES} + disease_offset)\n")
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

        # --- Table B: P(h>1, f) ---
        f.write(f"//\n")
        f.write(f"// === TABLE B: P(h>1, f) ===\n")
        f.write(f"// {n_nodes} entries (one per node)\n")
        f.write(f"// Each entry: Q s1.15 (16 bits)\n")
        f.write(f"//\n")

        for idx in range(n_nodes):
            nid = idx_to_nid[idx]
            node = tree.all_nodes[nid]
            n_users = node.n_users
            n_diseased = node.n_diseased

            if n_users == 0 or n_diseased == 0:
                p_gt1 = 1  # one LSB, prevents divide-by-zero
            else:
                p_gt1 = fixed_divide(n_diseased, n_users, 15)

            f.write(f"// [{idx:4d}] node={idx} ({nid}): {n_diseased}/{n_users}\n")
            f.write(f"{to_hex_16(p_gt1)}\n")

    total_entries = n_nodes * N_DISEASES + n_nodes
    print(f"  Probability tables: {total_entries} entries -> {output_path}")


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

    # Step 2: Export each table
    export_tree_topology(tree, node_index,
                         os.path.join(output_dir, "tree_topology.mem"))

    export_healthy_ranges(alg2_output, node_index,
                          os.path.join(output_dir, "healthy_ranges.mem"))

    export_action_library(alg3_output, node_index,
                          os.path.join(output_dir, "action_library.mem"))

    export_probability_tables(tree, node_index,
                              os.path.join(output_dir, "prob_tables.mem"))

    export_constants(node_index,
                     os.path.join(output_dir, "cds_params.vh"))

    print(f"{'='*60}")
    print(f"  All exports written to: {output_dir}/")
    print(f"{'='*60}\n")

    return node_index


# =====================================================================
# Standalone entry point
# =====================================================================

if __name__ == "__main__":
    from collections import defaultdict

    # Load dataset
    data_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent.parent / "CDS_NI_Algorithms" / "data" / "arrhythmia.data"
    )
    data, labels = load_dataset(data_path)

    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(
        Path(__file__).parent.parent / "FixedPointAnalysis" / "fpga_mem"
    )

    print(f"Dataset: {data.shape[0]} users, {data.shape[1]} features")
    print(f"Output:  {output_dir}")

    # Train on ALL data (no cross-validation — export a single trained model)
    print(f"\nTraining Algorithms 1-3 on full dataset...")

    # Algorithm 1: Build decision tree
    tree = build_decision_tree(data, labels)

    # Determine active nodes (same logic as ten_fold_cv)
    root_id = tree.root.node_id
    nodes_filter = [root_id]
    level2_by_feat: Dict[int, List[TreeNode]] = defaultdict(list)
    for n in tree.nodes_by_level.get(2, []):
        if not n.is_leaf:
            level2_by_feat[n.branching_feat_k].append(n)
    for feat_k, children in level2_by_feat.items():
        if len(children) >= 2:
            nodes_filter.extend(c.node_id for c in children)

    print(f"  Tree: {tree.count_nodes()} nodes, {len(nodes_filter)} active")

    # Algorithm 2: Train perceptor/executive
    alg2 = run_algorithm2(tree, data, labels, DEFAULT_N_BINS, nodes_filter)
    print(f"  Alg2: {alg2.n_perceptor_entries} perceptor, {alg2.n_executive_entries} executive")

    # Algorithm 3: Refine actions
    alg3 = run_algorithm3(alg2, tree, data, labels, nodes_filter, reset_per_h=False)
    print(f"  Alg3: {len(alg3.refined_actions)} retained, {len(alg3.removed_actions)} removed")

    # Export
    export_model_parameters(tree, alg2, alg3, nodes_filter, output_dir)
