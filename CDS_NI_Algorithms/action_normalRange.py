"""
Algorithm 2: CDS Perceptor and Executive Training

For each tree node and feature, computes:
  - Discretization into bins (lines 4-7)
  - Bayesian probability tables P(B_hat|h), P(h,f), P(B_hat), P(h|B_hat) (lines 8-10)
  - Healthy range [b_min, b_max] via Eq. 5 (lines 11-14)
  - Executive action weights r_{o|h} for each disease class (lines 15-23)

Key equations:
  Eq. 4: P(h|B_hat) = P(B_hat|h) * P(h,f) / P(B_hat)
  Eq. 5: All healthy users fall within [b_min, b_max]
  Line 19: r_{o|h} = P(B_hat < b_min|h) + P(B_hat > b_max|h)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_decision_tree import (
    DecisionTree, TreeNode, FeatureKind,
    FEATURE_NAMES, HEALTHY_CLASS, N_FEATURES,
    load_dataset, build_decision_tree, classify_features,
)


# --- Constants ---

DEFAULT_N_BINS: int = 10
LAPLACE_EPSILON: float = 0.0
ALL_DISEASE_CLASSES: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16)


# --- Data Structures ---

@dataclass
class DiscretizationResult:
    feature_idx: int
    feature_name: str
    node_id: str
    b_raw_min: float
    b_raw_max: float
    delta_b: float
    bin_edges: np.ndarray
    n_bins: int
    bin_assignments: np.ndarray
    valid_mask: np.ndarray
    valid_user_rows: np.ndarray
    bin_counts_all: np.ndarray
    is_binary: bool
    is_degenerate: bool
    _raw_values_valid: Optional[np.ndarray] = None

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())


@dataclass
class BayesianTables:
    class_labels: List[int]
    p_bin_given_h: np.ndarray
    p_h_and_f: np.ndarray
    p_bin: np.ndarray
    p_h_given_bin: np.ndarray
    n_users_per_class: Dict[int, int]

    @property
    def n_bins(self) -> int:
        return self.p_bin.shape[0]

    @property
    def n_classes(self) -> int:
        return len(self.class_labels)

    def class_col(self, h: int) -> int:
        try:
            return self.class_labels.index(h)
        except ValueError:
            return -1

    def p_bin_given_class(self, h: int) -> np.ndarray:
        col = self.class_col(h)
        if col < 0:
            return np.zeros(self.n_bins)
        return self.p_bin_given_h[:, col]

    def prevalence(self, h: int) -> float:
        col = self.class_col(h)
        return float(self.p_h_and_f[col]) if col >= 0 else 0.0


@dataclass
class HealthyRangeResult:
    b_min_healthy: float
    b_max_healthy: float
    n_kf: float
    n_healthy_valid: int
    fallback_used: bool


@dataclass
class PerceptorModelEntry:
    node_id: str
    focus_level: int
    branching_feat_k: int
    branch_f: int
    feature_idx: int
    feature_name: str
    n_users_node: int
    disc: DiscretizationResult
    bayes: BayesianTables
    healthy_range: HealthyRangeResult


@dataclass
class ExecutiveActionEntry:
    node_id: str
    focus_level: int
    branching_feat_k: int
    branch_f: int
    feature_idx: int
    feature_name: str
    disease_class: int
    action_weight: float
    p_below_normal: float
    p_above_normal: float
    p_h_and_f: float
    action_label: str


@dataclass
class Algorithm2Output:
    perceptor_library: List[PerceptorModelEntry] = field(default_factory=list)
    executive_library: List[ExecutiveActionEntry] = field(default_factory=list)
    perceptor_index: Dict[Tuple[str, int], PerceptorModelEntry] = field(default_factory=dict)
    executive_index: Dict[Tuple[str, int, int], ExecutiveActionEntry] = field(default_factory=dict)
    n_nodes_processed: int = 0
    n_perceptor_entries: int = 0
    n_executive_entries: int = 0

    def get_model(self, node_id: str, feature_idx: int) -> Optional[PerceptorModelEntry]:
        return self.perceptor_index.get((node_id, feature_idx))

    def get_action(self, node_id: str, feature_idx: int, disease_h: int) -> Optional[ExecutiveActionEntry]:
        return self.executive_index.get((node_id, feature_idx, disease_h))

    def actions_for_node(self, node_id: str) -> List[ExecutiveActionEntry]:
        return [e for e in self.executive_library if e.node_id == node_id]

    def top_actions(self, node_id: str, disease_h: int, top_k: int = 10) -> List[ExecutiveActionEntry]:
        acts = [e for e in self.executive_library
                if e.node_id == node_id and e.disease_class == disease_h]
        return sorted(acts, key=lambda e: e.action_weight, reverse=True)[:top_k]


# --- Discretization (Lines 4-7) ---

def _n_bins_for_node(n_valid: int, is_binary: bool) -> int:
    if is_binary:
        return 2
    if n_valid < 2:
        return 1
    return int(np.ceil(1 + np.log2(n_valid))) + 18


def compute_discretization(
    feature_idx: int,
    node: TreeNode,
    data: np.ndarray,
    feature_kinds: Dict[int, FeatureKind],
) -> Optional[DiscretizationResult]:
    fname = FEATURE_NAMES.get(feature_idx, f"feat_{feature_idx}")
    raw_values = data[node.user_indices, feature_idx]
    valid_mask = ~np.isnan(raw_values)
    valid_rows = np.where(valid_mask)[0]
    valid_vals = raw_values[valid_mask]
    n_valid = len(valid_vals)

    if n_valid == 0:
        return None

    b_raw_min = float(valid_vals.min())
    b_raw_max = float(valid_vals.max())
    is_binary = (feature_kinds[feature_idx] == FeatureKind.BINARY)

    # Degenerate case: all values identical
    if b_raw_min == b_raw_max:
        return DiscretizationResult(
            feature_idx=feature_idx, feature_name=fname, node_id=node.node_id,
            b_raw_min=b_raw_min, b_raw_max=b_raw_max, delta_b=1.0,
            bin_edges=np.array([b_raw_min - 0.5, b_raw_min + 0.5]),
            n_bins=1, bin_assignments=np.zeros(n_valid, dtype=int),
            valid_mask=valid_mask, valid_user_rows=valid_rows,
            bin_counts_all=np.array([n_valid]), is_binary=is_binary, is_degenerate=True,
        )

    n_bins_req = _n_bins_for_node(n_valid, is_binary)

    if is_binary:
        bin_edges = np.array([-0.5, 0.5, 1.5]) if n_bins_req > 1 else np.array([-0.5, 1.5])
        n_bins = 2 if n_bins_req > 1 else 1
        delta_b = 1.0 if n_bins == 2 else 2.0
    else:
        bin_edges = np.linspace(b_raw_min, b_raw_max, n_bins_req + 1)
        n_bins = n_bins_req
        delta_b = (b_raw_max - b_raw_min) / n_bins

    bin_asgn = np.clip(np.searchsorted(bin_edges[1:], valid_vals, side='right'), 0, n_bins - 1)
    bin_counts = np.bincount(bin_asgn, minlength=n_bins)

    return DiscretizationResult(
        feature_idx=feature_idx, feature_name=fname, node_id=node.node_id,
        b_raw_min=b_raw_min, b_raw_max=b_raw_max, delta_b=delta_b,
        bin_edges=bin_edges, n_bins=n_bins, bin_assignments=bin_asgn,
        valid_mask=valid_mask, valid_user_rows=valid_rows,
        bin_counts_all=bin_counts, is_binary=is_binary, is_degenerate=False,
    )


# --- Bayesian Tables (Lines 8-10) ---

def compute_bayesian_tables(
    disc: DiscretizationResult,
    node: TreeNode,
    labels: np.ndarray,
    N_total: int,
    class_labels: Optional[List[int]] = None,
) -> BayesianTables:
    if class_labels is None:
        class_labels = sorted(node.health_dist.keys())
    n_classes = len(class_labels)
    n_bins = disc.n_bins

    global_indices_valid = node.user_indices[disc.valid_user_rows]
    labels_valid = labels[global_indices_valid]

    counts_per_class_bin = np.zeros((n_classes, n_bins), dtype=float)
    users_per_class = np.zeros(n_classes, dtype=int)

    for ci, cls in enumerate(class_labels):
        class_mask = (labels_valid == cls)
        users_per_class[ci] = int(class_mask.sum())
        if users_per_class[ci] > 0:
            counts_per_class_bin[ci] = np.bincount(disc.bin_assignments[class_mask], minlength=n_bins)

    # P(B_hat|h)
    p_bin_given_h = np.zeros((n_bins, n_classes), dtype=float)
    for ci in range(n_classes):
        n_cls = users_per_class[ci]
        if n_cls == 0:
            p_bin_given_h[:, ci] = 1.0 / n_bins
        else:
            raw = counts_per_class_bin[ci] + LAPLACE_EPSILON
            p_bin_given_h[:, ci] = raw / raw.sum()

    # P(h, f)
    p_h_and_f = np.zeros(n_classes, dtype=float)
    for ci, cls in enumerate(class_labels):
        p_h_and_f[ci] = node.health_dist.get(cls, 0) / N_total

    # P(B_hat) = evidence
    p_bin = p_bin_given_h @ p_h_and_f
    p_bin_sum = p_bin.sum()
    if p_bin_sum > 0:
        p_bin /= p_bin_sum

    # P(h|B_hat) via Bayes rule
    p_h_given_bin = np.zeros((n_bins, n_classes), dtype=float)
    for b in range(n_bins):
        if p_bin[b] < 1e-300:
            p_h_given_bin[b] = 1.0 / n_classes
        else:
            p_h_given_bin[b] = (p_bin_given_h[b] * p_h_and_f) / p_bin[b]

    return BayesianTables(
        class_labels=class_labels,
        p_bin_given_h=p_bin_given_h,
        p_h_and_f=p_h_and_f,
        p_bin=p_bin,
        p_h_given_bin=p_h_given_bin,
        n_users_per_class=dict(zip(class_labels, users_per_class.tolist())),
    )


# --- Healthy Range (Lines 11-14, Eq. 5) ---

def compute_healthy_range(
    disc: DiscretizationResult,
    node: TreeNode,
    labels: np.ndarray,
) -> HealthyRangeResult:
    global_valid = node.user_indices[disc.valid_user_rows]
    labels_valid = labels[global_valid]
    healthy_mask = (labels_valid == HEALTHY_CLASS)
    n_healthy = int(healthy_mask.sum())

    if n_healthy > 0:
        healthy_vals = disc._raw_values_valid[healthy_mask]
        b_min = float(healthy_vals.min())
        b_max = float(healthy_vals.max())
        n_kf = (b_max - b_min) / disc.delta_b if disc.delta_b > 0 else 1.0
        return HealthyRangeResult(b_min, b_max, n_kf, n_healthy, fallback_used=False)

    # Fallback: no healthy users, use full range
    return HealthyRangeResult(
        disc.b_raw_min, disc.b_raw_max, float(disc.n_bins), 0, fallback_used=True,
    )


# --- Executive Actions (Lines 15-23) ---

def compute_executive_actions(
    disc: DiscretizationResult,
    healthy_range: HealthyRangeResult,
    bayes: BayesianTables,
    node: TreeNode,
    labels: np.ndarray,
) -> List[ExecutiveActionEntry]:
    actions = []
    b_min = healthy_range.b_min_healthy
    b_max = healthy_range.b_max_healthy
    if b_min > b_max:
        return actions

    global_valid = node.user_indices[disc.valid_user_rows]
    labels_valid = labels[global_valid]
    healthy_mask = (labels_valid == HEALTHY_CLASS)
    if healthy_mask.sum() == 0:
        return actions

    healthy_bins = disc.bin_assignments[healthy_mask]
    min_healthy_bin = int(healthy_bins.min())
    max_healthy_bin = int(healthy_bins.max())

    for cls in bayes.class_labels:
        if cls == HEALTHY_CLASS:
            continue
        p_h_f = bayes.prevalence(cls)
        if p_h_f <= 0:
            continue

        p_bin_h = bayes.p_bin_given_class(cls)
        p_below = float(p_bin_h[:min_healthy_bin].sum()) if min_healthy_bin > 0 else 0.0
        p_above = float(p_bin_h[max_healthy_bin + 1:].sum()) if max_healthy_bin < disc.n_bins - 1 else 0.0
        r_o_h = p_below + p_above

        if (p_below > 0 or p_above > 0) and p_h_f > 0:
            actions.append(ExecutiveActionEntry(
                node_id=node.node_id,
                focus_level=node.focus_level,
                branching_feat_k=node.branching_feat_k,
                branch_f=node.branch_f,
                feature_idx=disc.feature_idx,
                feature_name=disc.feature_name,
                disease_class=cls,
                action_weight=r_o_h,
                p_below_normal=p_below,
                p_above_normal=p_above,
                p_h_and_f=p_h_f,
                action_label=f"sensor_for_feat_{disc.feature_idx}({disc.feature_name})",
            ))
    return actions


# --- Per-Node Execution ---

def run_algorithm2_for_node(
    node: TreeNode,
    data: np.ndarray,
    labels: np.ndarray,
    feature_kinds: Dict[int, FeatureKind],
    class_labels: Optional[List[int]] = None,
) -> Tuple[List[PerceptorModelEntry], List[ExecutiveActionEntry]]:
    if class_labels is None:
        class_labels = sorted(node.health_dist.keys())

    perceptor_entries: List[PerceptorModelEntry] = []
    executive_entries: List[ExecutiveActionEntry] = []

    for feature_o in node.feature_indices:
        disc = compute_discretization(feature_o, node, data, feature_kinds)
        if disc is None:
            continue

        raw_values_valid = data[node.user_indices, feature_o][disc.valid_mask]
        disc._raw_values_valid = raw_values_valid

        bayes = compute_bayesian_tables(disc, node, labels, len(labels), list(class_labels))
        healthy_range = compute_healthy_range(disc, node, labels)

        perceptor_entries.append(PerceptorModelEntry(
            node_id=node.node_id, focus_level=node.focus_level,
            branching_feat_k=node.branching_feat_k, branch_f=node.branch_f,
            feature_idx=feature_o, feature_name=disc.feature_name,
            n_users_node=node.n_users, disc=disc, bayes=bayes,
            healthy_range=healthy_range,
        ))

        executive_entries.extend(
            compute_executive_actions(disc, healthy_range, bayes, node, labels)
        )

    return perceptor_entries, executive_entries


# --- Main Entry Point ---

def run_algorithm2(
    tree: DecisionTree,
    data: np.ndarray,
    labels: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    nodes_filter: Optional[List[str]] = None,
    class_labels: Optional[List[int]] = None,
) -> Algorithm2Output:
    feature_kinds = tree.feature_kinds
    if class_labels is None:
        class_labels = sorted(tree.root.health_dist.keys())

    output = Algorithm2Output()

    for m in sorted(tree.nodes_by_level.keys()):
        for node in tree.nodes_by_level[m]:
            if nodes_filter and node.node_id not in nodes_filter:
                continue

            perc, exec_ = run_algorithm2_for_node(node, data, labels, feature_kinds, list(class_labels))

            for e in perc:
                output.perceptor_library.append(e)
                output.perceptor_index[(e.node_id, e.feature_idx)] = e
            for e in exec_:
                output.executive_library.append(e)
                output.executive_index[(e.node_id, e.feature_idx, e.disease_class)] = e

            output.n_nodes_processed += 1

    output.n_perceptor_entries = len(output.perceptor_library)
    output.n_executive_entries = len(output.executive_library)
    return output
