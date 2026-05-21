"""
Algorithm 1: CDS Decision Tree Construction

Builds a multi-level decision tree that partitions users into sub-groups
based on Eq. 2 branching condition and feature-ordering exclusion (line 9).
Sex feature forces a top-level split into male/female subtrees.

Key equations:
  Eq. 2: ceil(|U_{m-1}| * P_f) >= u_min
  Eq. 3: u_min = 5 / threshold = 200
  Line 9: O_m = O_{m-1} - {1,...,k}  (m > 2 only)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DIAGNOSTIC_THRESHOLD: float = 0.025
U_MIN: int = math.ceil(5 / DIAGNOSTIC_THRESHOLD)
HEALTHY_CLASS: int = 1
N_FEATURES: int = 279
SEX_FEATURE_IDX: int = 1
MALE_VALUE: float = 0.0
FEMALE_VALUE: float = 1.0


class FeatureKind(Enum):
    BINARY = auto()
    CONTINUOUS = auto()

# This is a function that is built to map the dataset
def _build_feature_names() -> Dict[int, str]:
    names: Dict[int, str] = {
        0: "Age", 1: "Sex", 2: "Height", 3: "Weight",
        4: "QRS_dur", 5: "PR_int", 6: "QT_int", 7: "T_int",
        8: "P_int", 9: "QRS_angle", 10: "T_angle", 11: "P_angle",
        12: "QRST_angle", 13: "J_angle", 14: "Heart_rate",
    }
    channels = ["DI","DII","DIII","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6"]
    wave_labels = [
        "Q_wid","R_wid","S_wid","Rp_wid","Sp_wid","N_defl",
        "Rag_R","Diph_R","Rag_P","Diph_P","Rag_T","Diph_T",
    ]
    amp_labels = [
        "JJ_amp","Q_amp","R_amp","S_amp","Rp_amp","Sp_amp",
        "P_amp","T_amp","QRSA","QRSTA",
    ]
    # Here we are programatically generating the feature names which are used in the file for arryhthmia names
    for i, ch in enumerate(channels):
        base_w = 15 + i * 12
        for j, lbl in enumerate(wave_labels):
            names[base_w + j] = f"{ch}_{lbl}"
        base_a = 159 + i * 10
        for j, lbl in enumerate(amp_labels):
            names[base_a + j] = f"{ch}_{lbl}"
    return names

FEATURE_NAMES: Dict[int, str] = _build_feature_names()


# --- Data Structures ---

@dataclass
class BranchDef:
    feature_idx: int
    branch_idx: int
    label: str
    low: float
    high: float
    is_left_closed: bool = True
    is_right_closed: bool = False

    # Here we are creating bounds for branches.
    def contains(self, value: float) -> bool:
        if np.isnan(value):
            return False
        left_ok = value >= self.low if self.is_left_closed else value > self.low
        right_ok = value <= self.high if self.is_right_closed else value < self.high
        return left_ok and right_ok


@dataclass
class TreeNode:
    node_id: str
    focus_level: int
    branching_feat_k: int
    branch_f: int
    branch_def: Optional[BranchDef]
    user_indices: np.ndarray
    feature_indices: List[int]
    branch_prob: float
    health_dist: Dict[int, int]
    is_leaf: bool = False
    prune_reason: str = ""
    all_children: List[TreeNode] = field(default_factory=list)

    @property
    def n_users(self) -> int:
        return len(self.user_indices)

    @property
    def n_diseased(self) -> int:
        return sum(c for h, c in self.health_dist.items() if h != HEALTHY_CLASS)

    def add_child(self, child: TreeNode):
        self.all_children.append(child)


@dataclass
class DecisionTree:
    root: TreeNode
    feature_kinds: Dict[int, FeatureKind]
    threshold: float
    u_min: int
    all_nodes: Dict[str, TreeNode] = field(default_factory=dict)
    nodes_by_level: Dict[int, List[TreeNode]] = field(default_factory=dict)
    valid_branches: Dict[int, int] = field(default_factory=dict)
    pruned_branches: Dict[int, int] = field(default_factory=dict)

    def register(self, node: TreeNode):
        self.all_nodes[node.node_id] = node
        self.nodes_by_level.setdefault(node.focus_level, []).append(node)

    def depth(self) -> int:
        return max(self.nodes_by_level.keys()) if self.nodes_by_level else 0

    def count_nodes(self) -> int:
        return len(self.all_nodes)


@dataclass
class ForcedSexForest:
    male_tree: DecisionTree
    female_tree: DecisionTree
    male_indices: np.ndarray
    female_indices: np.ndarray
    n_users: int
    threshold: float


# --- Core Functions ---

def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None, na_values="?")
    if df.shape[1] != 280:
        raise ValueError(f"Expected 280 columns, got {df.shape[1]}")
    data = df.iloc[:, :279].to_numpy(dtype=float)
    labels = df.iloc[:, 279].to_numpy(dtype=int)
    return data, labels


def classify_features(data: np.ndarray) -> Dict[int, FeatureKind]:
    kinds: Dict[int, FeatureKind] = {}
    for col in range(N_FEATURES):
        valid = data[:, col][~np.isnan(data[:, col])]
        if len(valid) == 0 or not set(valid).issubset({0.0, 1.0}):
            kinds[col] = FeatureKind.CONTINUOUS
        else:
            kinds[col] = FeatureKind.BINARY
    return kinds


def health_distribution(user_indices: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
    dist: Dict[int, int] = {}
    for lbl in labels[user_indices]:
        dist[int(lbl)] = dist.get(int(lbl), 0) + 1
    return dist

# Here we determine the branches for a binary or continuous feature
def compute_fsd_branches(
    feature_idx: int,
    user_indices: np.ndarray,
    data: np.ndarray,
    kind: FeatureKind,
) -> List[BranchDef]:
    vals = data[user_indices, feature_idx]
    valid = vals[~np.isnan(vals)]
    if len(valid) == 0:
        return []

    if kind == FeatureKind.BINARY:
        branches = []
        for bval, label in [(0.0, "0"), (1.0, "1")]:
            if (valid == bval).any():
                branches.append(BranchDef(
                    feature_idx=feature_idx, branch_idx=int(bval),
                    label=label, low=bval, high=bval,
                    is_left_closed=True, is_right_closed=True,
                ))
        return branches

    # Continuous: split at median
    vmin, vmax = float(valid.min()), float(valid.max())
    if vmin == vmax:
        return [BranchDef(
            feature_idx=feature_idx, branch_idx=0, label=f"={vmin:.3g}",
            low=vmin, high=vmax, is_left_closed=True, is_right_closed=True,
        )]
    median = float(np.median(valid))
    return [
        BranchDef(feature_idx=feature_idx, branch_idx=1,
                  label=f"<={median:.3g}", low=vmin, high=median,
                  is_left_closed=True, is_right_closed=True),
        BranchDef(feature_idx=feature_idx, branch_idx=2,
                  label=f">{median:.3g}", low=median, high=vmax,
                  is_left_closed=False, is_right_closed=True),
    ]


def filter_users_by_branch(
    user_indices: np.ndarray, bdef: BranchDef, data: np.ndarray
) -> np.ndarray:
    vals = data[user_indices, bdef.feature_idx]
    mask = np.array([bdef.contains(v) for v in vals])
    return user_indices[mask]


def check_branching_condition(
    branch_user_count: int, parent_user_count: int, threshold: float
) -> Tuple[bool, int, int]:
    u_min_val = math.ceil(5 / threshold)
    lhs = branch_user_count
    return lhs >= u_min_val, lhs, u_min_val


def compute_branch_probability(branch_users: int, parent_users: int) -> float:
    return branch_users / parent_users if parent_users > 0 else 0.0


# --- Tree Building ---

def _compute_child_features(
    parent_features: List[int], branching_k: int, child_level: int
) -> List[int]:
    # Line 9: at m > 2, remove features with index <= k
    # This is to remove redundant branches
    if child_level <= 2:
        return list(parent_features)
    return [f for f in parent_features if f > branching_k]

# We split on each feature and go back to check if it meets threshold and prune if it does not.
def _try_split(parent, feature_k, kind, data, labels, threshold):
    m_child = parent.focus_level + 1
    branch_defs = compute_fsd_branches(feature_k, parent.user_indices, data, kind)
    if len(branch_defs) < 2:
        return [], 0

    created, n_pruned = [], 0
    for bdef in branch_defs:
        busers = filter_users_by_branch(parent.user_indices, bdef, data)
        passes, _, _ = check_branching_condition(len(busers), parent.n_users, threshold)
        if not passes:
            n_pruned += 1
            continue
        feats = _compute_child_features(parent.feature_indices, feature_k, m_child)
        child = TreeNode(
            node_id=f"{parent.node_id}|k{feature_k}_f{bdef.branch_idx}",
            focus_level=m_child,
            branching_feat_k=feature_k,
            branch_f=bdef.branch_idx,
            branch_def=bdef,
            user_indices=busers,
            feature_indices=feats,
            branch_prob=compute_branch_probability(len(busers), parent.n_users),
            health_dist=health_distribution(busers, labels),
        )
        created.append(child)

    if len(created) == 1:
        created[0].is_leaf = True
        created[0].prune_reason = "Only one branch passed Eq.2"

    return created, n_pruned

# This is the function used to expand the tree as far as possible to then prune afterwards.
def _expand_node(parent, data, labels, kinds, threshold, tree):
    m_child = parent.focus_level + 1
    n_created = 0

    for k in sorted(parent.feature_indices):
        kind = kinds.get(k, FeatureKind.CONTINUOUS)
        children, pruned = _try_split(parent, k, kind, data, labels, threshold)
        if children:
            n_created += len(children)
            for child in children:
                parent.add_child(child)
                tree.register(child)

    if n_created == 0:
        parent.is_leaf = True
        parent.prune_reason = f"No feature passed Eq.2 at m={m_child}"

    return n_created

# This is the function that is used to call other functions and build the decision tree. We use some constants that are specific to this dataset
def build_decision_tree(
    data: np.ndarray,
    labels: np.ndarray,
    threshold: float = DIAGNOSTIC_THRESHOLD,
    max_m: int = 2,
) -> DecisionTree:
    kinds = classify_features(data)
    all_indices = np.arange(data.shape[0])
    feature_indices = list(range(N_FEATURES))

    root = TreeNode(
        node_id="root",
        focus_level=1,
        branching_feat_k=-1,
        branch_f=0,
        branch_def=None,
        user_indices=all_indices,
        feature_indices=feature_indices,
        branch_prob=1.0,
        health_dist=health_distribution(all_indices, labels),
    )

    tree = DecisionTree(
        root=root, feature_kinds=kinds,
        threshold=threshold, u_min=math.ceil(5 / threshold),
    )
    tree.register(root)
    tree.nodes_by_level[1] = [root]

    current_nodes = [root]
    for m in range(2, max_m + 1):
        next_nodes = []
        for parent in current_nodes:
            _expand_node(parent, data, labels, kinds, threshold, tree)
            next_nodes.extend(parent.all_children)
        if not next_nodes:
            break
        current_nodes = next_nodes

    return tree