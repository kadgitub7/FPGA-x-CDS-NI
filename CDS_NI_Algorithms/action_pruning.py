"""
Algorithm 3: CDS Executive Actions Refining (FA = 0 Policy)

Greedy set-cover pass over the executive action library from Algorithm 2.
Prunes features that add no new discriminating information about unhealthy
users beyond what higher-weight features already cover.

Key logic (per node):
  - For each disease class h (outer loop, line 1)
  - For each feature o sorted by r_{o|h} descending (middle loop, line 4)
  - For each user u: flag if raw value outside healthy range (inner loop, lines 5-8)
  - s = cumulative flagged users (line 10)
  - If s <= buffer: prune feature (line 11-13)
  - Buffer = s (line 15)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from build_decision_tree import DecisionTree, TreeNode, HEALTHY_CLASS
from action_normalRange import (
    Algorithm2Output, ExecutiveActionEntry, PerceptorModelEntry,
)


@dataclass
class Algorithm3Output:
    refined_actions: List[ExecutiveActionEntry] = field(default_factory=list)
    removed_actions: List[ExecutiveActionEntry] = field(default_factory=list)
    n_nodes_processed: int = 0

    def retained_for_node(self, node_id: str) -> List[ExecutiveActionEntry]:
        return [a for a in self.refined_actions if a.node_id == node_id]

    def retained_for_node_disease(self, node_id: str, disease_h: int) -> List[ExecutiveActionEntry]:
        acts = [a for a in self.refined_actions
                if a.node_id == node_id and a.disease_class == disease_h]
        return sorted(acts, key=lambda e: e.action_weight, reverse=True)


def _refine_one_node(
    node: TreeNode,
    alg2_output: Algorithm2Output,
    data: np.ndarray,
    labels: np.ndarray,
    reset_per_h: bool,
) -> Tuple[List[ExecutiveActionEntry], List[ExecutiveActionEntry]]:
    nid = node.node_id
    n = len(node.user_indices)

    disease_classes = sorted(
        h for h in node.health_dist if h != HEALTHY_CLASS and node.health_dist[h] > 0
    )
    if not disease_classes:
        return [], []

    # Global accumulation state
    newinf = np.zeros(n, dtype=bool)
    buffer = 0
    retained, removed = [], []

    for h in disease_classes:
        if reset_per_h:
            buffer = 0
            newinf = np.zeros(n, dtype=bool)

        # Get and sort actions for this disease class
        h_actions = [
            copy.copy(e) for e in alg2_output.executive_library
            if e.node_id == nid and e.disease_class == h
        ]

        # Remove zero-weight, sort descending
        h_sorted = sorted(
            [a for a in h_actions if a.action_weight > 0.0],
            key=lambda e: e.action_weight, reverse=True,
        )
        for a in h_actions:
            if a.action_weight == 0.0:
                removed.append(a)

        if not h_sorted:
            continue

        for action in h_sorted:
            o = action.feature_idx

            model = alg2_output.get_model(nid, o)
            if model is None:
                removed.append(action)
                continue

            b_min = model.healthy_range.b_min_healthy
            b_max = model.healthy_range.b_max_healthy

            # Flag users outside healthy range
            raw_vals = data[node.user_indices, o]
            valid_mask = ~np.isnan(raw_vals)
            outside_mask = (raw_vals > b_max) | (raw_vals < b_min)
            newinf |= (valid_mask & outside_mask)

            s = int(newinf.sum())

            if s <= buffer:
                action.action_weight = 0.0
                removed.append(action)
            else:
                retained.append(action)

            buffer = s

    return retained, removed


def run_algorithm3(
    alg2_output: Algorithm2Output,
    tree: DecisionTree,
    data: np.ndarray,
    labels: np.ndarray,
    nodes_filter: Optional[List[str]] = None,
    reset_per_h: bool = False,
) -> Algorithm3Output:
    nodes_with_actions = {e.node_id for e in alg2_output.executive_library}

    if nodes_filter is not None:
        nodes_to_process = [nid for nid in nodes_filter if nid in nodes_with_actions]
    else:
        nodes_to_process = sorted(nodes_with_actions)

    output = Algorithm3Output()

    for nid in nodes_to_process:
        if nid not in tree.all_nodes:
            continue
        node = tree.all_nodes[nid]

        retained, removed = _refine_one_node(node, alg2_output, data, labels, reset_per_h)
        output.refined_actions.extend(retained)
        output.removed_actions.extend(removed)
        output.n_nodes_processed += 1

    return output
