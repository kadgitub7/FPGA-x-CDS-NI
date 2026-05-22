"""
Algorithm 4: CDS User-Health Prediction Pipeline

Prediction/inference phase. For a single test user, applies cognitive actions
(sensor activations), accumulates Assurance Factor (AF), and decides:
  UNHEALTHY  - feature value outside healthy range -> alarm
  HEALTHY    - rw = (1 - AF) <= threshold
  SCREENING  - all disease classes checked, AF insufficient

Key equations:
  Eq. 7: AF_t = P(h,f) * r_{j|h} / P(h>1,f) + AF_{t-1}
  Eq. 8: rw_t = 1 - AF_t

Includes LOOCV evaluation pipeline for accuracy measurement.
"""

from __future__ import annotations

import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from CDS_NI_Algorithms.build_decision_tree import (
    DecisionTree, TreeNode,
    FEATURE_NAMES, HEALTHY_CLASS, DIAGNOSTIC_THRESHOLD, U_MIN, N_FEATURES,
    load_dataset, build_decision_tree, build_sex_specific_tree,
    SEX_FEATURE_IDX,
)
from CDS_NI_Algorithms.action_normalRange import (
    Algorithm2Output, ExecutiveActionEntry, PerceptorModelEntry,
    run_algorithm2, DEFAULT_N_BINS,
)
from CDS_NI_Algorithms.action_pruning import Algorithm3Output, run_algorithm3


# --- Constants ---

ALL_DISEASE_CLASSES: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16)


# --- Data Structures ---

class HealthDecision(Enum):
    HEALTHY = "Healthy"
    UNHEALTHY = "Unhealthy"
    SCREENING = "Screening"
    UNKNOWN = "Unknown"


@dataclass
class FPGATraceStep:
    """One computation step's worth of intermediate values for FPGA
    fixed-point bit-budget analysis.

    step_type indicates the context:
      "pac"             — committed perception-action cycle
      "rl_sim"          — RL lookahead evaluation of one candidate action
      "branch_route"    — node routing comparison
      "threshold_check" — final rw vs threshold comparison
    """
    # --- Main PAC values ---
    raw_value: float = float("nan")   # BD_m^k(o,u) — sensor reading
    b_min: float = float("nan")       # Eq. 5 — lower healthy boundary
    b_max: float = float("nan")       # Eq. 5 — upper healthy boundary
    r_j_h: float = float("nan")       # Algorithm 3 — action weight
    p_h_f: float = float("nan")       # P(h, f^k_m) — disease prevalence in node
    p_h_gt1_f: float = float("nan")   # P(h>1, f^k_m) — denominator term
    numerator: float = float("nan")   # p_h_f * r_j_h — intermediate product before division
    delta_AF: float = float("nan")    # numerator / p_h_gt1_f — AF increment this step
    AF_real: float = float("nan")     # Eq. 7 cumulative — running assurance factor
    rw_real: float = float("nan")     # Eq. 8: 1 - AF — remaining risk

    # --- RL lookahead values (lines 11-17 of Algorithm 4) ---
    AF_sim: float = float("nan")      # simulated AF increment for candidate
    rw_sim: float = float("nan")      # 1 - (AF_sim + AF_real) — can go negative
    best_rw: float = float("nan")     # running minimum rw_sim across candidates

    # --- Node routing values (BranchDef.contains) ---
    branch_val: float = float("nan")  # sensor value used for tree routing
    branch_low: float = float("nan")  # tree partition lower bound
    branch_high: float = float("nan") # tree partition upper bound

    # --- Threshold comparison ---
    rw_final: float = float("nan")    # final 1 - AF compared to threshold
    threshold: float = float("nan")   # DIAGNOSTIC_THRESHOLD constant

    # --- Metadata (not profiled, just for traceability) ---
    feature_idx: int = -1
    disease_class: int = -1
    node_id: str = ""
    step_type: str = ""


@dataclass
class PredictionRecord:
    user_global_idx: int
    true_label: int
    decision: HealthDecision = HealthDecision.UNKNOWN
    is_correct: bool = False
    alarm_class: Optional[int] = None
    alarm_feature_idx: Optional[int] = None
    max_focus_reached: int = 1
    total_pac_count: int = 0
    total_actions_applied: int = 0
    initial_action_feat: Optional[int] = None
    af_trace: List[FPGATraceStep] = field(default_factory=list)

    @property
    def true_is_healthy(self) -> bool:
        return self.true_label == HEALTHY_CLASS

    @property
    def true_is_diseased(self) -> bool:
        return not self.true_is_healthy


@dataclass
class Algorithm4Output:
    records: List[PredictionRecord] = field(default_factory=list)
    n_healthy_correct: int = 0
    n_healthy_total: int = 0
    n_diseased_correct: int = 0
    n_diseased_total: int = 0
    n_screening: int = 0
    overall_accuracy: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    false_alarm_rate: float = 0.0

    def _recompute_stats(self) -> None:
        self.n_healthy_correct = sum(
            1 for r in self.records
            if r.true_is_healthy and r.decision != HealthDecision.UNHEALTHY
        )
        self.n_healthy_total = sum(1 for r in self.records if r.true_is_healthy)
        self.n_diseased_correct = sum(
            1 for r in self.records
            if r.true_is_diseased and r.decision == HealthDecision.UNHEALTHY
        )
        self.n_diseased_total = sum(1 for r in self.records if r.true_is_diseased)
        self.n_screening = sum(1 for r in self.records if r.decision == HealthDecision.SCREENING)

        n_total = len(self.records)
        n_correct = self.n_healthy_correct + self.n_diseased_correct
        self.overall_accuracy = n_correct / n_total if n_total else 0.0
        self.sensitivity = self.n_diseased_correct / self.n_diseased_total if self.n_diseased_total else 0.0
        self.specificity = self.n_healthy_correct / self.n_healthy_total if self.n_healthy_total else 0.0

        n_fa = sum(1 for r in self.records if r.true_is_healthy and r.decision == HealthDecision.UNHEALTHY)
        self.false_alarm_rate = n_fa / self.n_healthy_total if self.n_healthy_total else 0.0


# --- Helper Functions ---

def _is_outside_healthy_range(value: float, b_min: float, b_max: float) -> bool:
    if np.isnan(value) or np.isnan(b_min) or np.isnan(b_max) or b_min > b_max:
        return False
    return (value < b_min) or (value > b_max)


def _compute_p_h_f(node: TreeNode, disease_h: int) -> float:
    if node.n_users == 0:
        return 0.0
    return node.health_dist.get(disease_h, 0) / node.n_users


def _compute_p_h_gt1_f(node: TreeNode) -> float:
    if node.n_users == 0:
        return 1e-9
    n_diseased = node.n_diseased
    return n_diseased / node.n_users if n_diseased > 0 else 1e-9


def _compute_AF_increment(p_h_f: float, r_j_h: float, p_h_gt1_f: float) -> float:
    if p_h_gt1_f < 1e-12:
        return 0.0
    return max(0.0, (p_h_f * r_j_h) / p_h_gt1_f)


def _update_AF(AF_real: float, delta_AF: float) -> float:
    return min(1.0, max(0.0, AF_real + delta_AF))

def _find_all_applicable_nodes(
    user_idx: int, focus_level: int, tree: DecisionTree,
    data: np.ndarray, valid_node_ids: Optional[set] = None,
    record: Optional[PredictionRecord] = None,
) -> List[TreeNode]:
    if focus_level == 1:
        return [tree.root]
    matches = []
    for node in tree.nodes_by_level.get(focus_level, []):
        if valid_node_ids is not None and node.node_id not in valid_node_ids:
            continue
        if node.branch_def is not None:
            user_val = float(data[user_idx, node.branch_def.feature_idx])
            if record is not None and not np.isnan(user_val):
                record.af_trace.append(FPGATraceStep(
                    branch_val=user_val,
                    branch_low=node.branch_def.low,
                    branch_high=node.branch_def.high,
                    feature_idx=node.branch_def.feature_idx,
                    node_id=node.node_id, step_type="branch_route",
                ))
            if node.branch_def.contains(user_val):
                matches.append(node)
    return matches


def _get_sorted_disease_actions(
    node_id: str, disease_h: int, alg3_output: Algorithm3Output,
    consumed: Optional[Set[Tuple[int, int]]] = None,
) -> List[ExecutiveActionEntry]:
    acts = alg3_output.retained_for_node_disease(node_id, disease_h)
    out = [a for a in acts if a.action_weight > 0.0]
    if consumed:
        out = [a for a in out if (a.feature_idx, disease_h) not in consumed]
    return out


def _rl_select_best_action(
    candidates: List[ExecutiveActionEntry],
    node: TreeNode,
    disease_h: int,
    AF_real: float,
    alg2_output: Algorithm2Output,
    record: Optional[PredictionRecord] = None,
) -> Optional[ExecutiveActionEntry]:
    """RL lookahead: select action minimizing rw_sim (lines 11-17)."""
    if not candidates:
        return None

    p_h_f = _compute_p_h_f(node, disease_h)
    p_h_gt1_f = _compute_p_h_gt1_f(node)
    best_action, best_rw = None, float("inf")

    for action in candidates:
        AF_sim = _compute_AF_increment(p_h_f, action.action_weight, p_h_gt1_f)
        rw_sim = 1.0 - (AF_sim + AF_real)
        if rw_sim < best_rw:
            best_rw = rw_sim
            best_action = action

        if record is not None:
            record.af_trace.append(FPGATraceStep(
                p_h_f=p_h_f, p_h_gt1_f=p_h_gt1_f,
                r_j_h=action.action_weight,
                numerator=p_h_f * action.action_weight,
                AF_sim=AF_sim, rw_sim=rw_sim, best_rw=best_rw,
                AF_real=AF_real,
                feature_idx=action.feature_idx, disease_class=disease_h,
                node_id=node.node_id, step_type="rl_sim",
            ))

    return best_action


# --- Core Prediction (Single Node) ---

def _predict_at_node(
    user_idx: int,
    node: TreeNode,
    disease_classes: List[int],
    data: np.ndarray,
    alg2_output: Algorithm2Output,
    alg3_output: Algorithm3Output,
    AF_real: float,
    pac_counter: List[int],
    record: PredictionRecord,
    consumed: Optional[Set[Tuple[int, int]]] = None,
    init_action_h: Optional[int] = None,
) -> Tuple[HealthDecision, float, Optional[int]]:
    nid = node.node_id
    p_h_gt1_f = _compute_p_h_gt1_f(node)

    for h in disease_classes:
        p_h_f = _compute_p_h_f(node, h)
        sorted_actions = _get_sorted_disease_actions(nid, h, alg3_output, consumed)

        # Exclude init action from its disease class to prevent double-counting
        if init_action_h is not None and h == init_action_h and record.initial_action_feat is not None:
            sorted_actions = [a for a in sorted_actions if a.feature_idx != record.initial_action_feat]

        if not sorted_actions:
            continue

        O_mkf = [a.feature_idx for a in sorted_actions]
        C_buf = list(sorted_actions)

        while C_buf:
            selected = _rl_select_best_action(C_buf, node, h, AF_real, alg2_output, record)
            if selected is None:
                break

            C_buf = [a for a in C_buf if a.feature_idx != selected.feature_idx]
            j = selected.feature_idx
            V_j = float(data[user_idx, j])

            if np.isnan(V_j):
                continue

            pac_counter[0] += 1
            record.total_actions_applied += 1
            if consumed is not None:
                consumed.add((j, h))

            model = alg2_output.get_model(nid, j)
            if model is None:
                continue

            b_min = model.healthy_range.b_min_healthy
            b_max = model.healthy_range.b_max_healthy

            exec_entry = alg2_output.get_action(nid, j, h)
            r_j_h = exec_entry.action_weight if exec_entry else selected.action_weight

            numer = p_h_f * r_j_h
            delta_AF = _compute_AF_increment(p_h_f, r_j_h, p_h_gt1_f)
            AF_real = _update_AF(AF_real, delta_AF)
            rw_real = 1.0 - AF_real

            record.af_trace.append(FPGATraceStep(
                raw_value=V_j, b_min=b_min, b_max=b_max,
                r_j_h=r_j_h, p_h_f=p_h_f, p_h_gt1_f=p_h_gt1_f,
                numerator=numer, delta_AF=delta_AF,
                AF_real=AF_real, rw_real=rw_real,
                feature_idx=j, disease_class=h, node_id=nid,
                step_type="pac",
            ))

            if _is_outside_healthy_range(V_j, b_min, b_max):
                return HealthDecision.UNHEALTHY, AF_real, h

    # Post-disease-loop threshold check
    rw_final = 1.0 - AF_real
    record.af_trace.append(FPGATraceStep(
        rw_final=rw_final, threshold=DIAGNOSTIC_THRESHOLD,
        AF_real=AF_real, node_id=node.node_id,
        step_type="threshold_check",
    ))
    if rw_final <= DIAGNOSTIC_THRESHOLD:
        return HealthDecision.HEALTHY, AF_real, None

    # Check if focus can increase
    next_m = node.focus_level + 1
    can_increase = any(
        c.focus_level == next_m and c.n_users >= U_MIN
        for c in node.all_children
    )
    if can_increase:
        return HealthDecision.UNKNOWN, AF_real, None

    return HealthDecision.SCREENING, AF_real, None


# --- Main Prediction (Algorithm 4) ---

def run_algorithm4(
    user_idx: int,
    data: np.ndarray,
    labels: np.ndarray,
    tree: DecisionTree,
    alg2_output: Algorithm2Output,
    alg3_output: Algorithm3Output,
    rng_seed: Optional[int] = None,
) -> PredictionRecord:
    if rng_seed is not None:
        random.seed(rng_seed)
        np.random.seed(rng_seed)

    true_label = int(labels[user_idx])
    record = PredictionRecord(user_global_idx=user_idx, true_label=true_label)

    root_node = tree.root
    AF_real = 0.0
    pac_counter = [0]
    consumed = set()
    h_init = -1

    # Initialization: random action from root
    root_actions = alg3_output.retained_for_node(root_node.node_id)
    valid_candidates = [
        a for a in root_actions
        if not np.isnan(float(data[user_idx, a.feature_idx]))
    ]

    if valid_candidates:
        initial_action = random.choice(valid_candidates)
        j_init = initial_action.feature_idx
        h_init = initial_action.disease_class
        V_init = float(data[user_idx, j_init])
        record.initial_action_feat = j_init

        model = alg2_output.get_model(root_node.node_id, j_init)
        if model is not None:
            b_min = model.healthy_range.b_min_healthy
            b_max = model.healthy_range.b_max_healthy

            p_h_f = _compute_p_h_f(root_node, h_init)
            p_h_gt1_f = _compute_p_h_gt1_f(root_node)
            r_j_h_init = initial_action.action_weight
            numer_init = p_h_f * r_j_h_init
            delta_AF = _compute_AF_increment(p_h_f, r_j_h_init, p_h_gt1_f)
            AF_real = _update_AF(AF_real, delta_AF)
            rw_real = 1.0 - AF_real

            record.af_trace.append(FPGATraceStep(
                raw_value=V_init, b_min=b_min, b_max=b_max,
                r_j_h=r_j_h_init, p_h_f=p_h_f, p_h_gt1_f=p_h_gt1_f,
                numerator=numer_init, delta_AF=delta_AF,
                AF_real=AF_real, rw_real=rw_real,
                feature_idx=j_init, disease_class=h_init,
                node_id=root_node.node_id, step_type="pac",
            ))

            pac_counter[0] += 1
            record.total_pac_count += 1
            record.total_actions_applied += 1
            consumed.add((j_init, h_init))

            if _is_outside_healthy_range(V_init, b_min, b_max):
                record.decision = HealthDecision.UNHEALTHY
                record.is_correct = (true_label != HEALTHY_CLASS)
                record.alarm_class = h_init
                record.alarm_feature_idx = j_init
                return record

    all_disease_classes = sorted(ALL_DISEASE_CLASSES)
    decision = HealthDecision.UNKNOWN
    valid_node_ids = {e.node_id for e in alg2_output.perceptor_library}

    for current_focus in range(1, tree.depth() + 1):
        applicable_nodes = _find_all_applicable_nodes(
            user_idx, current_focus, tree, data, valid_node_ids, record,
        )
        if not applicable_nodes:
            decision = HealthDecision.SCREENING
            break

        record.max_focus_reached = current_focus
        level_decided = False

        for active_node in applicable_nodes:
            node_diseases = sorted(
                h for h in all_disease_classes if active_node.health_dist.get(h, 0) > 0
            )
            if not node_diseases:
                continue

            pac_counter = [0]
            decision, AF_real, alarm_class = _predict_at_node(
                user_idx, active_node, node_diseases, data,
                alg2_output, alg3_output, AF_real, pac_counter, record,
                consumed, h_init,
            )
            record.total_pac_count += pac_counter[0]

            if decision == HealthDecision.UNHEALTHY:
                record.alarm_class = alarm_class
                level_decided = True
                break
            if decision == HealthDecision.HEALTHY:
                level_decided = True
                break

        if level_decided:
            break
        if decision == HealthDecision.SCREENING:
            break

    if decision == HealthDecision.UNKNOWN:
        decision = HealthDecision.SCREENING

    if true_label == HEALTHY_CLASS:
        is_correct = (decision != HealthDecision.UNHEALTHY)
    else:
        is_correct = (decision == HealthDecision.UNHEALTHY)

    record.decision = decision
    record.is_correct = is_correct
    return record

def ten_fold_cv(data: np.ndarray,
    labels: np.ndarray,
    max_users: Optional[int] = None,
    rng_seed: int = 42,
    n_bins: int = DEFAULT_N_BINS,
    enable_forced_sex: bool = False,
) -> Algorithm4Output:
    n_total = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    n_users_total = data.shape[0]

    output = Algorithm4Output()
    random.seed(rng_seed)

    indices = list(range(n_total))
    random.shuffle(indices)
    fold_size = (n_total + 9) // 10

    for fold in range(10):
        start_idx = fold * fold_size
        end_idx = min(start_idx + fold_size, n_total)
        test_indices = indices[start_idx:end_idx]
        train_indices = [idx for idx in indices if idx not in test_indices]

        train_data = data[train_indices]
        train_labels = labels[train_indices]

        # Algorithm 1
        tree_i = build_decision_tree(train_data, train_labels)

        root_id = tree_i.root.node_id
        nodes_filter = [root_id]
        if not enable_forced_sex:
            level2_by_feat: Dict[int, List[TreeNode]] = defaultdict(list)
            for n in tree_i.nodes_by_level.get(2, []):
                if not n.is_leaf:
                    level2_by_feat[n.branching_feat_k].append(n)
            for feat_k, children in level2_by_feat.items():
                if len(children) >= 2:
                    nodes_filter.extend(c.node_id for c in children)

        # Algorithm 2
        alg2_i = run_algorithm2(tree_i, train_data, train_labels, n_bins, nodes_filter)

        # Algorithm 3
        alg3_i = run_algorithm3(alg2_i, tree_i, train_data, train_labels, nodes_filter, reset_per_h=False)

        # Algorithm 4 prediction — one record per test user
        for test_idx in test_indices:
            pred = run_algorithm4(test_idx, data, labels, tree_i, alg2_i, alg3_i, rng_seed)
            output.records.append(pred)

        n_correct = sum(1 for r in output.records if r.is_correct)
        n_evaluated = len(output.records)
        print(f"  Fold {fold+1}/10  evaluated={n_evaluated}/{n_total}  accuracy={n_correct/n_evaluated*100:.1f}%")

    output._recompute_stats()
    return output


# --- Reporting ---

def print_results(output: Algorithm4Output) -> None:
    n = len(output.records)
    print(f"\n{'='*60}")
    print("CDS PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"  Users evaluated  : {n}")
    print(f"  Overall accuracy : {output.overall_accuracy*100:.1f}%")
    print(f"  Sensitivity      : {output.sensitivity*100:.1f}%")
    print(f"  Specificity      : {output.specificity*100:.1f}%")
    print(f"  False alarm rate : {output.false_alarm_rate*100:.1f}%")
    print(f"  Screening count  : {output.n_screening}")

    # Per-class breakdown
    diseased = [r for r in output.records if r.true_is_diseased]
    if diseased:
        by_class: Dict[int, List[PredictionRecord]] = defaultdict(list)
        for r in diseased:
            by_class[r.true_label].append(r)
        print(f"\n  Per-class detection:")
        print(f"  {'class':>6} {'total':>6} {'detected':>9} {'rate':>7}")
        print(f"  {'-'*32}")
        for cls in sorted(by_class.keys()):
            recs = by_class[cls]
            detected = sum(1 for r in recs if r.decision == HealthDecision.UNHEALTHY)
            pct = detected / len(recs) * 100
            print(f"  {cls:6d} {len(recs):6d} {detected:9d} {pct:6.1f}%")

    print(f"{'='*60}")


# --- FPGA Fixed-Point Range Profiling ---

FPGA_TRACE_FIELDS: Tuple[str, ...] = (
    # Main PAC values
    "raw_value", "b_min", "b_max", "r_j_h",
    "p_h_f", "p_h_gt1_f", "numerator", "delta_AF", "AF_real", "rw_real",
    # RL lookahead values
    "AF_sim", "rw_sim", "best_rw",
    # Node routing values
    "branch_val", "branch_low", "branch_high",
    # Threshold comparison
    "rw_final", "threshold",
)

FPGA_FIELD_PAPER_REF: Dict[str, str] = {
    "raw_value":   "BD_m^k(o,u)",
    "b_min":       "Eq. 5 lower",
    "b_max":       "Eq. 5 upper",
    "r_j_h":       "Alg 3 r_{o|h}",
    "p_h_f":       "P(h, f^k_m)",
    "p_h_gt1_f":   "P(h>1, f^k_m)",
    "numerator":   "p_h_f * r_j_h",
    "delta_AF":    "Eq. 7 increment",
    "AF_real":     "Eq. 7 cumulative",
    "rw_real":     "Eq. 8: 1 - AF",
    "AF_sim":      "Alg 4 line 12",
    "rw_sim":      "Alg 4 line 13",
    "best_rw":     "Alg 4 line 14",
    "branch_val":  "BranchDef input",
    "branch_low":  "BranchDef low",
    "branch_high": "BranchDef high",
    "rw_final":    "Eq. 8 final",
    "threshold":   "Eq. 3 threshold",
}


def profile_fixed_point_ranges(
    data: np.ndarray,
    labels: np.ndarray,
    tree: DecisionTree,
    alg2_output: Algorithm2Output,
    alg3_output: Algorithm3Output,
    output_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Run all users through Algorithm 4 and record min/max/mean for every
    intermediate variable the FPGA will compute.  Returns the ranges dict
    and writes a human-readable report to *output_path*."""

    ranges: Dict[str, Dict[str, float]] = {}
    for fname in FPGA_TRACE_FIELDS:
        ranges[fname] = {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "count": 0}

    n_users = data.shape[0]
    for user_idx in range(n_users):
        record = run_algorithm4(user_idx, data, labels, tree,
                                alg2_output, alg3_output, rng_seed=42)
        for step in record.af_trace:
            for fname in FPGA_TRACE_FIELDS:
                val = getattr(step, fname)
                if np.isnan(val):
                    continue
                entry = ranges[fname]
                entry["min"] = min(entry["min"], val)
                entry["max"] = max(entry["max"], val)
                entry["sum"] += val
                entry["count"] += 1

    # Compute means
    for fname in FPGA_TRACE_FIELDS:
        entry = ranges[fname]
        entry["mean"] = entry["sum"] / entry["count"] if entry["count"] > 0 else 0.0

    # Print to console
    print(f"\n{'='*80}")
    print("FPGA FIXED-POINT RANGE ANALYSIS  (Algorithm 4 — all users)")
    print(f"{'='*80}")
    print(f"  {'Variable':<14} {'Paper Ref':<18} {'Min':>14} {'Max':>14} {'Mean':>14} {'Samples':>9}")
    print(f"  {'-'*83}")
    for fname in FPGA_TRACE_FIELDS:
        e = ranges[fname]
        ref = FPGA_FIELD_PAPER_REF[fname]
        if e["count"] > 0:
            print(f"  {fname:<14} {ref:<18} {e['min']:>14.6f} {e['max']:>14.6f} {e['mean']:>14.6f} {e['count']:>9d}")
        else:
            print(f"  {fname:<14} {ref:<18} {'(no data)':>14} {'':>14} {'':>14} {0:>9d}")
    print(f"{'='*80}")

    # Write to file
    if output_path is None:
        output_path = str(Path(__file__).parent.parent / "FixedPointAnalysis" / "VariableRangeData.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("FPGA Fixed-Point Range Analysis — Algorithm 4 Intermediate Variables\n")
        f.write(f"Dataset: {n_users} users, {N_FEATURES} features\n")
        f.write(f"Threshold: {DIAGNOSTIC_THRESHOLD}, U_MIN: {U_MIN}\n\n")

        f.write(f"{'Variable':<14} | {'Paper Reference':<18} | {'Min':>14} | {'Max':>14} | {'Mean':>14} | {'Samples':>9}\n")
        f.write("-" * 14 + "-+-" + "-" * 18 + "-+-" + "-" * 14 + "-+-" + "-" * 14 + "-+-" + "-" * 14 + "-+-" + "-" * 9 + "\n")

        for fname in FPGA_TRACE_FIELDS:
            e = ranges[fname]
            ref = FPGA_FIELD_PAPER_REF[fname]
            if e["count"] > 0:
                f.write(f"{fname:<14} | {ref:<18} | {e['min']:>14.6f} | {e['max']:>14.6f} | {e['mean']:>14.6f} | {e['count']:>9d}\n")
            else:
                f.write(f"{fname:<14} | {ref:<18} | {'(no data)':>14} | {'':>14} | {'':>14} | {0:>9d}\n")

        f.write(f"\n\nBit-Budget Guidance:\n")
        f.write(f"  raw_value spans the widest range — integer bits must cover [{ranges['raw_value']['min']:.1f}, {ranges['raw_value']['max']:.1f}]\n")
        f.write(f"  branch_val/branch_low/branch_high share the same range as raw_value\n")
        f.write(f"  b_min/b_max (healthy range) also share raw feature range\n")
        f.write(f"  Probabilities (p_h_f, p_h_gt1_f, r_j_h) are in [0,1] — need mostly fractional bits\n")
        f.write(f"  numerator = p_h_f * r_j_h — max {ranges['numerator']['max']:.6f}, product of two [0,1] values\n")
        f.write(f"  delta_AF = numerator / p_h_gt1_f — max {ranges['delta_AF']['max']:.6f}\n")
        f.write(f"  AF_real, rw_real are in [0,1] — 0 integer bits + sign bit sufficient\n")
        rw_sim_entry = ranges.get("rw_sim", {"min": 0, "max": 0})
        f.write(f"  rw_sim can go NEGATIVE (min={rw_sim_entry['min']:.6f}) — needs sign bit\n")
        f.write(f"  threshold is constant {DIAGNOSTIC_THRESHOLD} — stored as fixed-point literal\n")

    print(f"\n  Report written to: {output_path}")
    return ranges


# --- Main ---

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "data" / "arrhythmia.data")
    data, labels = load_dataset(path)

    max_u = int(sys.argv[2]) if len(sys.argv) > 2 else None
    output = ten_fold_cv(data, labels, max_users=max_u)
    print_results(output)
