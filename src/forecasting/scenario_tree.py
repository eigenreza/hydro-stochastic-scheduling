"""
Scenario tree construction for the multistage stochastic program.

Design
------
Planning horizon: T = 52 weeks.
Stage structure: 5 stages (decision points at weeks 1, 14, 27, 40, 52).
  - Stage boundaries chosen to align roughly with hydrological seasons:
      Stage 1:  week  1–13  (winter)
      Stage 2:  week 14–26  (spring)
      Stage 3:  week 27–39  (summer)
      Stage 4:  week 40–52  (autumn)
    Each stage except the last constitutes a decision period during which
    the operator observes the realised outcome and re-optimises. This gives
    a 4-recourse-stage tree (initial decision + 3 recourse decisions).
  - Branching factor: 4 per stage (4 inflow × price realisations per node).
  - Tree nodes: 1 + 4 + 16 + 64 + 256 = 341 nodes, 256 leaf scenarios.

This is explicitly documented as a tractable approximation to SDDP:
  - No Benders cut computation / value function approximation across iterations.
  - The stage-wise decisions are solved via the extensive-form deterministic
    equivalent (for the closed-loop policy) or the expected-value relaxation
    (for the open-loop policy).
  - The stage structure is coarser than a weekly SDDP implementation would use.

Scenario reduction: k-means clustering of the raw (200) Monte Carlo paths
into 256 representative paths (4^4 = 256), with probability weights
proportional to cluster sizes.  Reference: Heitsch & Römisch (2003),
"Scenario reduction algorithms in stochastic programming."

Saved to results/scenarios/:
  - scenario_tree.pkl      — full tree structure as nested dict
  - scenario_paths.csv     — 256 full inflow+price paths (52 weeks each)
  - scenario_weights.csv   — probability weights (sum to 1)
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

LOG = logging.getLogger(__name__)
SEED = 42
SCENARIOS_DIR = Path("results/scenarios")
SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

# Stage boundaries (1-indexed week numbers inclusive)
STAGE_WEEKS = [
    list(range(1, 14)),    # Stage 1: weeks  1–13
    list(range(14, 27)),   # Stage 2: weeks 14–26
    list(range(27, 40)),   # Stage 3: weeks 27–39
    list(range(40, 53)),   # Stage 4: weeks 40–52 (week 53 treated as 52)
]
N_STAGES = len(STAGE_WEEKS)
BRANCH_FACTOR = 4   # branching per stage
N_LEAF = BRANCH_FACTOR ** N_STAGES    # 256 leaf scenarios


@dataclass
class ScenarioNode:
    """A node in the scenario tree."""
    stage: int
    branch: int                          # branch index within parent's children
    weeks: list[int]                     # ISO week numbers covered by this node
    inflow: np.ndarray                   # shape (len(weeks),), GWh/week
    price: np.ndarray                    # shape (len(weeks),), NOK/MWh
    probability: float = 1.0
    children: list["ScenarioNode"] = field(default_factory=list)
    parent: "ScenarioNode | None" = field(default=None, repr=False)


def _reduce_to_n(inflow_mat: np.ndarray, price_mat: np.ndarray, n: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    K-means scenario reduction: cluster raw Monte Carlo paths into n scenarios.
    Returns (inflow_reduced, price_reduced, weights) each of length n.
    inflow_mat: (T, N_raw) array; price_mat: (T, N_raw) array.
    """
    T, N_raw = inflow_mat.shape
    # Feature matrix: concatenate inflow and price paths (normalised)
    combined = np.vstack([inflow_mat, price_mat]).T   # (N_raw, 2T)
    std = combined.std(axis=0) + 1e-8
    combined_norm = combined / std

    km = KMeans(n_clusters=n, random_state=SEED, n_init=10)
    labels = km.fit_predict(combined_norm)

    inflow_out = np.zeros((T, n))
    price_out = np.zeros((T, n))
    weights = np.zeros(n)
    for k in range(n):
        mask = labels == k
        if mask.sum() == 0:
            # Empty cluster: use the k-means centroid directly
            weights[k] = 1.0 / N_raw
            std = (np.vstack([inflow_mat, price_mat]).T).std(axis=0) + 1e-8
            centroid_raw = km.cluster_centers_[k] * std
            inflow_out[:, k] = centroid_raw[:T]
            price_out[:, k] = np.maximum(centroid_raw[T:], 1.0)
        else:
            weights[k] = mask.sum() / N_raw
            inflow_out[:, k] = inflow_mat[:, mask].mean(axis=1)
            price_out[:, k] = np.maximum(price_mat[:, mask].mean(axis=1), 1.0)

    return inflow_out, price_out, weights


def build_scenario_tree(
    inflow_scenarios: pd.DataFrame,
    price_scenarios: pd.DataFrame,
    forecast_start_week: pd.Timestamp,
    n_raw: int = 200,
    custom_stage_weeks: list[list[int]] | None = None,
) -> dict:
    """
    Build a scenario tree from raw Monte Carlo scenario paths.

    inflow_scenarios: (T, n_raw) DataFrame of weekly inflow paths (GWh/week).
    price_scenarios:  (T, n_raw) DataFrame of weekly price paths (NOK/MWh).
    custom_stage_weeks: override the default 4-stage structure. Each element
        is a list of 1-indexed week numbers belonging to that stage. Useful
        for sub-horizon re-solves in the rolling-horizon closed-loop policy
        where T < 52.

    Returns a dict containing:
        'root': the root ScenarioNode (full tree accessible via .children)
        'leaf_inflow': (T, n_leaves) array of leaf-node inflow paths
        'leaf_price':  (T, n_leaves) array of leaf-node price paths
        'leaf_weights': (n_leaves,) array of scenario probabilities (sum to 1)
        'forecast_dates': DatetimeIndex of the forecast weeks
    """
    rng = np.random.default_rng(SEED)
    T = inflow_scenarios.shape[0]
    assert inflow_scenarios.shape == price_scenarios.shape

    inflow_mat = inflow_scenarios.values   # (T, N_raw)
    price_mat = price_scenarios.values

    # Stage weeks: use custom if provided, else default; trim to valid weeks
    _stage_weeks: list[list[int]] = custom_stage_weeks if custom_stage_weeks is not None else STAGE_WEEKS
    _stage_weeks = [[w for w in sw if 1 <= w <= T] for sw in _stage_weeks]
    _stage_weeks = [sw for sw in _stage_weeks if sw]
    _n_stages = len(_stage_weeks)

    LOG.info("Reducing %d raw scenarios → %d representative leaf scenarios …",
             n_raw, N_LEAF)

    # Stage-wise scenario reduction (nested clustering)
    # For each stage, cluster the sub-paths within that stage given
    # the parent cluster assignment from previous stages.

    def _build_subtree(
        inflow_subset: np.ndarray,   # (T, N_sub) — all T weeks for this subset
        price_subset: np.ndarray,
        stage: int,
        parent_prob: float,
        global_week_offset: int,
    ) -> ScenarioNode:
        """Recursively build a subtree node."""
        stage_week_indices = [w - 1 for w in _stage_weeks[stage]]  # 0-indexed
        stage_inflow = inflow_subset[stage_week_indices, :]
        stage_price = price_subset[stage_week_indices, :]

        if stage == _n_stages - 1:
            # Leaf nodes — no further branching; cluster into BRANCH_FACTOR groups
            n_sub = inflow_subset.shape[1]
            n_clusters = min(BRANCH_FACTOR, n_sub)
            inf_r, pr_r, wts = _reduce_to_n(stage_inflow, stage_price, n_clusters, rng)

            node = ScenarioNode(
                stage=stage, branch=0,
                weeks=_stage_weeks[stage],
                inflow=np.zeros(len(STAGE_WEEKS[stage])),
                price=np.zeros(len(STAGE_WEEKS[stage])),
                probability=parent_prob,
            )
            for k in range(n_clusters):
                leaf = ScenarioNode(
                    stage=stage,
                    branch=k,
                    weeks=_stage_weeks[stage],
                    inflow=inf_r[:, k],
                    price=pr_r[:, k],
                    probability=parent_prob * wts[k],
                    parent=node,
                )
                node.children.append(leaf)
            return node
        else:
            # Intermediate node — cluster into BRANCH_FACTOR child groups
            n_sub = inflow_subset.shape[1]
            n_clusters = min(BRANCH_FACTOR, n_sub)
            inf_r, pr_r, wts = _reduce_to_n(stage_inflow, stage_price, n_clusters, rng)

            node = ScenarioNode(
                stage=stage, branch=0,
                weeks=_stage_weeks[stage],
                inflow=np.zeros(len(STAGE_WEEKS[stage])),
                price=np.zeros(len(STAGE_WEEKS[stage])),
                probability=parent_prob,
            )

            # For each cluster, select the raw scenarios closest to the centroid
            combined_stage = np.vstack([stage_inflow, stage_price]).T
            std = combined_stage.std(axis=0) + 1e-8
            cn = combined_stage / std
            centroids_norm = np.vstack([inf_r, pr_r]).T / std

            for k in range(n_clusters):
                dists = np.linalg.norm(cn - centroids_norm[k], axis=1)
                subset_mask = np.argsort(dists)[:max(1, n_sub // n_clusters)]

                child = ScenarioNode(
                    stage=stage,
                    branch=k,
                    weeks=_stage_weeks[stage],
                    inflow=inf_r[:, k],
                    price=pr_r[:, k],
                    probability=parent_prob * wts[k],
                    parent=node,
                )
                child.children = _build_subtree(
                    inflow_subset[:, subset_mask],
                    price_subset[:, subset_mask],
                    stage + 1,
                    parent_prob * wts[k],
                    global_week_offset + len(_stage_weeks[stage]),
                ).children
                for ch in child.children:
                    ch.parent = child
                node.children.append(child)
            return node

    root = _build_subtree(inflow_mat, price_mat, stage=0, parent_prob=1.0,
                          global_week_offset=0)

    # Collect leaf paths and weights
    def _collect_leaves(node: ScenarioNode, path_inf: list, path_pr: list,
                        leaves: list) -> None:
        if not node.children:
            leaves.append((node.probability, path_inf + list(node.inflow),
                           path_pr + list(node.price)))
        else:
            for ch in node.children:
                _collect_leaves(ch,
                                path_inf + list(node.inflow),
                                path_pr + list(node.price),
                                leaves)

    # Skip root (which has zero placeholder arrays); collect paths from root's children
    leaves_root = []
    for child in root.children:
        _collect_leaves(child, [], [], leaves_root)
    leaves_root.sort(key=lambda x: -x[0])

    n_leaves = len(leaves_root)
    leaf_weights = np.array([l[0] for l in leaves_root])
    leaf_weights /= leaf_weights.sum()   # normalise
    leaf_inflow = np.column_stack([l[1] for l in leaves_root])   # (52, n_leaves)
    leaf_price = np.column_stack([l[2] for l in leaves_root])

    LOG.info("Scenario tree built: %d leaf scenarios, weight sum=%.4f",
             n_leaves, leaf_weights.sum())

    tree = {
        "root": root,
        "leaf_inflow": leaf_inflow,
        "leaf_price": leaf_price,
        "leaf_weights": leaf_weights,
        "forecast_dates": inflow_scenarios.index,
        "stage_weeks": STAGE_WEEKS,
        "n_stages": N_STAGES,
        "branch_factor": BRANCH_FACTOR,
    }
    return tree


def save_scenario_tree(tree: dict, tag: str = "") -> None:
    suffix = f"_{tag}" if tag else ""
    pkl_path = SCENARIOS_DIR / f"scenario_tree{suffix}.pkl"
    with open(pkl_path, "wb") as f:
        # Save without the full root node (too large); save arrays and metadata
        save_data = {k: v for k, v in tree.items() if k != "root"}
        pickle.dump(save_data, f)

    dates = tree["forecast_dates"]
    n_leaves = tree["leaf_inflow"].shape[1]
    inflow_df = pd.DataFrame(
        tree["leaf_inflow"],
        index=dates,
        columns=[f"s{j:03d}" for j in range(n_leaves)],
    )
    price_df = pd.DataFrame(
        tree["leaf_price"],
        index=dates,
        columns=[f"s{j:03d}" for j in range(n_leaves)],
    )
    weights_s = pd.Series(
        tree["leaf_weights"],
        index=[f"s{j:03d}" for j in range(n_leaves)],
        name="probability",
    )
    inflow_df.to_csv(SCENARIOS_DIR / f"scenario_inflow{suffix}.csv")
    price_df.to_csv(SCENARIOS_DIR / f"scenario_price{suffix}.csv")
    weights_s.to_csv(SCENARIOS_DIR / f"scenario_weights{suffix}.csv")
    LOG.info("Scenario tree saved to %s", pkl_path)


def load_scenario_tree(tag: str = "") -> dict:
    suffix = f"_{tag}" if tag else ""
    pkl_path = SCENARIOS_DIR / f"scenario_tree{suffix}.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f)
