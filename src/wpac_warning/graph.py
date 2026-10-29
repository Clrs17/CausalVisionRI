# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Causal prior graph loader: converts phase-1 causal_graph_edges.json into
PyG tensors (edge_index + edge attributes) for the Causal GNN branch.

Node order = config data.env_vars (must match the panel columns), followed by
ONE extra target readout node (index n_env) that collects the prior
env -> DV24_YJ edges (B1 design: DV24 readout node; it has no input features
and no outgoing edges — env->env message passing is unchanged).

Edge attributes per prior edge: [lag / tau_max, sign (+1/-1), |coef|]
  - lag is normalized by data.tau_max (default 4) so the attribute is in [0,1];
  - weak edges with p_value > data.prior_min_pvalue (default 0.05) are dropped
    (the count is prior-dependent; never hard-code it).

The lag-0 subgraph is still returned for reference / potential diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

TARGET_NODE_NAME = "DV24_YJ"  # phase-1 outcome node; becomes the readout node


def load_prior_graph(cfg, expected_sha256: Optional[str] = None,
                     require_metadata: bool = False,
                     strict_schema: bool = True) -> dict:
    """Load the phase-1 discovery graph into PyG tensors.

    Parameters
    ----------
    expected_sha256 : verify the JSON file against this hash before parsing
        (the holdout protocol requires it; there is no fallback file).
    require_metadata : the JSON must carry phase-1 ``metadata`` with
        ``mask_type`` equal to ``data.prior_mask_type`` (default ``xyz``).
    strict_schema : the JSON ``var_names`` must equal ``data.env_vars`` plus
        exactly one ``DV24_YJ`` node; a mismatch is a hard error (it means the
        graph was built from a different variable set / panel).
    """
    env_vars = list(cfg.get("data.env_vars"))
    idx = {v: i for i, v in enumerate(env_vars)}
    n_env = len(env_vars)
    target_idx = n_env  # the readout node id (last node)
    tau_max = float(cfg.get("data.tau_max", 4))
    min_p = cfg.get("data.prior_min_pvalue", 0.05)
    min_p = float(min_p) if min_p is not None else None

    gpath = Path(cfg.get("data.graph_edges_json"))
    if not gpath.is_file():
        raise FileNotFoundError(f"prior graph not found: {gpath}")
    source_sha = None
    if expected_sha256:
        from wpac_warning.protocol import verify_sha256
        source_sha = verify_sha256(gpath, expected_sha256, "prior graph")
    raw = json.loads(gpath.read_text(encoding="utf-8"))
    edges = raw["edges"]
    metadata = raw.get("metadata", {})

    if require_metadata:
        want_mask = str(cfg.get("data.prior_mask_type", "xyz"))
        got_mask = metadata.get("mask_type")
        if got_mask != want_mask:
            raise ValueError(
                f"prior graph metadata.mask_type={got_mask!r} != {want_mask!r} "
                f"({gpath}); refusing to train on an unmasked graph")
    if strict_schema and "var_names" in raw:
        names = list(raw["var_names"])
        expected = env_vars + [TARGET_NODE_NAME]
        if sorted(names) != sorted(expected) or len(names) != len(expected):
            raise ValueError(
                "prior graph var_names do not match data.env_vars + "
                f"{TARGET_NODE_NAME}\n  graph:   {names}\n  config:  {expected}")


    src, dst, attr = [], [], []
    src0, dst0 = [], []
    lag0_pos, attr0 = [], []
    n_dropped_weak = 0
    n_dropped_node = 0
    for e in edges:
        s, d = e["src"], e["dst"]
        # endpoints must be env vars, except dst == DV24_YJ which maps to the
        # target readout node (B1)
        if s not in idx:
            n_dropped_node += 1
            continue
        if d in idx:
            d_idx = idx[d]
        elif d == TARGET_NODE_NAME:
            d_idx = target_idx
        else:
            n_dropped_node += 1
            continue
        # A8: drop statistically weak prior edges
        if min_p is not None and float(e.get("p_value", 0.0)) > min_p:
            n_dropped_weak += 1
            continue
        src.append(idx[s])
        dst.append(d_idx)
        coef = float(e["coef"])
        a = [float(e["lag"]) / tau_max, 1.0 if coef > 0 else -1.0, abs(coef)]
        attr.append(a)
        if int(e["lag"]) == 0:
            src0.append(idx[s])
            dst0.append(d_idx)
            # column index of this lag-0 edge in the FULL edge list; lets the
            # GNN pick lag-0 attentions positionally (the same (src, dst) pair
            # may also appear at other lags, so (src, dst) keys are ambiguous —
            # 21 such duplicate pairs exist after the p-value filter)
            lag0_pos.append(len(src) - 1)
            attr0.append(a)

    n_nodes = n_env + 1  # +1 target readout node
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(attr, dtype=torch.float32) if attr else torch.zeros((0, 3))
    edge_index_lag0 = torch.tensor([src0, dst0], dtype=torch.long) if src0 else torch.zeros((2, 0), dtype=torch.long)
    lag0_pos_t = torch.tensor(lag0_pos, dtype=torch.long) if lag0_pos else torch.zeros(0, dtype=torch.long)
    edge_attr_lag0 = torch.tensor(attr0, dtype=torch.float32) if attr0 else torch.zeros((0, 3))

    return {
        "env_vars": env_vars,
        "n_env": n_env,
        "n_nodes": n_nodes,            # n_env + 1 target readout node
        "target_idx": target_idx,
        "edge_index": edge_index,      # (2, E) env->env + env->target edges
        "edge_attr": edge_attr,        # (E, 3): lag/tau_max, sign, |coef|
        "edge_index_lag0": edge_index_lag0,  # contemporaneous subgraph (reference)
        "lag0_pos": lag0_pos_t,        # (E0,) column idx of lag-0 edges in edge_index
        "edge_attr_lag0": edge_attr_lag0,    # (E0, 3) attrs of the lag-0 edges
        "n_dropped_weak": n_dropped_weak,
        "n_dropped_node": n_dropped_node,
        "n_edges_loaded": int(len(edges)),
        "source_path": str(gpath),
        "source_sha256": source_sha,
        "metadata": metadata,
    }


def shuffle_prior_graph(prior: dict, seed: int) -> dict:
    """Degree-preserving null model of the prior graph (ablation C3).

    Randomizes the connectivity while keeping every edge's attribute triple
    (lag, sign, |coef|), so the GNN capacity, the adjacency_ce target
    machinery, and the loss weights are all unchanged — only "which variable
    talks to which" is randomized.

    Two strata are shuffled independently so the readout mechanism survives:
      - env->env edges: directed edge-swap Markov chain ((a->b),(c->d) ->
        (a->d),(c->b)), rejecting self-loops and duplicate (src,dst) pairs.
        Per-node in- and out-degrees within this subgraph are preserved
        EXACTLY;
      - env->target readout edges: sources re-drawn from the env vars with
        replacement (the masked baseline has four edges from two vars). The
        readout node's in-degree and the stratum's attr multiset are preserved;
        the per-node out-degree INTO the target is what this stratum's
        randomization varies — that re-assignment IS the null, by design.

    The lag-0 subgraph and lag0_pos indices are recomputed from the shuffled
    edge list by the same attribute rule (lag == 0) as the loader.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    ei = prior["edge_index"].numpy()
    ea = prior["edge_attr"].numpy()
    n_env = prior["n_env"]
    t_idx = prior["target_idx"]

    env_mask = ei[1] != t_idx
    env_edges = np.where(env_mask)[0]
    tgt_edges = np.where(~env_mask)[0]

    # env->env stratum: degree-preserving directed edge swaps
    src = ei[0].copy()
    dst = ei[1].copy()
    edge_set = set(zip(src[env_edges].tolist(), dst[env_edges].tolist()))
    n_swap_target = max(50 * len(env_edges), 5000)
    n_done = 0
    n_tries = 0
    while n_done < n_swap_target and n_tries < 50 * n_swap_target:
        n_tries += 1
        i, j = rng.choice(env_edges, size=2, replace=False)
        a, b = src[i], dst[i]
        c, d = src[j], dst[j]
        if a == c or b == d:
            continue
        if a == d or c == b:  # would create self-loops
            continue
        if (a, d) in edge_set or (c, b) in edge_set:
            continue
        edge_set.discard((a, b))
        edge_set.discard((c, d))
        dst[i], dst[j] = d, b
        edge_set.add((a, d))
        edge_set.add((c, b))
        n_done += 1
    if n_done < n_swap_target:
        raise RuntimeError(f"C3 shuffle: only {n_done}/{n_swap_target} swaps succeeded")

    # env->target stratum: re-draw sources from env vars (with replacement —
    # the original stratum has repeated sources across lags)
    if len(tgt_edges):
        new_sources = rng.choice(n_env, size=len(tgt_edges), replace=True)
        src[tgt_edges] = new_sources

    ei_new = np.stack([src, dst])
    # rebuild lag-0 subgraph from the same attribute rule (attr[0] == 0.0)
    lag0_cols = np.where(ea[:, 0] == 0.0)[0]
    out = dict(prior)
    out["edge_index"] = torch.tensor(ei_new, dtype=torch.long)
    out["edge_attr"] = torch.tensor(ea, dtype=torch.float32)
    out["edge_index_lag0"] = torch.tensor(ei_new[:, lag0_cols], dtype=torch.long)
    out["edge_attr_lag0"] = torch.tensor(ea[lag0_cols], dtype=torch.float32)
    out["lag0_pos"] = torch.tensor(lag0_cols, dtype=torch.long)
    out["shuffle_seed"] = seed
    return out
