"""
hidden_state_compare.py
=======================

Companion analysis tool for `slam_model_s2t_hidden.py`.

Given three dump directories (one per model: CoM / ECoM-direct /
ECoM-progressive), this script:

  1. Pairs samples across the three runs by `question_key`.
  2. For each matched sample, aligns the *text* token streams of the
     three runs by token id (LCS by default; first-occurrence as a
     fast fallback).
  3. For each aligned position triple, computes the per-layer cosine
     similarity:
            sim_direct[ell] = cos( h_com[ell],  h_ecom_direct[ell] )
            sim_prog[ell]   = cos( h_com[ell],  h_ecom_prog[ell]   )
     where ell indexes the L+1 hidden states returned by
     `output_hidden_states=True` (embedding layer included).
  4. Averages over all matched positions across all matched samples,
     and produces:
        - layer_similarity.json   (per-layer mean + std + #pairs)
        - layer_similarity.csv    (same data, machine-readable table)
        - layer_similarity.png    (matplotlib figure: two curves with
                                   95% CI bands, x = layer index)

Usage
-----

    python hidden_state_compare.py \\
        --com           /nfs/.../hidden_dump/com \\
        --ecom-direct   /nfs/.../hidden_dump/ecom_direct \\
        --ecom-prog     /nfs/.../hidden_dump/ecom_progressive \\
        --out           /nfs/.../analysis_out \\
        --align         lcs            # or first-occurrence
        --max-samples   200            # cap for speed (optional)
        --skip-bos                     # drop the leading BOS-like ids
        --layers-fp32                  # cast hiddens to fp32 before cos

Each --com / --ecom-direct / --ecom-prog argument may be:
  * a directory containing per-sample `*.pt` dumps + (optional)
    `index.jsonl`; or
  * a single `.pt` file.

The matched dimensionality must be identical across the three runs
(same backbone). The script will hard-fail with a clear message if not.
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("hidden_state_compare")


# ----------------------------------------------------------------------
# Dump loading
# ----------------------------------------------------------------------
def _list_pt_files(path: str) -> List[str]:
    """Return all .pt files under `path` (file or dir, recursive one level)."""
    if os.path.isfile(path):
        return [path] if path.endswith(".pt") else []
    out: List[str] = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isfile(full) and full.endswith(".pt"):
            out.append(full)
        elif os.path.isdir(full):
            for sub in sorted(os.listdir(full)):
                if sub.endswith(".pt"):
                    out.append(os.path.join(full, sub))
    return out


def _load_dumps(path: str) -> Dict[str, dict]:
    """
    Load all .pt dumps under `path`, return mapping
    question_key -> dump dict (already on CPU).
    """
    pt_files = _list_pt_files(path)
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found under {path}")

    out: Dict[str, dict] = {}
    for fp in pt_files:
        try:
            obj = torch.load(fp, map_location="cpu")
        except Exception as e:
            logger.warning("Failed to load %s: %s", fp, e)
            continue
        qk = obj.get("question_key") or os.path.splitext(os.path.basename(fp))[0]
        if qk in out:
            logger.warning("Duplicate question_key %s in %s — keeping the first.", qk, fp)
            continue
        out[qk] = obj
    logger.info("Loaded %d dumps from %s", len(out), path)
    return out


# ----------------------------------------------------------------------
# Text-token extraction + alignment
# ----------------------------------------------------------------------
def _extract_text_stream(dump: dict, skip_bos: bool = False) -> Tuple[List[int], List[int]]:
    """
    Return (token_ids_list, step_indices_list) over text-only steps.
    `token_ids` and `phase` from the dump are tensors / lists of length T.
    """
    token_ids = dump["token_ids"]
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    phase = dump.get("phase")
    if phase is None:
        # fall back: any token id >= 0 is treated as a text step
        phase = ["text" if tid >= 0 else "audio" for tid in token_ids]

    out_ids: List[int] = []
    out_steps: List[int] = []
    for i, (tid, ph) in enumerate(zip(token_ids, phase)):
        if ph != "text":
            continue
        if tid < 0:
            continue
        out_ids.append(int(tid))
        out_steps.append(i)

    if skip_bos and out_ids and out_ids[0] in (0, 1, 2):  # crude heuristic
        out_ids = out_ids[1:]
        out_steps = out_steps[1:]

    return out_ids, out_steps


def _lcs_indices(a: Sequence[int], b: Sequence[int]) -> List[Tuple[int, int]]:
    """
    Return list of (i, j) pairs where a[i] == b[j], constituting a
    longest common subsequence. O(len(a) * len(b)) time / memory.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    # Use uint16 if possible to save memory; LCS length <= min(n, m)
    dtype = np.int32
    dp = np.zeros((n + 1, m + 1), dtype=dtype)
    for i in range(n):
        ai = a[i]
        row_prev = dp[i]
        row_cur = dp[i + 1]
        for j in range(m):
            if ai == b[j]:
                row_cur[j + 1] = row_prev[j] + 1
            else:
                v1 = row_cur[j]
                v2 = row_prev[j + 1]
                row_cur[j + 1] = v1 if v1 >= v2 else v2

    # backtrack
    pairs: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1, j] >= dp[i, j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _first_occurrence_indices(a: Sequence[int], b: Sequence[int]) -> List[Tuple[int, int]]:
    """
    Greedy linear-scan alignment: for each id, pair its k-th occurrence
    in `a` with its k-th occurrence in `b`. Cheap and order-preserving.
    """
    bucket_a: Dict[int, List[int]] = defaultdict(list)
    bucket_b: Dict[int, List[int]] = defaultdict(list)
    for i, t in enumerate(a):
        bucket_a[t].append(i)
    for j, t in enumerate(b):
        bucket_b[t].append(j)
    pairs: List[Tuple[int, int]] = []
    for t, ia in bucket_a.items():
        ib = bucket_b.get(t, [])
        for i, j in zip(ia, ib):
            pairs.append((i, j))
    pairs.sort()
    return pairs


def _align_triplet(
    tids_com: Sequence[int],
    tids_d: Sequence[int],
    tids_p: Sequence[int],
    method: str,
) -> List[Tuple[int, int, int]]:
    """
    Align three streams. We first LCS-align com vs ecom_direct, then
    LCS-align com vs ecom_prog, and keep only the com-indices present
    in both pairings.
    """
    fn = _lcs_indices if method == "lcs" else _first_occurrence_indices
    pairs_d = fn(tids_com, tids_d)
    pairs_p = fn(tids_com, tids_p)
    map_d = {i: j for i, j in pairs_d}
    map_p = {i: k for i, k in pairs_p}
    triples: List[Tuple[int, int, int]] = []
    for i in sorted(map_d.keys() & map_p.keys()):
        triples.append((i, map_d[i], map_p[i]))
    return triples


# ----------------------------------------------------------------------
# Cosine similarity accumulator
# ----------------------------------------------------------------------
class LayerSimAccumulator:
    """
    Accumulates per-layer cosine similarity over arbitrary number of
    matched positions, with online mean + Welford variance.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.count = 0
        self.mean = np.zeros(num_layers, dtype=np.float64)
        self.m2 = np.zeros(num_layers, dtype=np.float64)

    def update(self, sims: np.ndarray) -> None:
        """
        sims: [N, L] cosine similarities for N positions.
        """
        if sims.size == 0:
            return
        for row in sims:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.m2 += delta * delta2

    @property
    def std(self) -> np.ndarray:
        if self.count < 2:
            return np.zeros_like(self.mean)
        return np.sqrt(self.m2 / (self.count - 1))

    @property
    def ci95(self) -> np.ndarray:
        if self.count < 2:
            return np.zeros_like(self.mean)
        return 1.96 * self.std / np.sqrt(self.count)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def _cos_per_layer(
    h_ref: torch.Tensor,  # [T, L, D]
    h_other: torch.Tensor,  # [T, L, D]
    fp32: bool,
) -> np.ndarray:
    if h_ref.shape != h_other.shape:
        raise ValueError(
            f"Hidden shapes differ: {tuple(h_ref.shape)} vs {tuple(h_other.shape)}"
        )
    if fp32:
        h_ref = h_ref.float()
        h_other = h_other.float()
    # F.cosine_similarity over the last dim (D), per (t, l)
    sim = F.cosine_similarity(h_ref, h_other, dim=-1)  # [T, L]
    return sim.cpu().numpy()


def _gather_hidden(
    dump: dict, positions: Sequence[int]
) -> torch.Tensor:
    """
    Index into dump['hiddens'] (shape [T, L+1, D]) along the time axis.
    """
    h = dump["hiddens"]
    if not isinstance(h, torch.Tensor):
        h = torch.as_tensor(h)
    if h.numel() == 0 or len(positions) == 0:
        return torch.empty(0)
    return h.index_select(0, torch.as_tensor(list(positions), dtype=torch.long))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--com", required=True, help="dump dir/file for CoM Reasoning")
    parser.add_argument("--ecom-direct", required=True, help="dump dir/file for ECoM-direct")
    parser.add_argument("--ecom-prog", required=True, help="dump dir/file for ECoM-progressive")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--align", choices=["lcs", "first-occurrence"], default="lcs")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = use all matched samples")
    parser.add_argument("--skip-bos", action="store_true", help="drop the first text token (heuristic)")
    parser.add_argument("--layers-fp32", action="store_true", help="cast to fp32 before cosine")
    parser.add_argument("--no-plot", action="store_true", help="skip PNG plot")
    parser.add_argument("--title", default=None, help="optional plot title")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    os.makedirs(args.out, exist_ok=True)

    dumps_com = _load_dumps(args.com)
    dumps_d = _load_dumps(args.ecom_direct)
    dumps_p = _load_dumps(args.ecom_prog)

    keys = sorted(set(dumps_com) & set(dumps_d) & set(dumps_p))
    logger.info(
        "Question keys: com=%d, ecom_direct=%d, ecom_prog=%d, intersection=%d",
        len(dumps_com), len(dumps_d), len(dumps_p), len(keys),
    )
    if not keys:
        logger.error("No common question_key across the three dumps. Aborting.")
        sys.exit(2)
    if args.max_samples > 0:
        keys = keys[: args.max_samples]
        logger.info("Capped to %d samples", len(keys))

    # discover layer count + hidden dim from first sample
    sample = dumps_com[keys[0]]
    if "hiddens" not in sample or sample["hiddens"].numel() == 0:
        logger.error("First com dump has empty hiddens, cannot determine layer count.")
        sys.exit(2)
    num_layers = int(sample["hiddens"].shape[1])
    hidden_dim = int(sample["hiddens"].shape[2])
    logger.info("Discovered num_layers=%d, hidden_dim=%d", num_layers, hidden_dim)

    acc_direct = LayerSimAccumulator(num_layers)
    acc_prog = LayerSimAccumulator(num_layers)

    per_sample_rows: List[dict] = []
    n_pairs_total = 0
    n_samples_used = 0

    for qk in keys:
        d_com = dumps_com[qk]
        d_dir = dumps_d[qk]
        d_prog = dumps_p[qk]

        tids_com, steps_com = _extract_text_stream(d_com, skip_bos=args.skip_bos)
        tids_dir, steps_dir = _extract_text_stream(d_dir, skip_bos=args.skip_bos)
        tids_prog, steps_prog = _extract_text_stream(d_prog, skip_bos=args.skip_bos)

        if not tids_com or not tids_dir or not tids_prog:
            logger.debug("Skip %s: empty text stream", qk)
            continue

        triples = _align_triplet(tids_com, tids_dir, tids_prog, args.align)
        if not triples:
            logger.debug("Skip %s: no aligned positions", qk)
            continue

        # map text-stream indices back to original step indices
        pos_com = [steps_com[i] for (i, _, _) in triples]
        pos_dir = [steps_dir[j] for (_, j, _) in triples]
        pos_prog = [steps_prog[k] for (_, _, k) in triples]

        h_com = _gather_hidden(d_com, pos_com)
        h_dir = _gather_hidden(d_dir, pos_dir)
        h_prog = _gather_hidden(d_prog, pos_prog)

        try:
            sim_dir = _cos_per_layer(h_com, h_dir, fp32=args.layers_fp32)
            sim_prog = _cos_per_layer(h_com, h_prog, fp32=args.layers_fp32)
        except ValueError as e:
            logger.error("Sample %s shape mismatch: %s — skipped.", qk, e)
            continue

        acc_direct.update(sim_dir)
        acc_prog.update(sim_prog)
        n_pairs_total += len(triples)
        n_samples_used += 1
        per_sample_rows.append({
            "question_key": qk,
            "n_pairs": len(triples),
            "mean_sim_direct": float(sim_dir.mean()),
            "mean_sim_prog": float(sim_prog.mean()),
        })

    if n_samples_used == 0:
        logger.error("No usable samples after alignment. Aborting.")
        sys.exit(3)

    logger.info(
        "Aggregation done: %d samples, %d aligned positions in total.",
        n_samples_used, n_pairs_total,
    )

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    per_layer = []
    for ell in range(num_layers):
        per_layer.append({
            "layer": ell,
            "ecom_direct_mean": float(acc_direct.mean[ell]),
            "ecom_direct_std": float(acc_direct.std[ell]),
            "ecom_direct_ci95": float(acc_direct.ci95[ell]),
            "ecom_progressive_mean": float(acc_prog.mean[ell]),
            "ecom_progressive_std": float(acc_prog.std[ell]),
            "ecom_progressive_ci95": float(acc_prog.ci95[ell]),
            "gap_prog_minus_direct": float(acc_prog.mean[ell] - acc_direct.mean[ell]),
        })

    summary = {
        "config": {
            "com": args.com,
            "ecom_direct": args.ecom_direct,
            "ecom_progressive": args.ecom_prog,
            "align": args.align,
            "skip_bos": args.skip_bos,
            "layers_fp32": args.layers_fp32,
            "max_samples": args.max_samples,
        },
        "stats": {
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "num_samples_used": n_samples_used,
            "num_aligned_positions": n_pairs_total,
        },
        "per_layer": per_layer,
        "per_sample": per_sample_rows,
        "aggregate": {
            "mean_sim_direct_over_layers": float(acc_direct.mean.mean()),
            "mean_sim_progressive_over_layers": float(acc_prog.mean.mean()),
            "mean_gap_prog_minus_direct": float((acc_prog.mean - acc_direct.mean).mean()),
            "upper_layer_gap_prog_minus_direct": float(
                (acc_prog.mean[num_layers // 2 :] - acc_direct.mean[num_layers // 2 :]).mean()
            ),
        },
    }

    json_path = os.path.join(args.out, "layer_similarity.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", json_path)

    csv_path = os.path.join(args.out, "layer_similarity.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_layer[0].keys()))
        w.writeheader()
        w.writerows(per_layer)
    logger.info("Wrote %s", csv_path)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if not args.no_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover
            logger.warning("matplotlib not available, skipping plot: %s", e)
            return

        xs = np.arange(num_layers)
        m_d = acc_direct.mean
        ci_d = acc_direct.ci95
        m_p = acc_prog.mean
        ci_p = acc_prog.ci95

        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
        ax.plot(xs, m_d, label="ECoM-direct vs CoM", color="#d8554a", linewidth=2)
        ax.fill_between(xs, m_d - ci_d, m_d + ci_d, color="#d8554a", alpha=0.18)
        ax.plot(xs, m_p, label="ECoM-progressive vs CoM", color="#2c7fb8", linewidth=2)
        ax.fill_between(xs, m_p - ci_p, m_p + ci_p, color="#2c7fb8", alpha=0.18)
        ax.set_xlabel("Layer index (0 = embedding)")
        ax.set_ylabel("Cosine similarity of last-token hidden state")
        ax.set_title(
            args.title
            or f"Per-layer hidden-state similarity vs CoM"
        )
        ax.set_xlim(0, num_layers - 1)
        ax.set_ylim(max(0.0, min(m_d.min(), m_p.min()) - 0.05), 1.0)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left")
        fig.tight_layout()
        png_path = os.path.join(args.out, "layer_similarity.png")
        fig.savefig(png_path)
        plt.close(fig)
        logger.info("Wrote %s", png_path)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print()
    print(f"=== Hidden-state similarity summary ===")
    print(f"  samples used      : {n_samples_used}")
    print(f"  aligned positions : {n_pairs_total}")
    print(f"  num_layers (+emb) : {num_layers}")
    print(f"  hidden_dim        : {hidden_dim}")
    print(f"  mean sim (direct->com)        : {summary['aggregate']['mean_sim_direct_over_layers']:.4f}")
    print(f"  mean sim (progressive->com)   : {summary['aggregate']['mean_sim_progressive_over_layers']:.4f}")
    print(f"  mean gap (prog - direct)      : {summary['aggregate']['mean_gap_prog_minus_direct']:+.4f}")
    print(f"  mean gap on upper half layers : {summary['aggregate']['upper_layer_gap_prog_minus_direct']:+.4f}")
    print()


if __name__ == "__main__":
    main()
