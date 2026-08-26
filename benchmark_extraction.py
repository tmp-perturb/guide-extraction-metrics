#!/usr/bin/env python3
"""
Guide Extraction Benchmark — Pseudobulk + Per-cell comparison.

Compares pipeline MEX output against a reference guide count matrix
at two levels:
  - Pseudobulk: Spearman ρ, log1p MSE/MAE, cell recovery, guide recall
  - Per-cell:   Spearman ρ distribution, UMI correlation, guides/cell,
                tool-only%, UMI=1%

Supports dual-guide (CRISPRi, 2 guides per cell pair) and single-guide
(CRISPRko, 1 guide_ID per cell) via config reference.guide_mode.

Usage:
    # Dual-guide mode (Replogle 2022 style)
    python extraction/benchmark_extraction.py --config my_config.yaml

    # Single-guide mode (Papalexi 2021 style)
    python extraction/benchmark_extraction.py --config my_config.yaml --guide-mode single
"""

import sys, os, json, argparse, time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.loaders import load_mex_matrix, load_anndata_sparse, load_guide_pair_mapping
from lib.matching import extract_16mer, extract_lane_id
from lib.metrics import spearman_rho, per_cell_spearman, per_cell_log1p_error


# ══════════════════════════════════════════════════════════════════════════
# Data loaders
# ══════════════════════════════════════════════════════════════════════════

def _load_ref_for_extraction(path: str, barcode_key: str = "obs_names"
                             ) -> Tuple[sparse.csr_matrix, List[str], List[str],
                                         Dict[Tuple[str, int], int]]:
    """Load extraction reference: expression matrix + barcode index.

    Returns:
        ref_mat: (n_cells, n_guides) CSR matrix (raw UMI).
        barcodes: List of full barcode strings.
        guide_ids: List of guide feature IDs.
        bc_index: {(16mer, lane): row_idx} — unique per cell.
    """
    import anndata as _ad
    ad = _ad.read_h5ad(path)
    X = ad.X
    if sparse.isspmatrix_csc(X):
        X_csr = X.tocsr()
    elif sparse.isspmatrix_csr(X):
        X_csr = X
    else:
        X_csr = sparse.csr_matrix(X)

    if barcode_key == "obs_names":
        barcodes = list(ad.obs_names)
    else:
        barcodes = list(ad.obs[barcode_key])

    guide_ids = list(ad.var_names)

    # Build compound key index
    bc_index = {}
    for i, bc in enumerate(barcodes):
        m16 = extract_16mer(bc)
        lane = extract_lane_id(bc)
        bc_index[(m16, lane)] = i

    return X_csr, barcodes, guide_ids, bc_index


# ══════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════

def _compute_pseudobulk_metrics(pipe_mat, pipe_barcodes, pipe_features,
                                ref_mat, ref_barcodes, ref_bc_index,
                                ref_feature_ids, sg2pair, pair2col
                                ) -> Dict:
    """Compute extraction pseudobulk metrics."""
    metrics = {}

    pipe_compounds = [(extract_16mer(bc), extract_lane_id(bc)) for bc in pipe_barcodes]
    pipe_compound_set = set(pipe_compounds)
    ref_compound_set = set(ref_bc_index.keys())

    # Cell recovery
    n_ref = len(ref_barcodes)
    shared = pipe_compound_set & ref_compound_set
    metrics["cell_recovery_rate"] = float(len(shared) / max(n_ref, 1))
    metrics["n_shared_cells"] = int(len(shared))
    metrics["n_ref_cells"] = int(n_ref)

    # Map features → pair IDs
    feat_to_sg = {i: f for i, f in enumerate(pipe_features)}
    map_rows, map_cols = [], []
    for feat_col, sg in feat_to_sg.items():
        pid = sg2pair.get(sg, sg)  # fallback: feature IS the pair_id
        if pid and pid in pair2col:
            map_rows.append(feat_col)
            map_cols.append(pair2col[pid])

    if not map_rows:
        metrics["pseudobulk_spearman_rho"] = None
        return metrics

    map_mat = sparse.csr_matrix(
        (np.ones(len(map_rows), dtype=np.float64),
         (np.array(map_rows, dtype=np.int32), np.array(map_cols, dtype=np.int32))),
        shape=(pipe_mat.shape[1], len(ref_feature_ids)))

    pair_mat = pipe_mat @ map_mat

    # Match on compound key
    pipe_compound_to_row = {k: i for i, k in enumerate(pipe_compounds)}
    valid_pipe, valid_ref = [], []
    for key in shared:
        pi = pipe_compound_to_row.get(key)
        ri = ref_bc_index.get(key)
        if pi is not None and ri is not None:
            valid_pipe.append(pi); valid_ref.append(ri)

    n_shared = len(valid_pipe)
    metrics["n_shared_cells_crispr"] = n_shared

    if n_shared > 0:
        pipe_pb = np.asarray(pair_mat[valid_pipe, :].sum(axis=0)).ravel()
        ref_pb = np.asarray(ref_mat[valid_ref, :].sum(axis=0)).ravel()
        mask = (pipe_pb > 0) | (ref_pb > 0)
        n_pb = int(mask.sum())

        if n_pb >= 3:
            rho, _ = spearmanr(pipe_pb[mask], ref_pb[mask])
        else:
            rho = np.nan

        pipe_lp = np.log1p(pipe_pb[mask]); ref_lp = np.log1p(ref_pb[mask])
        metrics["pseudobulk_spearman_rho"] = float(rho) if not np.isnan(rho) else None
        metrics["pseudobulk_log1p_mse"] = float(np.mean((pipe_lp - ref_lp) ** 2))
        metrics["pseudobulk_log1p_mae"] = float(np.mean(np.abs(pipe_lp - ref_lp)))
        metrics["n_pairs_for_pb"] = n_pb
    else:
        metrics["pseudobulk_spearman_rho"] = None

    return metrics


def _compute_percell_metrics(pipe_mat, pipe_barcodes, pipe_features,
                             ref_mat, ref_barcodes, ref_bc_index,
                             ref_feature_ids, sg2pair, guide_mode="dual"
                             ) -> Dict:
    """Compute per-cell extraction metrics."""
    metrics = {}

    pipe_compounds = [(extract_16mer(bc), extract_lane_id(bc)) for bc in pipe_barcodes]
    pipe_compound_to_row = {k: i for i, k in enumerate(pipe_compounds)}
    ref_compound_set = set(ref_bc_index.keys())
    pair2col = {p: i for i, p in enumerate(ref_feature_ids)}
    shared = set(pipe_compounds) & ref_compound_set

    feat_to_sg = {i: f for i, f in enumerate(pipe_features)}

    rho_list = []; umi_pipe_list = []; umi_ref_list = []
    n_guides_pipe_list = []; n_guides_ref_list = []
    guide_umi_all = []; tool_only_count = 0; total_guide_umi = 0
    per_cell_detected = []

    for key in shared:
        pi = pipe_compound_to_row.get(key)
        ri = ref_bc_index.get(key)
        if pi is None or ri is None:
            continue

        pipe_row = pipe_mat[pi, :]
        ref_row = ref_mat[ri, :]

        # Build aligned vectors
        pipe_vec = np.zeros(len(ref_feature_ids), dtype=np.float64)
        for j in range(pipe_row.indptr[0], pipe_row.indptr[1]):
            col = pipe_row.indices[j]; sg = feat_to_sg.get(col)
            pid = sg2pair.get(sg, sg)  # use sgID if pair ID not mapped
            if pid in pair2col:
                pipe_vec[pair2col[pid]] += pipe_row.data[j]

        ref_vec = np.asarray(ref_row.toarray()).ravel() if sparse.issparse(ref_row) else ref_row

        # Per-guide UMI stats
        pipe_umis = pipe_vec[pipe_vec > 0]
        ref_umis = ref_vec[ref_vec > 0]
        guide_umi_all.extend(pipe_umis.tolist())
        total_guide_umi += len(pipe_umis)

        # tool-only%: guides in pipe but not in ref
        n_tool_only = np.sum((pipe_vec > 0) & (ref_vec == 0))
        tool_only_count += n_tool_only

        # Per-cell stats
        n_guides_pipe_list.append(int(np.sum(pipe_vec > 0)))
        n_guides_ref_list.append(int(np.sum(ref_vec > 0)))
        umi_pipe_list.append(float(pipe_vec.sum()))
        umi_ref_list.append(float(ref_vec.sum()))

        # Spearman ρ for this cell
        mask = (pipe_vec > 0) | (ref_vec > 0)
        if mask.sum() >= 5:
            sv = np.log1p(pipe_vec[mask]); rv = np.log1p(ref_vec[mask])
            if sv.std() > 1e-8 and rv.std() > 1e-8:
                try:
                    r, _ = spearmanr(sv, rv)
                    if not np.isnan(r): rho_list.append(float(r))
                except Exception: pass

        # Guide recall (per ground truth)
        if guide_mode == "dual":
            # For each cell, ground truth is 2 sgIDs → 1 pair_id
            gt_pair = None
            for j in range(ref_row.indptr[0], ref_row.indptr[1] if sparse.issparse(ref_row) else len(ref_row)):
                col_j = ref_row.indices[j] if sparse.issparse(ref_row) else j
                val = ref_row.data[j] if sparse.issparse(ref_row) else ref_row[0,j]
                if val > 0:
                    gt_pair = ref_feature_ids[col_j]
                    break
            if gt_pair:
                det = 0
                for j in range(pipe_row.indptr[0], pipe_row.indptr[1]):
                    col_j = pipe_row.indices[j]
                    sg_j = feat_to_sg.get(col_j, "")
                    pid_j = sg2pair.get(sg_j, sg_j)
                    if pid_j == gt_pair:
                        det = 1; break
                per_cell_detected.append(det)
        else:
            # Single guide: detect if guide_ID present
            gt_guide = None
            for j in range(ref_row.indptr[0], ref_row.indptr[1] if sparse.issparse(ref_row) else len(ref_row)):
                col_j = ref_row.indices[j] if sparse.issparse(ref_row) else j
                val = ref_row.data[j] if sparse.issparse(ref_row) else ref_row[0,j]
                if val > 0:
                    gt_guide = ref_feature_ids[col_j]; break
            if gt_guide:
                det = 1 if pipe_vec[ref_feature_ids.index(gt_guide) if gt_guide in ref_feature_ids else 0] > 0 else 0
                per_cell_detected.append(det)

    # Aggregate
    n_cells = len(umi_pipe_list)
    if n_cells > 0:
        rho_arr = np.array(rho_list, dtype=np.float32)
        metrics["per_cell_spearman_rho_median"] = float(np.median(rho_arr)) if len(rho_arr) > 0 else np.nan
        metrics["per_cell_spearman_rho_mean"] = float(np.mean(rho_arr)) if len(rho_arr) > 0 else np.nan
        metrics["pct_rho_negative"] = float(np.mean(rho_arr < 0) * 100) if len(rho_arr) > 0 else np.nan
        metrics["n_rho_valid"] = int(len(rho_arr))

        log_pipe = np.log1p(umi_pipe_list); log_ref = np.log1p(umi_ref_list)
        from scipy.stats import pearsonr as pr
        r_umi, _ = pr(log_pipe, log_ref)
        metrics["umi_corr_pearson"] = float(r_umi)

        metrics["median_guides_pipe"] = float(np.median(n_guides_pipe_list))
        metrics["median_guides_detected_ref"] = float(np.median(n_guides_ref_list))
        metrics["median_umi_pipe"] = float(np.median(umi_pipe_list))
        metrics["median_umi_ref"] = float(np.median(umi_ref_list))
        metrics["pct_tool_only"] = float(tool_only_count / max(total_guide_umi, 1) * 100)
        metrics["pct_umi_eq_1"] = float(np.sum(np.array(guide_umi_all) == 1) / max(len(guide_umi_all), 1) * 100)
        metrics["expected_guides_per_cell"] = 2 if guide_mode == "dual" else 1

    if per_cell_detected:
        rec_arr = np.array(per_cell_detected, dtype=np.float64)
        expected = 2 if guide_mode == "dual" else 1
        metrics["per_cell_guide_recall_median"] = float(np.median(rec_arr / expected))
        metrics["per_cell_guide_recall_full"] = float(np.mean(rec_arr >= expected))
        metrics["n_cells_for_recall"] = int(len(per_cell_detected))

    return metrics


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Guide Extraction Benchmark')
    parser.add_argument('--config', required=True, help='Benchmark config YAML')
    parser.add_argument('--guide-mode', choices=['dual', 'single'], default=None,
                        help='Override config guide_mode (dual: CRISPRi, single: CRISPRko)')
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ref_cfg = cfg.get('reference', {})
    guide_mode = args.guide_mode or ref_cfg.get('guide_mode', 'dual')
    output_dir = Path(cfg.get('output', {}).get('base_dir', './benchmark_output')) / 'extraction'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load reference
    ref_path = ref_cfg['extraction_ref']
    bc_key = ref_cfg.get('extraction_ref_barcodes', 'obs_names')
    print(f"Loading extraction reference: {ref_path}")
    ref_mat, ref_barcodes, ref_feature_ids, ref_bc_index = _load_ref_for_extraction(ref_path, bc_key)
    n_ref = len(ref_barcodes)
    print(f"  {ref_mat.shape[0]} cells × {ref_mat.shape[1]} guides, {ref_mat.nnz/1e6:.1f}M nnz")
    print(f"  {len(ref_bc_index)} unique (16mer, lane) keys")

    # 2. Load guide mapping (dual-guide mode only)
    sg2pair = {}
    pair2col = {p: i for i, p in enumerate(ref_feature_ids)}
    if guide_mode == "dual":
        guide_csv = ref_cfg.get('guide_csv')
        if guide_csv:
            sg2pair, _ = load_guide_pair_mapping(guide_csv)
            print(f"  {len(sg2pair)} sgID→pair mappings loaded")
    else:
        # Single-guide: 1:1 identity mapping
        sg2pair = {g: g for g in ref_feature_ids}
    pair2col = {p: i for i, p in enumerate(ref_feature_ids)}

    # 3. Process each tool
    all_metrics = {}
    for tool_name, tool_cfg in cfg.get('tools', {}).items():
        mex_dir = tool_cfg.get('extraction_mex')
        if not mex_dir:
            continue
        label = tool_cfg.get('label', tool_name)
        print(f"\nTool: {label}")

        t0 = time.time()
        pipe_mat, pipe_barcodes, pipe_features = load_mex_matrix(Path(mex_dir))
        print(f"  {pipe_mat.shape[0]} cells × {pipe_mat.shape[1]} guides, {pipe_mat.nnz/1e6:.2f}M nnz")

        # Pseudobulk
        pb = _compute_pseudobulk_metrics(
            pipe_mat, pipe_barcodes, pipe_features,
            ref_mat, ref_barcodes, ref_bc_index,
            ref_feature_ids, sg2pair, pair2col)

        # Per-cell
        pc = _compute_percell_metrics(
            pipe_mat, pipe_barcodes, pipe_features,
            ref_mat, ref_barcodes, ref_bc_index,
            ref_feature_ids, sg2pair, guide_mode)

        metrics = {**pb, **pc, "wall_time_s": round(time.time() - t0, 1),
                   "_label": label}
        all_metrics[tool_name] = metrics

        print(f"  Pb ρ={metrics.get('pseudobulk_spearman_rho', 'N/A')}, "
              f"Per-cell ρ median={metrics.get('per_cell_spearman_rho_median', 'N/A')}, "
              f"Recovery={metrics.get('cell_recovery_rate', 0):.4f}")

        # Save per-tool
        out_json = output_dir / f"{tool_name}.json"
        clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
        with open(out_json, 'w') as f:
            json.dump(clean, f, indent=2, default=str)

    # 4. Cross-tool summary
    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nSummary: {summary_path}")


if __name__ == '__main__':
    main()
