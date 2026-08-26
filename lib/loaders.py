"""Shared data loaders for all benchmark levels.

Covers: MEX (Market Exchange Format), h5ad, h5mu, and assignment CSV.
"""

import re
import gzip, csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy import sparse


# ══════════════════════════════════════════════════════════════════════════
# MEX (Market Exchange Format)
# ══════════════════════════════════════════════════════════════════════════

def load_mex_matrix(mex_dir: Path) -> Tuple[sparse.csr_matrix, List[str], List[str]]:
    """Load a merged MEX trio: (n_cells, n_features) CSR, barcodes, feature IDs.

    Args:
        mex_dir: Directory containing merged_matrix.mtx.gz,
                 merged_barcodes.tsv.gz, merged_features.tsv.gz.

    Returns:
        (mat, barcodes, features) where mat is CSR, barcodes is list of
        strings, features is list of guide IDs (first column of features TSV).
    """
    mtx_path = mex_dir / "merged_matrix.mtx.gz"

    # Parse header for dimensions
    with gzip.open(mtx_path, "rt") as f:
        for line in f:
            if line.startswith("%"):
                continue
            n_cells, n_feat, nnz = map(int, line.strip().split())
            break

    # Read COO triples
    data_arr = np.zeros(nnz, dtype=np.float64)
    row_arr = np.zeros(nnz, dtype=np.int32)
    col_arr = np.zeros(nnz, dtype=np.int32)
    with gzip.open(mtx_path, "rt") as f:
        idx = 0; header_found = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if not header_found:
                header_found = True; continue
            r, c, v = line.split()
            row_arr[idx] = int(r) - 1
            col_arr[idx] = int(c) - 1
            data_arr[idx] = float(v)
            idx += 1

    mat = sparse.coo_matrix(
        (data_arr[:idx], (row_arr[:idx], col_arr[:idx])),
        shape=(n_cells, n_feat), dtype=np.float64
    ).tocsr()

    def _read_gz(p: Path) -> List[str]:
        with gzip.open(p, "rt") as fh:
            return [ln.rstrip("\n") for ln in fh]

    barcodes = _read_gz(mex_dir / "merged_barcodes.tsv.gz")
    features = [line.split("\t")[0] for line in _read_gz(mex_dir / "merged_features.tsv.gz")]

    return mat, barcodes, features


# ══════════════════════════════════════════════════════════════════════════
# h5ad / h5mu
# ══════════════════════════════════════════════════════════════════════════

def load_anndata_sparse(path: str, obs_index: str = "index"
                        ) -> Tuple[sparse.csr_matrix, List[str], Optional[List[int]]]:
    """Load a sparse AnnData matrix with cell barcodes and lane info.

    Args:
        path: Path to .h5ad file.
        obs_index: Column name in adata.obs to use as barcode.
                   If "index", uses adata.obs_names.

    Returns:
        (X_csr, barcodes, gem_groups).
        gem_groups is None if unavailable in obs.
    """
    import anndata as _ad
    ad = _ad.read_h5ad(path)
    X = ad.X
    if sparse.issparse(X):
        X_csr = X.tocsr() if sparse.isspmatrix_csc(X) else X.tocsr()
    else:
        X_csr = sparse.csr_matrix(X)

    if obs_index == "index":
        barcodes = list(ad.obs_names)
    else:
        barcodes = list(ad.obs[obs_index])

    gem_groups = None
    if "gem_group" in ad.obs:
        gem_groups = list(ad.obs["gem_group"].astype(int))

    return X_csr, barcodes, gem_groups


# ══════════════════════════════════════════════════════════════════════════
# Assignment CSV
# ══════════════════════════════════════════════════════════════════════════

def load_assignment_csv(fpath: str, sort_key: str = "prob_gaussian",
                        sort_desc: bool = True
                        ) -> Dict[tuple, List[Tuple[str, float, int]]]:
    """Load an assignment CSV into a per-cell sorted guide list.

    Expected CSV columns: cell, gRNA, UMI_counts [, prob_gaussian, ...].

    Args:
        fpath: Path to assignment CSV.
        sort_key: Column name used for per-cell ranking.
        sort_desc: True → higher score first.

    Returns:
        Dict mapping (lane, 16mer) → [(guide_name, score, umi), ...]
        sorted by score. Each key represents one cell.
    """
    import re
    bc_re = re.compile(r'^([ACGT]{16})-L(\d+)$')

    pgmm = defaultdict(list)
    has_prob = (sort_key == "prob_gaussian")

    with open(fpath) as f:
        for row in csv.DictReader(f):
            cell = row.get("cell", "").strip()
            guide = row.get("gRNA", "").strip()
            if not cell or not guide:
                continue
            m = bc_re.match(cell)
            if not m:
                continue
            key = (int(m.group(2)), m.group(1))  # (lane, 16mer)
            umi = int(float(row.get("UMI_counts", 0) or 0))
            if has_prob:
                prob = float(row.get("prob_gaussian", 0) or 0)
                pgmm[key].append((guide, prob, umi))
            else:
                score = float(row.get(sort_key, umi))
                pgmm[key].append((guide, score, umi))

    # Sort per cell
    if has_prob:
        for k in pgmm:
            pgmm[k].sort(key=lambda x: (-x[1], -x[2]))
    else:
        for k in pgmm:
            pgmm[k].sort(key=lambda x: -x[1] if sort_desc else x[1])

    return dict(pgmm)


def load_fishash_topk(fpath: str
                      ) -> Dict[tuple, List[Tuple[str, float, int]]]:
    """Load fishash full assignment CSV. Sorts per cell by log_pval ASC.

    Fishash raw output assigns many guides per cell (gpC ~20). This loader
    reads the full CSV and selects the top-1 guide per cell by log_pval ASC
    (most significant first). Consistent with scprocess-perturb
    standardize_assignment.py.

    Returns dict mapping (lane, 16mer) → [(guide, log_pval, umi), ...]
    sorted by log_pval ASC.
    """
    import re
    bc_re = re.compile(r'^([ACGT]{16})-L(\d+)$')

    per_cell = defaultdict(list)
    with open(fpath) as f:
        for row in csv.DictReader(f):
            cell = row.get("cell", "").strip()
            guide = row.get("gRNA", "").strip()
            if not cell or not guide:
                continue
            m = bc_re.match(cell)
            if not m:
                continue
            key = (int(m.group(2)), m.group(1))  # (lane, 16mer)
            umi = int(float(row.get("UMI_counts", 0) or 0))
            lp  = float(row.get("log_pval", 0) or 0)
            per_cell[key].append((guide, lp, umi))

    for k in per_cell:
        per_cell[k].sort(key=lambda x: x[1])  # log_pval ASC

    return dict(per_cell)


# ══════════════════════════════════════════════════════════════════════════
# Guide mapping (dual-guide CRISPRi: sgID → pair_id)
# ══════════════════════════════════════════════════════════════════════════

# sgID sanitisation — MUST match feature_reference_adapter.py's SANITIZE_RE
# (the extraction step rewrites guide feature IDs via [^a-zA-Z0-9_-] -> "_", e.g.
# "AAAS_-_53715438.23-P1P2" -> "AAAS_-_53715438_23-P1P2"). The guide CSV keeps the
# raw sgIDs, so sg2pair must register BOTH the raw and the sanitised spelling; the
# pipeline features (sanitised) then map onto pair IDs. Without this, guide->pair
# mapping silently fails and all pipeline-side metrics degenerate to 0/NaN.
_SGID_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_sgid(s: str) -> str:
    return _SGID_SANITIZE_RE.sub("_", s)


def load_guide_pair_mapping(csv_path: str
                            ) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
    """Build sgID→pair_id and pair_id→(sgA, sgB) mappings from guide CSV.

    CSV format (Replogle 2022 raw_guides): columns include
    'unique sgRNA pair ID', 'sgID_A', 'sgID_B'.

    Both the raw sgID and its adapter-sanitised form are registered as keys so
    the mapping works whichever spelling the extraction pipeline emits.

    Returns:
        (sg2pair, pair2guides) dicts.
    """
    sg2pair: Dict[str, str] = {}
    pair2guides: Dict[str, Tuple[str, str]] = {}

    def _register(sg: str, pid: str) -> None:
        sg2pair[sg] = pid
        sg2pair[_sanitize_sgid(sg)] = pid  # match sanitised pipeline feature IDs

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("unique sgRNA pair ID", "").strip()
            sgA = row.get("sgID_A", "").strip()
            sgB = row.get("sgID_B", "").strip()
            if pid and sgA:
                _register(sgA, pid)
                pair2guides[pid] = (sgA, sgB)
            if pid and sgB:
                _register(sgB, pid)

    return sg2pair, pair2guides


def load_ground_truth_sgid_ab(h5ad_path: str
                              ) -> Tuple[Dict[Tuple[int, str], str], int]:
    """Load ground truth pair labels from h5ad obs/sgID_AB.

    Returns:
        (gt_dict, n_total_cells) where gt_dict maps
        (gem_group, 16mer) → 'sgA|sgB' pair label.
    """
    import h5py, re
    bc_re = re.compile(r'^([ACGT]{16})-(\d+)$')

    with h5py.File(h5ad_path, "r") as f:
        cats = f['obs']['__categories']['sgID_AB']
        gt = {}
        n = len(f['obs']['cell_barcode'])
        for i in range(n):
            bc_raw = f['obs']['cell_barcode'][i]
            bc = bc_raw.decode() if isinstance(bc_raw, bytes) else str(bc_raw)
            m = bc_re.match(bc)
            if m:
                key = (int(m.group(2)), m.group(1))
                code = int(f['obs']['sgID_AB'][i])
                label = cats[code]
                label = label.decode() if isinstance(label, bytes) else str(label)
                gt[key] = label
    return gt, n
