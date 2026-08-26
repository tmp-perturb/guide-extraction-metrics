"""Benchmark metric functions used across all levels.

All functions are stateless and accept numpy arrays.
No class wrappers — import and call directly.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import List, Optional, Tuple


# ── Correlation ────────────────────────────────────────────────────────

def spearman_rho(x: np.ndarray, y: np.ndarray,
                 mask_zeros: bool = True) -> Tuple[float, int]:
    """Spearman rank correlation between two vectors.

    Args:
        x, y: Input vectors.
        mask_zeros: If True, only compute on positions where
                    (x > 0) | (y > 0).

    Returns:
        (rho, n_genes_used)
    """
    if mask_zeros:
        mask = (x > 0) | (y > 0)
        x = x[mask]; y = y[mask]
    n = len(x)
    if n < 3:
        return (np.nan, n)
    try:
        rho, _ = spearmanr(x, y)
        return (float(rho) if not np.isnan(rho) else np.nan, n)
    except Exception:
        return (np.nan, n)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation between two vectors."""
    if len(x) < 3:
        return np.nan
    try:
        r, _ = pearsonr(x, y)
        return float(r) if not np.isnan(r) else np.nan
    except Exception:
        return np.nan


def per_cell_spearman(sc_vecs: np.ndarray, ref_vecs: np.ndarray,
                      min_common: int = 10) -> np.ndarray:
    """Per-cell Spearman ρ on log1p-transformed vectors.

    Args:
        sc_vecs: (n_cells, n_genes) pipeline matrix.
        ref_vecs: (n_cells, n_genes) reference matrix.
        min_common: Minimum genes with signal in either to
                    compute ρ for a cell.

    Returns:
        Float array of length n_cells (NaN where insufficient genes
        or zero variance).
    """
    n_cells = sc_vecs.shape[0]
    rho_arr = np.full(n_cells, np.nan, dtype=np.float32)
    for i in range(n_cells):
        sv = sc_vecs[i]; rv = ref_vecs[i]
        mask = (sv > 0) | (rv > 0)
        if mask.sum() < min_common:
            continue
        sv_lp = np.log1p(sv[mask]); rv_lp = np.log1p(rv[mask])
        if sv_lp.std() < 1e-8 or rv_lp.std() < 1e-8:
            continue
        try:
            r, _ = spearmanr(sv_lp, rv_lp)
            if not np.isnan(r):
                rho_arr[i] = float(r)
        except Exception:
            pass
    return rho_arr


def per_cell_log1p_error(sc_vecs: np.ndarray, ref_vecs: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell log1p MSE and MAE."""
    n_cells = sc_vecs.shape[0]
    mse = np.zeros(n_cells, dtype=np.float32)
    mae = np.zeros(n_cells, dtype=np.float32)
    for i in range(n_cells):
        sl = np.log1p(sc_vecs[i])
        rl = np.log1p(ref_vecs[i])
        mse[i] = float(np.mean((sl - rl) ** 2))
        mae[i] = float(np.mean(np.abs(sl - rl)))
    return mse, mae


# ── Recovery & Accuracy ─────────────────────────────────────────────────

def cell_recovery_rate(shared_keys, n_ref_total: int) -> float:
    """Fraction of reference cells matched in pipeline output.

    Args:
        shared_keys: Set of matched (16mer, lane) or 16mer keys.
        n_ref_total: Total number of cells in reference.
    """
    return len(shared_keys) / max(n_ref_total, 1)


def guide_recall(per_cell_detected: List[int], expected: int = 2) -> dict:
    """Guide-level recall statistics.

    Args:
        per_cell_detected: Number of expected guides detected per cell.
        expected: Expected number of guides (2 for dual, 1 for single).

    Returns:
        dict with median, full (fraction with all expected detected),
        and n_cells tested.
    """
    arr = np.array(per_cell_detected, dtype=np.float64) / expected
    return {
        "recall_median": float(np.median(arr)),
        "recall_full": float(np.mean(arr >= 1.0)),
        "n_cells_for_recall": len(arr),
    }


# ── Clustering ──────────────────────────────────────────────────────────

def adjusted_rand_index(y_true, y_pred) -> float:
    """Adjusted Rand Index for clustering agreement."""
    from sklearn.metrics import adjusted_rand_score
    try:
        return float(adjusted_rand_score(y_true, y_pred))
    except Exception:
        return np.nan


# ── KD Efficiency ───────────────────────────────────────────────────────

def log2_fold_change(mean_assigned: float, mean_control: float,
                     eps: float = 1e-8) -> float:
    """log2 fold change: log2(assigned + ε) - log2(control + ε)."""
    import math
    return math.log2(max(mean_assigned + eps, eps)) - math.log2(max(mean_control + eps, eps))
