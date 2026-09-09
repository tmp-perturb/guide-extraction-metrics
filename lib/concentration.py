"""Extraction-level signal concentration summaries.

These calculations are copied from the existing per-cell difficulty-table
logic.  They intentionally operate on the extraction MEX directly so the
extraction metrics module can report method-level summaries without depending
on the guide-assignment benchmark's difficulty-table branch.
"""

from typing import Dict

import numpy as np
from scipy import sparse


def signal_concentration_metrics(matrix: sparse.spmatrix) -> Dict[str, object]:
    """Return extraction-level concentration summaries for one MEX matrix.

    The per-cell definitions match ``guide_assignment_difficulty``:
    ``delta`` is the top-1/top-2 UMI gap normalized by cell library size,
    ``entropy_lib`` is Shannon entropy normalized by the total feature
    library, ``entropy_det`` is normalized by the number of detected guides,
    and ``k80`` is the number of ranked guides needed to explain 80% of UMIs.

    The Omnibenchmark score is a median across non-empty cells.  The number of
    valid cells is emitted as denominator metadata, not as a performance
    outcome.
    """
    mat = matrix.tocsr() if not sparse.isspmatrix_csr(matrix) else matrix
    n_total_guides = mat.shape[1]

    deltas = []
    entropy_lib = []
    entropy_det = []
    k80 = []

    for i in range(mat.shape[0]):
        vals = mat.getrow(i).data.astype(np.float64, copy=False)
        if vals.size == 0:
            continue

        libsize = float(vals.sum())
        if not np.isfinite(libsize) or libsize <= 0:
            continue

        sorted_vals = np.sort(vals)[::-1]
        top1 = sorted_vals[0]
        top2 = sorted_vals[1] if sorted_vals.size >= 2 else 0.0
        deltas.append((top1 - top2) / max(libsize, 1e-8))

        freqs = vals / libsize
        h = -float(np.sum(freqs * np.log2(freqs + 1e-300)))
        entropy_lib.append(h / np.log2(max(n_total_guides, 2)))
        entropy_det.append(h / np.log2(max(vals.size, 2)))

        cumulative = np.cumsum(sorted_vals)
        k = int(np.searchsorted(cumulative, 0.80 * libsize, side="right") + 1)
        k80.append(min(k, vals.size))

    n_valid = len(deltas)
    if n_valid == 0:
        return {
            "delta_median": None,
            "entropy_lib_median": None,
            "entropy_det_median": None,
            "k80_median": None,
            "n_cells_for_concentration": 0,
        }

    return {
        "delta_median": float(np.median(deltas)),
        "entropy_lib_median": float(np.median(entropy_lib)),
        "entropy_det_median": float(np.median(entropy_det)),
        "k80_median": float(np.median(k80)),
        "n_cells_for_concentration": int(n_valid),
    }
