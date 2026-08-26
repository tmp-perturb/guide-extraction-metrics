#!/usr/bin/env python3
"""Omnibenchmark metrics module: guide_extraction_metrics.

Scores one guide-extraction MEX output against a reference guide count matrix,
at pseudobulk and per-cell level. This is a thin re-orchestration wrapper: it
reuses the EXISTING evaluation logic verbatim (vendored `benchmark_extraction.py`
+ `lib/`), so results are identical to the standalone benchmark. Only the outer
per-tool loop / config parsing is replaced by the Omnibenchmark CLI contract:
Omnibenchmark fans out over methods/datasets, and this entrypoint scores exactly
ONE lineage (one method's MEX vs the reference).

Omnibenchmark CLI contract:
    --output_dir <dir> --name <node_id>
    --guide_extraction.matrix   <merged_matrix.mtx.gz>     (upstream methods output)
    --guide_extraction.barcodes <merged_barcodes.tsv.gz>
    --guide_extraction.features <merged_features.tsv.gz>
    --data.reference <reference.h5ad>   (extraction reference matrix, raw UMI)
    --data.guide_csv <guide_library.csv>  (dual-guide sgID->pair mapping)
    --guide_mode <dual|single>            (parameter)

Output written into <output_dir>:
    {name}.scores.json    (pseudobulk + per-cell metrics for this lineage)
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # vendored lib/ + benchmark_extraction.py

from lib.loaders import load_mex_matrix, load_guide_pair_mapping  # noqa: E402
from benchmark_extraction import (  # noqa: E402
    _load_ref_for_extraction,
    _compute_pseudobulk_metrics,
    _compute_percell_metrics,
)


def _mex_dir_from_inputs(matrix, barcodes, features, workdir):
    """load_mex_matrix() expects a dir with merged_{matrix,barcodes,features}.
    Present the three injected input files under those canonical names."""
    d = os.path.join(workdir, "mex")
    os.makedirs(d, exist_ok=True)
    for src, name in ((matrix, "merged_matrix.mtx.gz"),
                      (barcodes, "merged_barcodes.tsv.gz"),
                      (features, "merged_features.tsv.gz")):
        dst = os.path.join(d, name)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    return d


def main():
    p = argparse.ArgumentParser(description="Omnibenchmark module: guide_extraction_metrics")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", default="node")
    # stage inputs (dotted ids kept literal, read via getattr)
    p.add_argument("--guide_extraction.matrix", required=True)
    p.add_argument("--guide_extraction.barcodes", required=True)
    p.add_argument("--guide_extraction.features", required=True)
    p.add_argument("--data.reference", required=True)
    p.add_argument("--data.guide_csv", required=True)
    # parameter
    p.add_argument("--guide_mode", default="dual", choices=["dual", "single"])
    args = p.parse_args()

    mtx = getattr(args, "guide_extraction.matrix")
    bcs = getattr(args, "guide_extraction.barcodes")
    fts = getattr(args, "guide_extraction.features")
    ref_path = getattr(args, "data.reference")
    guide_csv = getattr(args, "data.guide_csv")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- load reference (same loader as the standalone benchmark) ----
    ref_mat, ref_barcodes, ref_feature_ids, ref_bc_index = _load_ref_for_extraction(ref_path)

    # ---- guide mapping (dual: sgID->pair; single: identity) ----
    if args.guide_mode == "dual":
        sg2pair, _ = load_guide_pair_mapping(guide_csv)
    else:
        sg2pair = {g: g for g in ref_feature_ids}
    pair2col = {p_: i for i, p_ in enumerate(ref_feature_ids)}

    # ---- load this lineage's pipeline MEX ----
    with tempfile.TemporaryDirectory() as work:
        mex_dir = _mex_dir_from_inputs(mtx, bcs, fts, work)
        from pathlib import Path
        pipe_mat, pipe_barcodes, pipe_features = load_mex_matrix(Path(mex_dir))

        # ---- compute metrics (identical functions to the standalone benchmark) ----
        pb = _compute_pseudobulk_metrics(
            pipe_mat, pipe_barcodes, pipe_features,
            ref_mat, ref_barcodes, ref_bc_index,
            ref_feature_ids, sg2pair, pair2col)
        pc = _compute_percell_metrics(
            pipe_mat, pipe_barcodes, pipe_features,
            ref_mat, ref_barcodes, ref_bc_index,
            ref_feature_ids, sg2pair, args.guide_mode)

    metrics = {**pb, **pc}
    out = os.path.join(args.output_dir, f"{args.name}.scores.json")
    clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
    with open(out, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    print("guide_extraction_metrics: wrote", os.path.basename(out))


if __name__ == "__main__":
    main()
