# guide-extraction-metrics

Omnibenchmark metrics module for guide-extraction MEX outputs. It reuses the
existing extraction evaluation code to calculate pseudobulk and per-cell
metrics against the supplied reference matrix, and reports extraction-level
signal concentration summaries directly from the MEX output.

The entrypoint scores one lineage at a time and writes `{name}.scores.json`.

Reported metrics include pseudobulk and per-cell count agreement, cell
recovery, and the extraction-level summaries `delta_median`,
`entropy_lib_median`, `entropy_det_median`, and `k80_median`. Guide recall is
not part of the Omnibenchmark extraction metric output; the historical
evaluator remains vendored for archived-result parity.
