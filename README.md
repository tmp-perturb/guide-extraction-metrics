# guide-extraction-metrics

Omnibenchmark metrics module for guide-extraction MEX outputs. It reuses the
existing extraction evaluation code to calculate pseudobulk and per-cell
metrics against the supplied reference matrix.

The entrypoint scores one lineage at a time and writes `{name}.scores.json`.
