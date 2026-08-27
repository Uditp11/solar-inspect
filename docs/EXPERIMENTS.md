# Experiments

Every training run, in order. A run with `dirty=yes` was made against a
working tree that did not match its commit, so its numbers are not
reproducible from that SHA -- treat them as indicative only.

**`split` is the first 8 hex of `configs/d1_split.json`'s sha256.** D1 was
deduplicated and re-split at `6f197e6`, so a row on `4cbb0c3d` and a row on
`af8781b1` are numbers on two different datasets and comparing them is a
mistake with no visible symptom. The column exists to make that impossible to
do by accident.

| run | git SHA | dirty | split | config | seed | eval split | epochs | wall | macro-F1 | acc | null acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260827T134744Z | `6f1f6fe` | yes | `4cbb0c3d` | `configs/cls_baseline.yaml` | 0 | val | 30 | 14 s | **0.5907** | 0.7783 | 0.5008 |
| 20260827T135123Z | `6bc2d76` | no | `4cbb0c3d` | `configs/cls_baseline.yaml` | 0 | val | 30 | 14 s | **0.5907** | 0.7783 | 0.5008 |
