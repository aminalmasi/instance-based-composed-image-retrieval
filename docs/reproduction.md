# Reproducing the thesis results

1. Download ICIR and validate the official metadata.
2. Extract SigLIP2 SO400M Patch16 384 query-image, contextualized-text, corpus, and database-image
   features using one frozen model revision.
3. Run the first-stage configuration and verify `59.64` mAP@100, `60.24` Macro-mAP@100, and `93.08`
   Recall@100.
4. Download the Route A checkpoint release and extract aligned SigLIP1 scores for the fixed top 100.
5. Generate query descriptors and extract aligned RZEn scores for Route B.
6. Rerank each shortlist using the four configured variables.

Expected final results are:

| Route | mAP@100 | Macro-mAP@100 | Recall@100 |
|---|---:|---:|---:|
| First stage | 59.64 | 60.24 | 93.08 |
| Route A | 66.05 | 64.14 | 93.08 |
| Route B | 66.42 | 65.65 | 93.08 |

Every run writes its resolved configuration, model revisions, array shapes, input checksums,
per-query metrics, top-100 rankings, and summary metrics. A regression run is accepted only if its
query count, candidate-pool statistics, and metrics match the recorded manifest within tolerance.
