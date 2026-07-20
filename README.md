# Instance-Based Composed Image Retrieval

This repository implements the two-stage retrieval pipeline developed for the thesis
**Instance-Based Composed Image Retrieval with Vision-Language Models**.

An input query contains a reference image and a textual modification. A database candidate is
relevant only when it depicts the same literal instance and satisfies the requested modification.
The implementation keeps these two signals visible and combines them through multiplicative
fusion.

## Pipeline

The first stage uses **SigLIP2 SO400M Patch16 384**:

1. Encode query and database images.
2. Apply leave-one-instance-out image centering.
3. Contextualize the short modification with a generic subject corpus.
4. Compute image and text similarities inside each query's instance-specific hard-negative pool.
5. Multiply both similarities and retain the top 100 candidates.

Two alternative second-stage routes rerank that fixed shortlist:

- **Route A:** an ILIAS-trained SigLIP1 linear probe supplies an instance-focused image score,
  while frozen SigLIP1 supplies a contextualized text score.
- **Route B:** RZEnEmbed-7B re-embeds candidates using an instance descriptor and supplies new
  image-side and text-side scores.

Both routes multiply their two new signals with the two first-stage SigLIP2 signals. The candidate
set is unchanged, so reranking can improve AP@100 but cannot change Recall@100.

## Reported results

The final full-query ICIR protocol uses each query's own positives and curated hard negatives.

| Method | mAP@100 | Macro-mAP@100 | Recall@100 |
|---|---:|---:|---:|
| SigLIP2 first stage | 59.64 | 60.24 | 93.08 |
| Route A: ILIAS linear probe | 66.05 | 64.14 | 93.08 |
| Route B: descriptor-guided RZEn | **66.42** | **65.65** | 93.08 |

These values are regression targets, not bundled predictions. Reproducing them requires the ICIR
dataset, the documented feature banks, and the released lightweight checkpoints.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[models,dev]"
```

For metric and pipeline tests without loading neural models:

```bash
pip install -e ".[dev]"
pytest
```

## Data

Datasets and large feature arrays are intentionally not committed. ICIR is distributed under its
own CC-BY-NC-SA license. Follow [docs/data.md](docs/data.md) to download it and create a local
manifest. ILIAS must likewise be obtained from its original source under its own terms.

Every command receives paths through YAML configuration or command-line arguments. No source file
contains machine-specific absolute paths.

## Repository layout

```text
configs/       Reproducible data, first-stage, and reranking configurations
docs/          Data preparation, methods, metrics, and reproduction notes
src/           Installable retrieval and training package
scripts/       Thin experiment entry points
slurm/         Optional cluster launch templates
tests/         Unit and end-to-end tests using synthetic data
checkpoints/   Release manifest and checksums for lightweight trained heads
```

## Reproducibility

The default metric profile matches the experiments used for the thesis results:

- AP@100 uses a denominator of `min(number of positives, 100)`.
- Recall@100 uses all positives as its denominator.
- mAP@100 averages AP over queries.
- Macro-mAP@100 averages AP within each instance and then across instances.
- Candidate ties are resolved deterministically using their original index.

An all-positive AP denominator is also available for auditing the equation stated in the thesis.
See [docs/metrics.md](docs/metrics.md).

## Acknowledgments

This project builds on the ICIR benchmark and BASIC implementation, SigLIP and SigLIP2, ILIAS,
and RZEnEmbed. Their code, data, and model weights remain subject to their original licenses.

## License

The original code in this repository is released under the Apache License 2.0. Dataset annotations,
images, pretrained models, generated descriptors, and third-party code are not relicensed here.
