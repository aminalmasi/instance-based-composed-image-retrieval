# Data preparation

## ICIR

Download ICIR from its official distribution or Hugging Face dataset repository. The dataset is
licensed separately under CC-BY-NC-SA and is not redistributed here.

The local-folder reader expects the official metadata plus image directories:

```text
data/icir/
  query/
  database/
  query_files.csv
  database_files.csv
```

The pipeline converts this metadata into a manifest with stable query and database identifiers.
For each query, its candidate pool contains the positives matching the instance and modification,
plus hard negatives marked `hn` for that same instance.

## ILIAS

ILIAS supplies instance-retrieval supervision for Route A. Store it outside the repository and
provide paths through `configs/data/ilias.yaml`. Generated splits, embeddings, and hard-negative
tables are referenced by manifests and are never hard-coded in the package.

## Large features

Feature arrays are memory-mapped NumPy files. A manifest records:

- model identifier and revision;
- embedding dimension and dtype;
- image preprocessing parameters;
- metadata CSV and feature-array paths;
- number and order of samples;
- SHA-256 checksums for released artifacts.
