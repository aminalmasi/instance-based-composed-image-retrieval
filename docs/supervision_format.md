# Linear-probe supervision format

Route A training reads one CSV row per embedding sample. Required columns are:

| Column | Meaning |
|---|---|
| `group_id` | Query-centered training group |
| `instance_id` | Identity used for instance-balanced evaluation |
| `split` | `train`, `val`, or `test` |
| `role` | `query`, `positive`, or `negative` |
| `embedding_bank` | Symbolic array name from the YAML configuration |
| `embedding_row` | Row index in that array |

Optional `positive_rank` orders positives before the top-three selection. Optional
`negative_hardness` accepts `hard`, `very_hard`, and `super_hard`; other labels remain usable and
are sampled after these ordered levels.

This normalized interface separates the published trainer from local ILIAS filenames. A preparation
script converts the original positivity, hard-negative, and split files into this table while
preserving query groups and preventing instance leakage.
