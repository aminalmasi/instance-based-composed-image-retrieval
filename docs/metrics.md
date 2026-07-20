# Evaluation protocol

Every ICIR query is ranked only against its associated positive images and curated hard negatives.
The global image collection is not treated as one shared negative pool.

For a query `q`, let `r_q(k)` indicate whether the candidate at rank `k` is relevant and let
`P_q(k)` denote precision through rank `k`.

The metric implementation used to produce the final thesis artifacts is:

```text
AP_q@K = sum(P_q(k) * r_q(k), k=1..K) / min(number_of_positives_q, K)
Recall_q@K = hits_in_top_K / number_of_positives_q
mAP@K = mean(AP_q@K over queries)
Macro-mAP@K = mean(mean(AP_q@K within instance) over instances)
```

`MetricProfile.THESIS` reproduces this implementation. `MetricProfile.ALL_RELEVANT` changes only
the AP denominator to the number of all positives, matching the stricter equation written in the
thesis text. Both profiles are tested so the distinction cannot be introduced accidentally.

Scores are ranked in descending order with a stable sort. Candidates tied on score retain their
input order, making repeated evaluations deterministic.
