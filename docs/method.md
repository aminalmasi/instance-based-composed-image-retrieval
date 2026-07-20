# Method

## First-stage retrieval

The first stage uses SigLIP2 SO400M Patch16 384 and searches each query's own ICIR candidate pool.
Database and query image embeddings are initially L2-normalized. For instance `g`, the visual
center is computed from every database image whose instance is not `g`. The center is subtracted
from the query and candidate image vectors without a second normalization step.

Short modification text is contextualized by attaching it to generic subject terms and averaging
the resulting text embeddings. The contextual corpus mean is subtracted from this query. Candidate
image embeddings are then compared separately with the centered query image and contextualized text
query. Negative similarities are clipped to zero and the two signals are multiplied.

## Route A: supervised instance adaptation

Route A trains a linear projection over frozen SigLIP1 image embeddings using ILIAS instance
supervision. Training supports pairwise sigmoid, local InfoNCE, and multi-negative margin losses.
The released primary checkpoint uses local InfoNCE and hard negatives. At inference time, the probe
image similarity and frozen SigLIP1 contextualized text similarity are multiplied with both
first-stage SigLIP2 scores.

## Route B: descriptor-guided multimodal reranking

Route B generates an identity descriptor from the query image while using the modification only to
locate the intended subject. The descriptor must not reveal the requested target state. RZEnEmbed-7B
uses the descriptor as an attention guide when embedding each shortlisted candidate. The resulting
candidate representation is compared with the identity-oriented query signal and the modification
signal. These two scores are multiplied with the two original SigLIP2 scores.

Both rerankers operate on a fixed top-100 shortlist. They change ordering metrics but do not recover
positives omitted by the first stage.
