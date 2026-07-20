from __future__ import annotations

try:
    import torch
    from torch.nn import functional as F
except ModuleNotFoundError as error:  # pragma: no cover - depends on optional extra
    raise ModuleNotFoundError("install instance-cir[torch] to use training losses") from error


def _scores(
    anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if anchors.ndim != 2 or positives.shape != anchors.shape:
        raise ValueError("anchors and positives must have shape [batch, dim]")
    if negatives.ndim != 3 or negatives.shape[0] != anchors.shape[0]:
        raise ValueError("negatives must have shape [batch, negatives, dim]")
    positive_scores = (anchors * positives).sum(dim=-1)
    negative_scores = torch.einsum("bd,bkd->bk", anchors, negatives)
    return positive_scores, negative_scores


def pairwise_sigmoid(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    negative_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    positive_scores, negative_scores = _scores(anchors, positives, negatives)
    positive_loss = F.binary_cross_entropy_with_logits(
        positive_scores, torch.ones_like(positive_scores)
    )
    negative_loss = F.binary_cross_entropy_with_logits(
        negative_scores, torch.zeros_like(negative_scores), reduction="none"
    )
    if negative_weights is not None:
        negative_loss = negative_loss * negative_weights
    return positive_loss + negative_loss.mean()


def local_infonce(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    positive_scores, negative_scores = _scores(anchors, positives, negatives)
    logits = torch.cat([positive_scores[:, None], negative_scores], dim=1) / temperature
    return F.cross_entropy(
        logits, torch.zeros(len(anchors), dtype=torch.long, device=anchors.device)
    )


def multi_negative_margin(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    *,
    margin: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    positive_scores, negative_scores = _scores(anchors, positives, negatives)
    losses = F.relu(margin - positive_scores[:, None] + negative_scores)
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum(dim=1).mean()
    if reduction == "top_1":
        return losses.max(dim=1).values.mean()
    if reduction == "top_half":
        count = max(1, losses.shape[1] // 2)
        return losses.topk(count, dim=1).values.mean()
    raise ValueError(f"unsupported reduction: {reduction}")
