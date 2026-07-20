from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as error:  # pragma: no cover - depends on optional extra
    raise ModuleNotFoundError("install instance-cir[torch] to use projection heads") from error


class LinearProjectionHead(nn.Module):
    """Frozen-backbone linear probe with optional LayerNorm and normalized output."""

    def __init__(self, input_dim: int, output_dim: int, *, use_layernorm: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.norm = nn.LayerNorm(output_dim) if use_layernorm else nn.Identity()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.norm(self.linear(embeddings)), dim=-1)
