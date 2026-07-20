from __future__ import annotations

import gc
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from instance_cir.features import l2_normalize


class RzenEncoder:
    """Adapter for the official RZEnEmbed inference implementation."""

    def __init__(
        self,
        model_root: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attention: str = "sdpa",
        max_image_tokens: int = 768,
        max_length: int = 1400,
    ) -> None:
        try:
            import torch
        except ModuleNotFoundError as error:  # pragma: no cover
            raise ModuleNotFoundError("install instance-cir[models] to run RZEn") from error

        model_root = Path(model_root).expanduser().resolve()
        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))
        try:
            from rzen_embed_inference import RzenEmbed
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                f"rzen_embed_inference.py was not found under {model_root}"
            ) from error
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for practical RZEn inference")
        self.torch = torch
        self.model = RzenEmbed(
            model_path=str(model_root),
            device=device,
            torch_dtype=dtype,
            attn_implementation=attention,
            max_image_tokens=max_image_tokens,
            max_length=max_length,
        )

    def encode_fused_adaptive(
        self,
        image_paths: Sequence[str],
        texts: Sequence[str],
        instruction: str,
        *,
        start_batch_size: int = 8,
        max_batch_size: int = 24,
    ) -> np.ndarray:
        if len(image_paths) != len(texts):
            raise ValueError("RZEn image paths and texts must align")
        chunks = []
        cursor = 0
        batch_size = min(max(1, start_batch_size), max_batch_size, len(image_paths))
        success_streak = 0
        while cursor < len(image_paths):
            current = min(batch_size, len(image_paths) - cursor)
            try:
                with self.torch.inference_mode():
                    values = self.model.get_fused_embeddings(
                        images=list(image_paths[cursor : cursor + current]),
                        texts=list(texts[cursor : cursor + current]),
                        instruction=instruction,
                        batch_size=current,
                        show_progress_bar=False,
                    )
                chunks.append(values.detach().float().cpu().numpy())
                cursor += current
                success_streak += 1
                if success_streak >= 4 and batch_size < max_batch_size:
                    batch_size = min(max_batch_size, batch_size + 2)
                    success_streak = 0
            except RuntimeError as error:
                message = str(error).lower()
                if "out of memory" not in message and "cuda" not in message:
                    raise
                gc.collect()
                self.torch.cuda.empty_cache()
                if current <= 1:
                    raise
                batch_size = max(1, current // 2)
                success_streak = 0
        return l2_normalize(np.concatenate(chunks)).astype(np.float32)
