from __future__ import annotations

import gc
from collections.abc import Sequence

import numpy as np

try:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError("install instance-cir[models] to extract SigLIP features") from error

from instance_cir.features import l2_normalize


class SiglipEncoder:
    """Frozen SigLIP/SigLIP2 feature extractor with adaptive OOM backoff."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str = "main",
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        self.device = torch.device(
            device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model_dtype = dtype_map[dtype] if self.device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = (
            AutoModel.from_pretrained(model_id, revision=revision, torch_dtype=model_dtype)
            .to(self.device)
            .eval()
        )

    @staticmethod
    def _tensor(output, kind: str):
        if isinstance(output, torch.Tensor):
            return output
        preferred = [f"{kind}_embeds", "pooler_output", "last_hidden_state"]
        for name in preferred:
            value = getattr(output, name, None)
            if isinstance(value, torch.Tensor):
                return value[:, 0] if name == "last_hidden_state" else value
        raise RuntimeError(f"cannot extract {kind} embeddings from model output")

    def _clear(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 128,
        max_length: int = 64,
    ) -> np.ndarray:
        output = []
        cursor = 0
        current_batch = min(batch_size, max(1, len(texts)))
        while cursor < len(texts):
            chunk = list(texts[cursor : cursor + current_batch])
            try:
                inputs = self.processor(
                    text=chunk,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                features = self._tensor(self.model.get_text_features(**inputs), "text")
                output.append(features.float().cpu().numpy())
                cursor += len(chunk)
            except torch.OutOfMemoryError:
                self._clear()
                if current_batch == 1:
                    raise
                current_batch = max(1, current_batch // 2)
        return l2_normalize(np.concatenate(output)).astype(np.float16)

    @torch.inference_mode()
    def encode_images(
        self,
        image_paths: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> np.ndarray:
        output = []
        cursor = 0
        current_batch = min(batch_size, max(1, len(image_paths)))
        while cursor < len(image_paths):
            paths = image_paths[cursor : cursor + current_batch]
            images = []
            try:
                for path in paths:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                features = self._tensor(self.model.get_image_features(**inputs), "image")
                output.append(features.float().cpu().numpy())
                cursor += len(paths)
            except torch.OutOfMemoryError:
                self._clear()
                if current_batch == 1:
                    raise
                current_batch = max(1, current_batch // 2)
            finally:
                for image in images:
                    image.close()
        return l2_normalize(np.concatenate(output)).astype(np.float16)
