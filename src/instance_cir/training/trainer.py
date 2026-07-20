from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import torch
    from torch.optim import AdamW
except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError("install instance-cir[torch] to train a probe") from error

from instance_cir.metrics import MetricProfile, evaluate_rankings
from instance_cir.models.projection import LinearProjectionHead
from instance_cir.ranking import stable_top_k
from instance_cir.training.data import EmbeddingBanks, GroupSampler, load_training_groups
from instance_cir.training.losses import local_infonce, multi_negative_margin, pairwise_sigmoid


def _resolve_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent
    config["supervision_csv"] = (
        str((base / config["supervision_csv"]).resolve())
        if not Path(config["supervision_csv"]).is_absolute()
        else config["supervision_csv"]
    )
    config["embedding_banks"] = {
        name: str((base / value).resolve()) if not Path(value).is_absolute() else value
        for name, value in config["embedding_banks"].items()
    }
    return config


def _batch_indices(length: int, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(length)
    for start in range(0, length, batch_size):
        yield order[start : start + batch_size]


def _task_loss(config: dict, anchors, positives, negatives):
    name = config["loss"]["name"]
    if name == "local_infonce":
        return local_infonce(
            anchors,
            positives,
            negatives,
            temperature=float(config["loss"].get("temperature", 0.05)),
        )
    if name == "pairwise_sigmoid":
        return pairwise_sigmoid(anchors, positives, negatives)
    if name == "multi_negative_margin":
        return multi_negative_margin(
            anchors,
            positives,
            negatives,
            margin=float(config["loss"].get("margin", 0.2)),
            reduction=config["loss"].get("reduction", "top_half"),
        )
    raise ValueError(f"unsupported loss: {name}")


@torch.no_grad()
def _evaluate(model, groups, banks, split: str, device: torch.device, top_k: int = 100):
    relevance = []
    totals = []
    instances = []
    model.eval()
    for group in (item for item in groups if item.split == split):
        query = torch.from_numpy(banks.get(group.query)).to(device)[None]
        positives = list(group.positives)
        negatives = [ref for refs in group.negatives.values() for ref in refs]
        candidate_refs = positives + negatives
        candidates = torch.from_numpy(np.stack([banks.get(ref) for ref in candidate_refs])).to(
            device
        )
        query_projected = model(query)
        candidate_projected = model(candidates)
        scores = (candidate_projected @ query_projected.T).squeeze(1).cpu().numpy()
        order = stable_top_k(scores, top_k)
        flags = np.asarray([index < len(positives) for index in order], dtype=bool)
        relevance.append(flags)
        totals.append(len(positives))
        instances.append(group.instance_id)
    return evaluate_rankings(relevance, totals, instances, k=top_k, profile=MetricProfile.THESIS)


def train_probe(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = _resolve_config(config_path)
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    banks = EmbeddingBanks(config["embedding_banks"])
    groups = load_training_groups(
        config["supervision_csv"], top_positives=int(config.get("top_positives", 3))
    )
    sampler = GroupSampler(
        groups,
        banks,
        split="train",
        k_negatives=int(config["loss"]["k_negatives"]),
        seed=seed,
        negative_sampling=config["loss"].get("negative_sampling", "hardest_first"),
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.get("device", "cuda") == "cuda" else "cpu"
    )
    model = LinearProjectionHead(
        banks.embedding_dim,
        int(config["model"].get("output_dim", banks.embedding_dim)),
        use_layernorm=bool(config["model"].get("use_layernorm", False)),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"].get("weight_decay", 0.0)),
    )
    epochs = int(config["optimizer"]["epochs"])
    batch_size = int(config["optimizer"]["batch_size"])
    clip_norm = config["optimizer"].get("gradient_clip_norm")
    history = []
    best_map = -math.inf

    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)
        losses = []
        rng = np.random.default_rng(seed + epoch)
        for indices in _batch_indices(len(sampler), batch_size, rng):
            samples = [sampler.sample(int(index)) for index in indices]
            anchors = torch.from_numpy(np.stack([sample[0] for sample in samples])).to(device)
            positives = torch.from_numpy(np.stack([sample[1] for sample in samples])).to(device)
            negatives = torch.from_numpy(np.stack([sample[2] for sample in samples])).to(device)
            batch, count, dim = negatives.shape
            optimizer.zero_grad(set_to_none=True)
            projected_anchors = model(anchors)
            projected_positives = model(positives)
            projected_negatives = model(negatives.reshape(batch * count, dim)).reshape(
                batch, count, -1
            )
            loss = _task_loss(config, projected_anchors, projected_positives, projected_negatives)
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation = _evaluate(model, groups, banks, "val", device)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_mAP@100": 100.0 * validation.map_at_k,
            "val_Macro-mAP@100": 100.0 * validation.macro_map_at_k,
            "val_Recall@100": 100.0 * validation.recall_at_k,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        if validation.map_at_k > best_map:
            best_map = validation.map_at_k
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "embedding_dim": banks.embedding_dim,
                    "validation": row,
                },
                output_dir / "best_checkpoint.pt",
            )

    checkpoint = torch.load(
        output_dir / "best_checkpoint.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test = _evaluate(model, groups, banks, "test", device)
    summary = {
        "best_epoch": checkpoint["epoch"],
        "best_validation_mAP@100": 100.0 * best_map,
        "test_mAP@100": 100.0 * test.map_at_k,
        "test_Macro-mAP@100": 100.0 * test.macro_map_at_k,
        "test_Recall@100": 100.0 * test.recall_at_k,
    }
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
