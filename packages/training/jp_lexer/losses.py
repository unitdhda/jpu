"""Masked multitask losses for the Phase 2 heads."""
from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    denominator = mask.expand_as(values).sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def multitask_loss(outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor],
                   weights: Mapping[str, float] | None = None) -> Dict[str, torch.Tensor]:
    """Return component losses and weighted ``total``; padding never contributes."""
    weights = dict(weights or {"boundaries": 1.0, "atom": .7, "inflection": .6,
                               "function": .4, "role": .4})
    valid = targets["mask"]
    parts = {
        "boundaries": _masked_mean(F.binary_cross_entropy_with_logits(
            outputs["boundaries"], targets["boundaries"], reduction="none"), valid),
        "atom": _masked_mean(F.cross_entropy(outputs["atom"].transpose(1, 2), targets["atom"], reduction="none"), valid),
        "inflection": _masked_mean(F.binary_cross_entropy_with_logits(
            outputs["inflection"], targets["inflection"], reduction="none"),
            valid & targets["inflection_mask"]),
        "function": _masked_mean(F.cross_entropy(outputs["function"].transpose(1, 2), targets["function"], reduction="none"),
                                  valid & targets["function_mask"]),
        "role": _masked_mean(F.cross_entropy(outputs["role"].transpose(1, 2), targets["role"], reduction="none"), valid),
    }
    parts["total"] = sum(weights[name] * parts[name] for name in parts)
    return parts
