"""Compact PyTorch surface analyzer.

The encoder intentionally uses only hash embeddings, separable dilated
convolutions, and a diagonal affine scan. No runtime teacher or dictionary is
needed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Mapping, Optional

import torch
from torch import nn

from data.schemas.canonical import ATOM_TYPES, BUNSETSU_ROLES, INFLECTIONS, PARTICLE_FUNCTIONS
from .features import CAT_FEATURE_NAMES


@dataclass
class ModelConfig:
    codepoint_buckets: int = 8192
    codepoint_dim: int = 8
    bigram_buckets: int = 8192
    bigram_dim: int = 4
    hidden: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    categorical_dim: int = len(CAT_FEATURE_NAMES)

    @classmethod
    def for_size(cls, size: str) -> "ModelConfig":
        """Return comparable small/primary/large variants."""
        variants = {
            "50k": dict(codepoint_buckets=4096, codepoint_dim=6, bigram_buckets=4096,
                        bigram_dim=3, hidden=48),
            "100k": dict(codepoint_buckets=8192, codepoint_dim=6, bigram_buckets=8192,
                         bigram_dim=3, hidden=56),
            "150k": dict(codepoint_buckets=8192, codepoint_dim=8, bigram_buckets=8192,
                         bigram_dim=4, hidden=64),
            # Keep the largest required variant below the 250k hard ceiling;
            # unlike the primary model it spends capacity in wider hashes.
            "250k": dict(codepoint_buckets=8192, codepoint_dim=16, bigram_buckets=8192,
                         bigram_dim=8, hidden=72),
        }
        if size not in variants:
            raise ValueError("unknown model size: %s" % size)
        return cls(**variants[size])


class SeparableResidualBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(hidden, hidden, 3, padding=dilation,
                                   dilation=dilation, groups=hidden, bias=True)
        self.pointwise = nn.Conv1d(hidden, hidden, 1, bias=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # [batch, time, channels] -> convolution layout -> back.
        residual = x
        y = x.transpose(1, 2)
        y = self.pointwise(self.depthwise(y)).transpose(1, 2)
        y = torch.nn.functional.gelu(y)
        y = self.norm(y)
        y = (residual + y) * mask.unsqueeze(-1)
        return y


class DiagonalAffineScan(nn.Module):
    """Two independent diagonal recurrent scans with learned input projections."""
    def __init__(self, hidden: int):
        super().__init__()
        self.forward_project = nn.Linear(hidden, hidden)
        self.backward_project = nn.Linear(hidden, hidden)
        self.forward_decay = nn.Parameter(torch.full((hidden,), 1.1))
        self.backward_decay = nn.Parameter(torch.full((hidden,), 1.1))

    def _scan(self, x: torch.Tensor, mask: torch.Tensor, project: nn.Linear,
              decay_parameter: torch.Tensor, reverse: bool) -> torch.Tensor:
        batch, length, hidden = x.shape
        result = x.new_zeros((batch, length, hidden))
        state = x.new_zeros((batch, hidden))
        indices = range(length - 1, -1, -1) if reverse else range(length)
        decay = torch.sigmoid(decay_parameter)
        for i in indices:
            active = mask[:, i].unsqueeze(-1)
            state = active * (decay * state + project(x[:, i]))
            result[:, i] = state
        return result

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        forward = self._scan(x, mask, self.forward_project, self.forward_decay, False)
        backward = self._scan(x, mask, self.backward_project, self.backward_decay, True)
        return (forward + backward) * mask.unsqueeze(-1)

    def export_forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Parallel equivalent used for fixed-length ONNX export.

        The training path keeps the simple recurrence. The old exporter
        unrolled it into thousands of tiny ONNX nodes, which is hostile to
        browser WebGPU. A triangular decay matrix expresses the same scan as
        two batched matrix multiplies, allowing the provider to run larger,
        parallel kernels. Padded positions are explicitly removed because the
        sequential reference resets state at every masked position.
        """
        _, length, hidden = x.shape
        positions = torch.arange(length, device=x.device)
        distance = positions[:, None] - positions[None, :]
        # Materialize the triangular masks as constants. ONNX Runtime WebGPU
        # has no Trilu kernel; exporting torch.tril/triu leaves unsupported
        # operators in the graph and can abort session execution.
        lower = torch.tensor([[1.0 if column <= row else 0.0 for column in range(length)]
                              for row in range(length)], dtype=x.dtype, device=x.device)
        upper = torch.tensor([[1.0 if column >= row else 0.0 for column in range(length)]
                              for row in range(length)], dtype=x.dtype, device=x.device)

        def parallel(project: nn.Linear, decay_parameter: torch.Tensor,
                     distances: torch.Tensor, triangle: torch.Tensor) -> torch.Tensor:
            source = project(x) * mask.unsqueeze(-1)
            decay = torch.sigmoid(decay_parameter)
            # Express powers with Log/Exp rather than ONNX Pow. Pow's
            # generated WebGPU shader is invalid on some browser/provider
            # combinations, while Log and Exp have stable kernels here.
            powers = torch.exp(torch.log(decay).view(1, 1, hidden) *
                               distances.clamp_min(0).unsqueeze(-1))
            weights = (powers * triangle.unsqueeze(-1)).permute(2, 0, 1)
            source = source.permute(0, 2, 1).reshape(-1, length, 1)
            # Multiplying by a batch-shaped marker gives ONNX a dynamic batch
            # dimension; expand(-1, ...) would preserve the traced batch of 1.
            batch_marker = mask[:, :1].to(weights.dtype).reshape(-1, 1, 1, 1)
            weights = (weights.unsqueeze(0) * batch_marker).reshape(-1, length, length)
            result = torch.bmm(weights, source).reshape(-1, hidden, length).permute(0, 2, 1)
            return result * mask.unsqueeze(-1)

        return parallel(self.forward_project, self.forward_decay, distance, lower) + \
            parallel(self.backward_project, self.backward_decay, -distance, upper)


class JpLexer(nn.Module):
    """Multihead model operating on hashed Unicode feature tensors."""
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.codepoint_embedding = nn.Embedding(c.codepoint_buckets, c.codepoint_dim)
        self.bigram_embedding = nn.Embedding(c.bigram_buckets, c.bigram_dim)
        input_dim = c.codepoint_dim + c.bigram_dim + c.categorical_dim
        self.input_projection = nn.Linear(input_dim, c.hidden)
        self.convolution = nn.ModuleList(SeparableResidualBlock(c.hidden, d) for d in c.dilations)
        self.scan = DiagonalAffineScan(c.hidden)
        self.boundary_head = nn.Linear(c.hidden, 5)
        self.atom_head = nn.Linear(c.hidden, len(ATOM_TYPES))
        self.function_head = nn.Linear(c.hidden, len(PARTICLE_FUNCTIONS))
        self.inflection_head = nn.Linear(c.hidden, len(INFLECTIONS))
        self.role_head = nn.Linear(c.hidden, len(BUNSETSU_ROLES))

    def encode(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        cp = self.codepoint_embedding(inputs["codepoints"])
        bg = self.bigram_embedding(inputs["bigrams"])
        x = torch.cat((cp, bg, inputs["categorical"]), dim=-1)
        mask = inputs["mask"]
        x = self.input_projection(x) * mask.unsqueeze(-1)
        for block in self.convolution:
            x = block(x, mask)
        return self.scan(x, mask)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self.encode(inputs)
        return {"boundaries": self.boundary_head(h), "atom": self.atom_head(h),
                "function": self.function_head(h), "inflection": self.inflection_head(h),
                "role": self.role_head(h)}

    def forward_export(self, inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with a provider-friendly parallel scan."""
        cp = self.codepoint_embedding(inputs["codepoints"])
        bg = self.bigram_embedding(inputs["bigrams"])
        mask = inputs["mask"]
        x = torch.cat((cp, bg, inputs["categorical"]), dim=-1)
        x = self.input_projection(x) * mask.unsqueeze(-1)
        for block in self.convolution:
            x = block(x, mask)
        h = self.scan.export_forward(x, mask)
        return {"boundaries": self.boundary_head(h), "atom": self.atom_head(h),
                "function": self.function_head(h), "inflection": self.inflection_head(h),
                "role": self.role_head(h)}

    def parameter_report(self) -> Dict[str, object]:
        groups = {
            "hash_features": ("codepoint_embedding", "bigram_embedding"),
            "input_projection": ("input_projection",),
            "encoder": ("convolution", "scan"),
            "heads": ("boundary_head", "atom_head", "function_head", "inflection_head", "role_head"),
        }
        by_subsystem = {}
        for group, prefixes in groups.items():
            by_subsystem[group] = sum(p.numel() for name, p in self.named_parameters()
                                      if any(name == x or name.startswith(x + ".") for x in prefixes))
        total = sum(p.numel() for p in self.parameters())
        return {"total_parameters": total, "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "by_subsystem": by_subsystem, "fp32_bytes": total * 4,
                "fp16_bytes": total * 2, "int8_bytes": total,
                "config": self.config.__dict__.copy()}

    @classmethod
    def from_size(cls, size: str) -> "JpLexer":
        return cls(ModelConfig.for_size(size))
