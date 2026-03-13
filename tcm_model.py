"""
tcm_model.py
============
Neural network architectures for the CS229 TCM herb-prediction project.

Hierarchy of models (ablation study):
  1. TANBaseline      – linear map from symptom features to concepts/herbs
  2. PlainMLP         – deep MLP, no concept bottleneck
  3. ConceptBM        – Concept Bottleneck Model: encoder → concepts → herbs
  4. ConceptBM_CNN    – CBM + 1D CNN herb compatibility module (main model)

All models share the same HerbHead / ConceptHead interface so that the
training loop can call them uniformly.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    """
    MLP encoder: input_dim -> hidden_dim (with LayerNorm + Dropout).
    Two hidden layers for the full model; one for lightweight variants.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        n_layers: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_d = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_d = hidden_dim
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConceptHead(nn.Module):
    """
    Linear head: shared_repr -> concept logits (14 binary outputs).
    Uses sigmoid at inference; BCEWithLogitsLoss during training.
    """

    def __init__(self, hidden_dim: int, n_concepts: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, n_concepts)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)          # raw logits


class HerbHeadBaseline(nn.Module):
    """
    Step 1 – simple multi-label baseline.
    Linear: shared_repr -> herb logits (H outputs).
    """

    def __init__(self, hidden_dim: int, n_herbs: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, n_herbs)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)          # raw logits


class HerbCNNCompatibility(nn.Module):
    """
    Step 3 – CNN compatibility module.

    Takes the baseline herb probability vector (already sigmoided, shape B×H)
    as a 1D sequence, applies 1-2 conv layers over it to capture local
    compatibility patterns among neighbouring herbs (sorted by frequency),
    then concatenates the CNN output back with the encoder representation
    before a final herb linear layer.

    Architecture:
        herb_probs  (B, H)
          → unsqueeze → (B, 1, H)
          → Conv1d(1, cnn_channels, kernel_size) + ReLU
          → Conv1d(cnn_channels, cnn_channels, kernel_size) + ReLU
          → AdaptiveAvgPool1d(pool_out)
          → flatten → (B, cnn_channels * pool_out)

        cat([encoder_repr (B, hidden_dim), cnn_feat (B, cnn_dim)])
          → Linear → herb_logits (B, H)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_herbs: int,
        cnn_channels: int = 32,
        kernel_size: int = 5,
        pool_out: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_herbs = n_herbs

        # Two 1-D conv layers
        self.conv1 = nn.Conv1d(1, cnn_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool = nn.AdaptiveAvgPool1d(pool_out)
        self.drop = nn.Dropout(dropout)

        cnn_out_dim = cnn_channels * pool_out
        self.fc = nn.Linear(hidden_dim + cnn_out_dim, n_herbs)

    def forward(
        self,
        encoder_repr: torch.Tensor,   # (B, hidden_dim)
        herb_probs: torch.Tensor,     # (B, H)  — sigmoided baseline output
    ) -> torch.Tensor:
        # CNN branch
        x = herb_probs.unsqueeze(1)                          # (B, 1, H)
        x = F.relu(self.conv1(x))                            # (B, C, H)
        x = F.relu(self.conv2(x))                            # (B, C, H)
        x = self.pool(x)                                     # (B, C, pool_out)
        x = self.drop(x.flatten(1))                          # (B, C*pool_out)

        combined = torch.cat([encoder_repr, x], dim=-1)      # (B, hidden+cnn)
        return self.fc(combined)                             # raw logits (B, H)


# ---------------------------------------------------------------------------
# Full model wrappers
# ---------------------------------------------------------------------------

class TANBaseline(nn.Module):
    """
    Ablation 1 — simple linear baseline (no hidden layers).
    Directly maps raw features → concept logits and herb logits.
    Approximates a TCM Attention Network baseline with no learned representation.
    """

    def __init__(self, input_dim: int, n_concepts: int, n_herbs: int):
        super().__init__()
        self.concept_head = nn.Linear(input_dim, n_concepts)
        self.herb_head = nn.Linear(input_dim, n_herbs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        concept_logits = self.concept_head(x)
        herb_logits = self.herb_head(x)
        return concept_logits, herb_logits, None   # no concept probs used


class PlainMLP(nn.Module):
    """
    Ablation 2 — deep MLP with no concept bottleneck.
    Encoder shared across concept head and herb head, but concepts are NOT
    used as an intermediate representation for herb prediction.
    """

    def __init__(
        self,
        input_dim: int,
        n_concepts: int,
        n_herbs: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, hidden_dim, dropout, n_layers=2)
        self.concept_head = ConceptHead(hidden_dim, n_concepts)
        self.herb_head = HerbHeadBaseline(hidden_dim, n_herbs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        h = self.encoder(x)
        concept_logits = self.concept_head(h)
        herb_logits = self.herb_head(h)
        return concept_logits, herb_logits, None


class ConceptBM(nn.Module):
    """
    Ablation 3 — Concept Bottleneck Model.
    Encoder → concept predictions → (concept preds + encoder repr) → herb logits.
    Concept supervision is applied explicitly at the concept head.
    """

    def __init__(
        self,
        input_dim: int,
        n_concepts: int,
        n_herbs: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, hidden_dim, dropout, n_layers=2)
        self.concept_head = ConceptHead(hidden_dim, n_concepts)
        # Herb head receives encoder repr concatenated with concept probabilities
        self.herb_fc = nn.Linear(hidden_dim + n_concepts, n_herbs)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        h = self.encoder(x)
        concept_logits = self.concept_head(h)
        concept_probs = torch.sigmoid(concept_logits)
        combined = torch.cat([h, concept_probs], dim=-1)
        herb_logits = self.herb_fc(self.drop(combined))
        return concept_logits, herb_logits, None


class ConceptBM_CNN(nn.Module):
    """
    Main model — CBM + CNN herb compatibility module.

    Forward pass:
      x -> encoder -> concept_logits
                   -> baseline_herb_logits  (Step 1)
      baseline_herb_probs (sigmoid) -> CNN_module -> refined_herb_logits  (Step 3)

    The CNN takes the sorted herb probability vector as a 1-D sequence and
    applies local convolutional filters to capture compatibility patterns
    among neighbouring herbs (ordered by training frequency).
    """

    def __init__(
        self,
        input_dim: int,
        n_concepts: int,
        n_herbs: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        cnn_channels: int = 32,
        cnn_kernel: int = 5,
        cnn_pool_out: int = 16,
    ):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, hidden_dim, dropout, n_layers=2)
        self.concept_head = ConceptHead(hidden_dim, n_concepts)

        # Step 1 — baseline herb head (from encoder repr only)
        self.herb_baseline = HerbHeadBaseline(hidden_dim, n_herbs)

        # Step 3 — CNN compatibility module
        self.herb_cnn = HerbCNNCompatibility(
            hidden_dim=hidden_dim,
            n_herbs=n_herbs,
            cnn_channels=cnn_channels,
            kernel_size=cnn_kernel,
            pool_out=cnn_pool_out,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_cnn: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (concept_logits, herb_logits, baseline_herb_logits).

        herb_logits     : output of CNN compatibility module (if use_cnn=True)
                          else falls back to baseline herb logits
        """
        h = self.encoder(x)
        concept_logits = self.concept_head(h)
        baseline_herb_logits = self.herb_baseline(h)

        if use_cnn:
            baseline_probs = torch.sigmoid(baseline_herb_logits.detach())
            herb_logits = self.herb_cnn(h, baseline_probs)
        else:
            herb_logits = baseline_herb_logits

        return concept_logits, herb_logits, baseline_herb_logits


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def tcm_loss(
    concept_logits: torch.Tensor,
    herb_logits: torch.Tensor,
    y_concept: torch.Tensor,
    y_herb: torch.Tensor,
    concept_weight: float = 1.0,
    herb_weight: float = 1.0,
    baseline_herb_logits: torch.Tensor | None = None,
    baseline_weight: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined BCE loss for concept head + herb head.
    Optionally adds an auxiliary loss on the baseline herb head (for ConceptBM_CNN).
    """
    loss_concept = F.binary_cross_entropy_with_logits(concept_logits, y_concept, reduction="mean")
    loss_herb = F.binary_cross_entropy_with_logits(herb_logits, y_herb, reduction="mean")

    total = concept_weight * loss_concept + herb_weight * loss_herb

    components = {"concept": loss_concept.item(), "herb": loss_herb.item()}

    if baseline_herb_logits is not None:
        loss_baseline = F.binary_cross_entropy_with_logits(
            baseline_herb_logits, y_herb, reduction="mean"
        )
        total = total + baseline_weight * loss_baseline
        components["herb_baseline"] = loss_baseline.item()

    return total, components
