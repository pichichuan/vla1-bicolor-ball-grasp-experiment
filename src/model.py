"""A deliberately small vision-language-action policy for an 8 GB laptop GPU."""

from __future__ import annotations

import torch
from torch import nn


class MiniVLA(nn.Module):
    def __init__(self, vocabulary_size: int, state_dim: int, action_dim: int):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
        )
        self.embedding = nn.Embedding(vocabulary_size, 32, padding_idx=0)
        self.language = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.proprioception = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(128 + 64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        vision_features = self.vision(images)
        token_features = self.embedding(token_ids)
        mask = (token_ids != 0).unsqueeze(-1)
        summed = (token_features * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        language_features = self.language(summed / counts)
        state_features = self.proprioception(states)
        return self.action_head(
            torch.cat((vision_features, language_features, state_features), dim=1)
        )


class ColorAttentionVLA(nn.Module):
    """Use language as a query over spatial visual features."""

    def __init__(self, vocabulary_size: int, state_dim: int, action_dim: int):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.embedding = nn.Embedding(vocabulary_size, 32, padding_idx=0)
        self.language_query = nn.Sequential(nn.Linear(32, 64), nn.Tanh())
        self.proprioception = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(64 + 64 + 2 + 64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, images, token_ids, states):
        feature_map = self.vision(images)
        token_features = self.embedding(token_ids)
        mask = (token_ids != 0).unsqueeze(-1)
        language = (token_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        query = self.language_query(language)

        logits = torch.einsum("bchw,bc->bhw", feature_map, query) / 8.0
        attention = logits.flatten(1).softmax(dim=1).view_as(logits)
        attended = (feature_map * attention.unsqueeze(1)).sum(dim=(2, 3))
        global_visual = feature_map.mean(dim=(2, 3))

        height, width = logits.shape[-2:]
        y_coords = torch.linspace(-1.0, 1.0, height, device=images.device)
        x_coords = torch.linspace(-1.0, 1.0, width, device=images.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        location = torch.stack(
            (
                (attention * xx).sum(dim=(1, 2)),
                (attention * yy).sum(dim=(1, 2)),
            ),
            dim=1,
        )
        state_features = self.proprioception(states)
        return self.action_head(
            torch.cat(
                (attended, global_visual, location, query, state_features), dim=1
            )
        )


class ColorGroundingModel(nn.Module):
    """Predict a language-conditioned spatial heatmap for the referred object."""

    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.embedding = nn.Embedding(vocabulary_size, 32, padding_idx=0)
        self.language_query = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64)
        )

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        feature_map = self.vision(images)
        token_features = self.embedding(token_ids)
        mask = (token_ids != 0).unsqueeze(-1)
        sentence = (token_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        query = self.language_query(sentence)
        return torch.einsum("bchw,bc->bhw", feature_map, query) / 8.0
