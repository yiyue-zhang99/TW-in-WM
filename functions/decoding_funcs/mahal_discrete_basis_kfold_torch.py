"""Category-stratified GPU Mahalanobis decoding with balanced training means."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from mahal_kfold_core_torch import mahal_kfold_distances


def make_folds(
    theta: torch.Tensor,
    n_folds: int,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    """Make balanced, category-stratified folds from CPU angles or bin IDs."""
    _, category_ids = torch.unique(theta, return_inverse=True)
    # Draw a new reproducible sklearn seed for each split.
    fold_seed = torch.randint(2**32, (1,), generator=generator).item()
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=fold_seed)
    return [
        torch.from_numpy(test_indices)
        for _, test_indices in splitter.split(
            np.zeros(theta.numel()), category_ids.numpy()
        )
    ]


def mahal_discrete_kfold(
    data: torch.Tensor,
    theta: torch.Tensor,
    n_folds: int,
    template_angles: torch.Tensor,
    generator: torch.Generator,
    basis_exponent: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode trial x feature x time data with raised-cosine basis templates.

    Angles are doubled-angle radians; theta matches equally spaced categories.
    Returned distances are mean-centred, sign-reversed and aligned to each trial.
    """
    theta = torch.remainder(theta.flatten(), 2 * torch.pi)
    folds = make_folds(theta.cpu(), n_folds, generator)
    bin_mask = (theta[None, :] == template_angles[:, None]).to(data.dtype)
    distances = mahal_kfold_distances(
        data, bin_mask, folds, generator, template_angles, basis_exponent
    )

    theta_distance = torch.remainder(
        theta[:, None] - template_angles[None, :],
        2 * torch.pi,
    )
    trial_cos = -(
        torch.cos(theta_distance).T[:, :, None] * distances
    ).mean(dim=0)
    similarity = -(distances - distances.mean(dim=0, keepdim=True))
    # Align by category offsets to avoid rounding at the opposite angle (+/-pi).
    n_bins = template_angles.numel()
    trial_bins = bin_mask.argmax(dim=0)
    relative_bins = torch.arange(n_bins, device=data.device) - n_bins // 2
    # Align using trial angle minus template angle.
    # relative_order = (trial_bins[:, None] - relative_bins[None, :]) % n_bins
    # Align using template angle minus trial angle.
    relative_order = (trial_bins[:, None] + relative_bins[None, :]) % n_bins
    aligned = torch.gather(
        similarity.permute(1, 0, 2), 1,
        relative_order[:, :, None].expand(-1, -1, data.shape[2]),
    )
    return trial_cos, aligned.permute(1, 0, 2)
