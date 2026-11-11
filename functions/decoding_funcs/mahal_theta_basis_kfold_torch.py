"""Continuous-orientation Mahalanobis decoding scored at assigned bin centres.

Basis: (0.5 + 0.5*cos(doubled_angle_delta))**basis_exponent.
"""

from __future__ import annotations

import torch


from mahal_kfold_core_torch import mahal_kfold_distances
from mahal_discrete_basis_kfold_torch import make_folds


def _nearest_orientation_bins(
    theta: torch.Tensor, centres: torch.Tensor
) -> torch.Tensor:
    """Assign doubled orientation angles to their nearest circular centre."""
    delta = torch.remainder(
        theta[:, None] - centres[None, :] + torch.pi, 2 * torch.pi
    ) - torch.pi
    return delta.abs().argmin(dim=1)


def mahal_continuous_basis_kfold(
    data: torch.Tensor,
    theta: torch.Tensor,
    n_folds: int,
    bin_centres: torch.Tensor,
    basis_exponent: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bin doubled-angle radians, then score and align relative to bin centres."""
    theta = torch.remainder(theta.flatten(), 2 * torch.pi)
    n_times = data.shape[2]
    n_bins = bin_centres.numel()
    category = _nearest_orientation_bins(theta, bin_centres)
    folds = make_folds(category.cpu(), n_folds, generator)
    bin_mask = category[None, :] == torch.arange(n_bins, device=data.device)[:, None]
    distances = mahal_kfold_distances(
        data, bin_mask, folds, generator, bin_centres, basis_exponent
    )

    # Score and align using the actual trial angle.
    # theta_distance = torch.remainder(
    #     theta[:, None] - bin_centres[None, :] + torch.pi, 2 * torch.pi
    # ) - torch.pi
    # trial_cos = -(
    #     torch.cos(theta_distance).T[:, :, None] * distances
    # ).mean(dim=0)
    # relative_order = theta_distance.argsort(dim=1)

    # Score using the assigned bin centre.
    theta_binned = bin_centres[category]
    theta_distance = torch.remainder(
        bin_centres[None, :] - theta_binned[:, None] + torch.pi, 2 * torch.pi
    ) - torch.pi
    trial_cos = -(
        torch.cos(theta_distance).T[:, :, None] * distances
    ).mean(dim=0)
    # Align curves by template angle minus the trial's bin centre.
    similarity = -(distances - distances.mean(dim=0, keepdim=True))
    relative_bins = torch.arange(n_bins, device=data.device) - n_bins // 2
    relative_order = (category[:, None] + relative_bins[None, :]) % n_bins
    similarity_by_trial = similarity.permute(1, 0, 2)
    aligned = torch.gather(
        similarity_by_trial,
        1,
        relative_order[:, :, None].expand(-1, -1, n_times),
    )
    return trial_cos, aligned.permute(1, 0, 2)
