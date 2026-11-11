"""GPU k-fold core following mahal_theta_kfold_basis_b.m."""

from __future__ import annotations

import torch

from helper import _covdiag_batched, _mahalanobis_batched


def mahal_kfold_distances(
    data: torch.Tensor,
    bin_mask: torch.Tensor,
    folds: list[torch.Tensor],
    generator: torch.Generator,
    centres: torch.Tensor,
    basis_exponent: float,
) -> torch.Tensor:
    """Compute held-out Mahalanobis distances across all folds.

    data: input EEG features, shape (trial, feature, time), on CPU or GPU.
    bin_mask: bin membership (0/1), shape (bin, trial), on data.device.
    folds: list of CPU index tensors, each containing one fold's test trial indices.
    generator: random generator on CPU for trial subsampling.
    centres: bin centres in doubled-angle radians, shape (bin,), on data.device.
    basis_exponent: scalar power in (0.5 + 0.5*cos(delta))**basis_exponent.

    returns: raw distances, shape (bin, trial, time), on data.device.
    """
    delta = centres[:, None] - centres[None, :]
    basis = (0.5 + 0.5 * torch.cos(delta)).pow(basis_exponent)
    basis = basis / basis.sum(dim=1, keepdim=True)  # bin (current template) x bin (contributing mean)

    n_trials, _, n_times = data.shape
    n_templates = bin_mask.shape[0]
    distances = torch.full(
        (n_templates, n_trials, n_times), torch.nan,
        dtype=data.dtype, device=data.device,
    )
    all_indices = torch.arange(n_trials)

    for test_fold in folds:
        train_mask = torch.ones(n_trials, dtype=torch.bool)
        train_mask[test_fold] = False
        train_fold = all_indices[train_mask]
        train_indices = train_fold.to(data.device)
        test_indices = test_fold.to(data.device)
        train_data, test_data = data[train_indices], data[test_indices]

        train_bin_mask = bin_mask[:, train_indices]
        bin_counts = train_bin_mask.sum(dim=1).to(torch.long)
        n_selected_trials_per_bin = int(bin_counts.min().item())
        bin_means = []
        for bin_index in range(n_templates):
            bin_trial_indices = torch.where(train_bin_mask[bin_index].cpu().bool())[0]
            selected_trial_indices = bin_trial_indices[
                torch.randperm(bin_trial_indices.numel(), generator=generator)[
                    :n_selected_trials_per_bin
                ]
            ]
            bin_mean = train_data[selected_trial_indices.to(data.device)].mean(dim=0)
            bin_means.append(bin_mean)

        means = torch.stack(bin_means)  # bin x feature x time
        # Circular convolution across orientation bins using basis weights.
        templates = torch.einsum("kj,jft->kft", basis, means)  # bin x feature x time

        # Covariance uses all training trials, including those not sampled.
        covariance = _covdiag_batched(train_data.permute(2, 0, 1))  # time x trial x feature

        distances[:, test_indices, :] = _mahalanobis_batched(templates, test_data, covariance)  # template x test trial x time

    return distances
