"""Repeated basis decoding for discrete categories or shifted orientation bins."""

from __future__ import annotations

import numpy as np
import torch

from mahal_discrete_basis_kfold_torch import mahal_discrete_kfold
from mahal_theta_basis_kfold_torch import mahal_continuous_basis_kfold


@torch.inference_mode()
def decode_discrete_basis_repetitions(
    features: np.ndarray,
    theta: np.ndarray,
    *,
    n_folds: int = 8,
    repetitions: int = 100,
    basis_exponent: float = 5.0,
    device: str,
    dtype: str = "float32",
    seed: int = 1,
):
    """Repeat decoding using the existing discrete orientation categories."""
    return _decode_basis_repetitions(
        features, theta, n_bins=None, n_orientation_spaces=1,
        n_folds=n_folds, repetitions=repetitions, basis_exponent=basis_exponent,
        device=device, dtype=dtype, seed=seed,
    )


@torch.inference_mode()
def decode_continuous_basis_repetitions(
    features: np.ndarray,
    theta: np.ndarray,
    *,
    n_folds: int = 8,
    n_bins: int = 16,
    n_orientation_spaces: int = 8,
    basis_exponent: float = 15.0,
    repetitions: int = 100,
    device: str,
    dtype: str = "float32",
    seed: int = 1,
):
    """Repeat decoding within each shifted bin space, then average the spaces."""
    return _decode_basis_repetitions(
        features, theta, n_bins=n_bins, n_orientation_spaces=n_orientation_spaces,
        n_folds=n_folds, repetitions=repetitions, basis_exponent=basis_exponent,
        device=device, dtype=dtype, seed=seed,
    )


def _decode_basis_repetitions(
    features: np.ndarray,
    theta: np.ndarray,
    *,
    n_bins: int | None,
    n_orientation_spaces: int,
    n_folds: int,
    repetitions: int,
    basis_exponent: float,
    device: str,
    dtype: str,
    seed: int,
):
    """n_bins: None for exact categories; otherwise the number of continuous-angle bins.

    returns: mean distances (bin, time), trial scores (trial, time).
    """
    torch_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
    target = torch.device(device)
    data_t = torch.as_tensor(features, dtype=torch_dtype, device=target)
    theta_t = torch.remainder(
        torch.as_tensor(theta, dtype=torch_dtype, device=target).flatten(),
        2 * torch.pi,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)

    if n_bins is None:
        bin_centers = torch.unique(theta_t)
    else:
        bin_centers = torch.arange(
            n_bins, dtype=torch_dtype, device=target
        ) * (2 * torch.pi / n_bins)
    n_templates = bin_centers.numel()
    trial_sum = torch.zeros(
        (data_t.shape[0], data_t.shape[2]), dtype=torch_dtype, device=target
    )
    distance_sum = torch.zeros(
        (n_templates, data_t.shape[2]), dtype=torch_dtype, device=target
    )
    total_runs = repetitions * n_orientation_spaces
    for space_index in range(n_orientation_spaces):
        if n_bins is None:
            centres = bin_centers
        else:
            shift = space_index * (2 * torch.pi / n_bins / n_orientation_spaces)
            centres = torch.remainder(bin_centers + shift, 2 * torch.pi)
        for repetition in range(repetitions):
            if n_bins is None:
                trial_cos, distances = mahal_discrete_kfold(
                    data_t, theta_t, n_folds, centres, generator, basis_exponent,
                )
            else:
                trial_cos, distances = mahal_continuous_basis_kfold(
                    data_t, theta_t, n_folds, centres, basis_exponent, generator,
                )
            trial_sum += trial_cos
            distance_sum += distances.mean(dim=1)
            if (repetition + 1) % 10 == 0 or repetition + 1 == repetitions:
                print(
                    f"  space {space_index + 1}/{n_orientation_spaces}, "
                    f"repetition {repetition + 1}/{repetitions}", flush=True,
                )

    trialwise = trial_sum / total_runs
    return (
        (distance_sum / total_runs).cpu().numpy(),
        trialwise.cpu().numpy(),
    )
