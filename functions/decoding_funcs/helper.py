"""Covariance and Mahalanobis distance helpers for PyTorch decoders."""

from __future__ import annotations

import logging

import torch


logger = logging.getLogger(__name__)


def _covdiag_batched(x: torch.Tensor) -> torch.Tensor:
    """Ledoit-Wolf diagonal-target covariance for time x trial x feature."""
    n_trials, n_features = x.shape[1:]
    x = x - x.mean(dim=1, keepdim=True)
    # Compute the feature covariance matrix at each time point, normalized by n_trials.
    sample = torch.bmm(x.transpose(1, 2), x) / n_trials
    prior = torch.diag_embed(torch.diagonal(sample, dim1=-2, dim2=-1))
    d = (sample - prior).square().sum(dim=(-2, -1)) / n_features
    y = x.square()
    yty_sum = torch.bmm(y.transpose(1, 2), y).sum(dim=(-2, -1))
    r2 = (
        yty_sum / (n_features * n_trials**2)
        - sample.square().sum(dim=(-2, -1)) / (n_features * n_trials)
    )
    shrinkage = torch.where(
        d > 0, (r2 / d).clamp(0, 1), torch.ones_like(d)
    )
    return (
        shrinkage[:, None, None] * prior
        + (1 - shrinkage[:, None, None]) * sample
    )


def _cholesky_batched(covariance: torch.Tensor) -> torch.Tensor:
    """Factor time x feature x feature covariance; jitter only failed times once."""
    chol, info = torch.linalg.cholesky_ex(covariance)
    failed = info != 0
    if torch.any(failed):
        failed_covariance = covariance[failed]
        eps = torch.finfo(covariance.dtype).eps
        # Scale by mean variance.
        scale = torch.diagonal(failed_covariance, dim1=-2, dim2=-1).mean(dim=-1)
        jitter = (100 * eps * scale.clamp_min(eps))[:, None, None]
        logger.warning(
            "Cholesky failed; retrying with diagonal jitter at batch time indices "
            "%s (zero-based), device=%s, dtype=%s, jitter range=[%.6g, %.6g].",
            torch.nonzero(failed, as_tuple=True)[0].tolist(),
            covariance.device,
            covariance.dtype,
            jitter.min().item(),
            jitter.max().item(),
        )
        corrected = failed_covariance + jitter * torch.eye(
            covariance.shape[-1], dtype=covariance.dtype, device=covariance.device
        )
        # Preserve successful factors; let a failed retry raise an error.
        chol = chol.clone()
        chol[failed] = torch.linalg.cholesky(corrected)
    return chol


def _mahalanobis_batched(
    templates: torch.Tensor,
    tests: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    """Return template x test x time distances without matrix inversion."""
    # Inputs become time x template/test x feature.
    templates_t = templates.permute(2, 0, 1)
    tests_t = tests.permute(2, 0, 1)
    difference = templates_t[:, :, None, :] - tests_t[:, None, :, :]
    time_count, template_count, test_count, feature_count = difference.shape
    rhs = difference.reshape(time_count, template_count * test_count, feature_count)

    chol = _cholesky_batched(covariance)

    whitened = torch.linalg.solve_triangular(
        chol, rhs.transpose(1, 2), upper=False
    )
    distances = whitened.square().sum(dim=1).sqrt()
    return distances.reshape(
        time_count, template_count, test_count
    ).permute(1, 2, 0)
