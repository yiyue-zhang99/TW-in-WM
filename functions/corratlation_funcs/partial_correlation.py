"""Residuals and partial correlations for TW–fixed-window decoding analyses."""
import numpy as np
from scipy import stats


def residual(values, design):
    """Remove the columns in design (including an intercept supplied by the caller)."""
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def partial_correlation(x, y, design):
    """Correlate x and y after removing the same nuisance design from each."""
    return stats.pearsonr(residual(x, design), residual(y, design)).statistic
