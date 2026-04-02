"""
sampler_utils.py
================
Normalizes output from emcee, dynesty, and UltraNest into a common format
so that save_results() can be sampler-agnostic.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SamplerResult:
    """Common container for sampler output.

    Attributes
    ----------
    samples : np.ndarray
        Flat posterior samples, shape (n_samples, ndim).
    log_likelihood : np.ndarray
        Log-likelihood per sample, shape (n_samples,).
    weights : np.ndarray
        Importance weights per sample, normalized to sum=1.
        Uniform (1/n_samples) for emcee and UltraNest.
    sampler_type : str
        One of "emcee", "dynesty", "ultranest".
    evidence : float or None
        ln(Z) marginal likelihood. None for emcee.
    evidence_err : float or None
        Uncertainty on ln(Z). None for emcee.
    metadata : dict
        Sampler-specific extras (tau for emcee, niter for nested, etc.).
    raw_chain : np.ndarray or None
        For emcee: the un-flattened, pre-burn-in chain (nsteps, nwalkers, ndim)
        for walker trace plots. None for nested samplers.
    """
    samples: np.ndarray
    log_likelihood: np.ndarray
    weights: np.ndarray
    sampler_type: str
    evidence: Optional[float] = None
    evidence_err: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    raw_chain: Optional[np.ndarray] = None


def weighted_percentile(data, weights, percentiles):
    """Compute weighted percentiles.

    Parameters
    ----------
    data : np.ndarray
        1D array of values.
    weights : np.ndarray
        1D array of weights (same length as data). Need not be normalized.
    percentiles : array-like
        Percentiles to compute, in [0, 100].

    Returns
    -------
    np.ndarray
        Weighted percentile values, same length as percentiles.
    """
    percentiles = np.asarray(percentiles) / 100.0
    sorted_idx = np.argsort(data)
    sorted_data = data[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumulative = np.cumsum(sorted_weights)
    cumulative = (cumulative - 0.5 * sorted_weights) / cumulative[-1]
    return np.interp(percentiles, cumulative, sorted_data)
