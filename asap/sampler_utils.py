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
    cumulative = (cumulative - 0.5 * sorted_weights) / cumulative[-1] # Normalize to [0, 1], using midpoints of weights for interpolation
    return np.interp(percentiles, cumulative, sorted_data)


def extract_emcee(sampler, burn_frac=0.5):
    """Extract a SamplerResult from an emcee EnsembleSampler.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The sampler after run_mcmc() has completed.
    burn_frac : float
        Fraction of steps to discard as burn-in (default 0.5).

    Returns
    -------
    SamplerResult
    """
    # Raw chain: (nsteps, nwalkers, ndim)
    raw_chain = sampler.get_chain()
    raw_log_prob = sampler.get_log_prob()  # (nsteps, nwalkers)

    nsteps = raw_chain.shape[0]
    burn = round(burn_frac * nsteps)

    # Discard burn-in
    chain_post_burn = raw_chain[burn:]
    log_prob_post_burn = raw_log_prob[burn:]

    # Flatten: (n_remaining * nwalkers, ndim)
    n_remaining, nwalkers, ndim = chain_post_burn.shape
    samples_flat = chain_post_burn.reshape(n_remaining * nwalkers, ndim)
    log_prob_flat = log_prob_post_burn.reshape(n_remaining * nwalkers)

    n_samples = len(samples_flat)
    weights = np.ones(n_samples) / n_samples

    # Autocorrelation time (best effort)
    try:
        tau = sampler.get_autocorr_time(tol=0)
    except Exception:
        tau = np.full(ndim, np.nan)

    return SamplerResult(
        samples=samples_flat,
        log_likelihood=log_prob_flat,
        weights=weights,
        sampler_type="emcee",
        evidence=None,
        evidence_err=None,
        metadata={"tau": tau, "nsteps": nsteps, "burn": burn},
        raw_chain=raw_chain,
    )


def extract_dynesty(sampler):
    """Extract a SamplerResult from a dynesty NestedSampler.

    Parameters
    ----------
    sampler : dynesty.NestedSampler or dynesty.DynamicNestedSampler
        The sampler after run_nested() has completed.

    Returns
    -------
    SamplerResult
    """
    res = sampler.results

    # Importance weights from log-weights
    log_wt = res.logwt
    weights = np.exp(log_wt - log_wt.max())
    weights /= weights.sum()

    return SamplerResult(
        samples=res.samples,
        log_likelihood=res.logl,
        weights=weights,
        sampler_type="dynesty",
        evidence=float(res.logz[-1]),
        evidence_err=float(res.logzerr[-1]),
        metadata={"niter": res.niter, "results_obj": res},
        raw_chain=None,
    )


def extract_ultranest(result):
    """Extract a SamplerResult from an UltraNest run result.

    Parameters
    ----------
    result : dict
        The dictionary returned by ReactiveNestedSampler.run().

    Returns
    -------
    SamplerResult
    """
    # UltraNest provides both equally-weighted posterior draws (result['samples'])
    # and importance-weighted nested sampling samples (result['weighted_samples']).
    # We use the importance-weighted samples for correct posteriors; the posterior
    # draws are preserved in metadata for reference.
    ws = result["weighted_samples"]
    samples_weighted = np.array(ws["points"])
    logl_weighted = np.array(ws["logl"])
    w = np.array(ws["weights"], dtype=float)

    wsum = w.sum()
    if (not np.isfinite(wsum)) or (wsum <= 0):
        w = np.ones(len(samples_weighted), dtype=float)
        w /= len(samples_weighted)
    else:
        w /= wsum

    return SamplerResult(
        samples=samples_weighted,
        log_likelihood=logl_weighted,
        weights=w,
        sampler_type="ultranest",
        evidence=float(result["logz"]),
        evidence_err=float(result["logzerr"]),
        metadata={
            "ncall": result.get("ncall", None),
            "niter": result.get("niter", None),
            "insertion_order_MWW_test": result.get("insertion_order_MWW_test", None),
            "posterior_samples": np.array(result["samples"]),
        },
        raw_chain=None,
    )
