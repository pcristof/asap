"""Joint line fitting with SHARED magnetic filling factors.

Every region of the line list is one "line". All lines share the same
magnetic filling factors a_0..a_{nB-1} (a_0 = 1 - sum(a_i>0)); each line has
its OWN local continuum level c0_r and its OWN line-depth factor k_r:

    mix_r(l)   = sum_i a_i * S_{i,r}(l)             S_{i,r}: model, field i
    model_r(l) = c0_r * (1 - k_r * (1 - mix_r(l)))  =  alpha_r + beta_r * mix_r(l)

with alpha_r = c0_r (1 - k_r) and beta_r = c0_r k_r.  For given filling factors
the model is LINEAR in (alpha_r, beta_r), so these two per-line numbers are
integrated out ANALYTICALLY (flat prior on alpha_r, beta_r; Gaussian
likelihood). The MCMC therefore only samples the nB-1 shared filling factors,
which keeps emcee fast and well behaved however many lines are fitted.
The posterior of every line's c0_r and k_r is recovered afterwards by drawing
(alpha_r, beta_r) from their exact Gaussian conditional for each sample of the
filling factors:  c0_r = alpha_r + beta_r,  k_r = beta_r / (alpha_r + beta_r).

Teff, log(g), [M/H], [a/Fe] and RV are FIXED (config.ini or command line), and
so is the broadening unless --fit-vsini / --fit-vmac / --fit-vb are given: those
add shared (all-lines) parameters with flat priors to the MCMC. The basis is
then kept UNBROADENED and each likelihood call broadens the mixed model once per
line (plain interpolated grids only; same operations as gen_spec). The per-field models S_{i,r} are computed once through the
regular SpectralAnalysis.gen_spec with one-hot filling factors and the
continuum adjustment switched off; this relies on the pipeline being linear
in the filling factors (check with --check-linearity).

Usage:
    python -m asap.linefit_shared <star> [options]
"""
import os
import time
import shutil
import argparse
import numpy as np
import emcee


# ---------------------------------------------------------------------------
# Per-step broadening engine (same operations as gen_spec_int_spectra);
# only used when rotation / macroturbulence / Gaussian broadening are fitted
# ---------------------------------------------------------------------------
_C_KMS = 299792.458          # scipy.constants.c * 1e-3, as in analysis_tools


def doppler(v):
    """Relativistic Doppler factor (same formula as analysis_tools.doppler)."""
    return np.sqrt((1.0 + v / _C_KMS) / (1.0 - v / _C_KMS))


class Broadener:
    """Turns a mixed, unbroadened model spectrum (on the model wavelength grid
    of one region) into the model on the observed wavelengths:

        Doppler shift -> broadening -> interpolation -> zero outside the model
        range -> veiling

    exactly as SpectralAnalysis.gen_spec_int_spectra + spectral_analysis_pack.
    broaden_spectra do for one region, with the continuum adjustment off.
    """

    def __init__(self, wmod, basis_m, obs_wvl, veiling, vinstru,
                 smooth_vel=0.0, vmac_mode="g", epsilon=0.6):
        self.wmod = np.asarray(wmod, dtype=float)          # (npix_m,)
        self.basis_m = np.asarray(basis_m, dtype=float)    # (nB, npix_m)
        self.obs_wvl = np.asarray(obs_wvl, dtype=float)    # (npix,)
        self.veiling = None if veiling is None else np.asarray(veiling, float)
        self.vinstru = float(vinstru)
        self.smooth_vel = float(smooth_vel)
        self.vmac_mode = vmac_mode
        self.epsilon = epsilon

    def observe(self, a, rv, vsini, vmac, vb):
        """a: filling factors (nB,). Returns the model on obs_wvl."""
        from asap.c_tools import effects as effects_cy       # lazy
        mix = a @ self.basis_m
        w = self.wmod * doppler(rv)
        totvb = np.sqrt(self.vinstru ** 2 + vb ** 2 + self.smooth_vel ** 2)
        if (totvb == 0.0) and (vsini == 0.0) and (vmac == 0.0):
            spec = mix
        else:
            spec = effects_cy.broaden_spectrum_2_cy(
                w, mix, vinstru=totvb, vsini=vsini, epsilon=self.epsilon,
                vmac=vmac, vmac_mode=self.vmac_mode)
        wlim = w[mix != 0][-1]
        wlow = w[0]
        out = np.interp(self.obs_wvl, w, spec)
        out[self.obs_wvl > wlim] = 0.0
        out[self.obs_wvl < wlow] = 0.0
        if self.veiling is not None:
            out = (out + self.veiling) / (1.0 + self.veiling)
        return out


# ---------------------------------------------------------------------------
# Core: no heavy imports, testable on synthetic data
# ---------------------------------------------------------------------------
class SharedProblem:
    """Marginal log-posterior of the shared filling factors (and, optionally,
    of shared broadening parameters)."""

    def __init__(self, indices, wvls, fluxes, errs, idxs, bases, bs,
                 broadeners=None, fit_broad=(), fixed_broad=None,
                 bounds=None):
        """Per line r: wvls[r], fluxes[r], errs[r] (npix,), idxs[r] (fitted
        pixels), bases[r] (nB, npix). bs: field strengths in kG.

        Optional broadening fit: broadeners[r] is a Broadener for line r,
        fit_broad a subset of ("vsini", "vmac", "vb") that is sampled (shared
        by all lines), fixed_broad a dict with the values of rv, vsini, vmac,
        vb used for everything not fitted, bounds {name: (lo, hi)} flat
        priors. Without broadeners, `bases` (already broadened) is used."""
        self.lines = [int(i) for i in indices]
        self.nl = len(self.lines)
        self.bs = np.asarray(bs, dtype=float)
        self.nB = len(self.bs)
        self.wvls = [np.asarray(w, float) for w in wvls]
        self.idxs = [np.asarray(i) for i in idxs]
        self.bases_full = [np.asarray(b, float) for b in bases]   # (nB, npix)
        self.broadeners = broadeners
        self.fit_broad = tuple(fit_broad)
        if self.fit_broad and broadeners is None:
            raise ValueError("fit_broad needs broadeners")
        self.fixed_broad = dict(rv=0.0, vsini=0.0, vmac=0.0, vb=0.0)
        self.fixed_broad.update(fixed_broad or {})
        self.bounds = dict(vsini=(0.0, 100.0), vmac=(0.0, 50.0),
                           vb=(0.0, 50.0))
        self.bounds.update(bounds or {})
        y, w, B, counts = [], [], [], []
        self.lognorm = np.empty(self.nl)
        for r in range(self.nl):
            ix = self.idxs[r]
            f = np.asarray(fluxes[r], float)[ix]
            e = np.asarray(errs[r], float)[ix]
            y.append(f)
            w.append(1.0 / e ** 2)
            B.append(self.bases_full[r][:, ix])
            counts.append(len(ix))
            self.lognorm[r] = -0.5 * np.sum(np.log(2 * np.pi * e ** 2))
        self.counts = np.array(counts)
        self.starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]])
        self.y = np.concatenate(y)
        self.w = np.concatenate(w)
        self.B = np.hstack(B)                                     # (nB, N)
        self.wy = self.w * self.y
        self.S0 = np.add.reduceat(self.w, self.starts)
        self.Y0 = np.add.reduceat(self.wy, self.starts)
        self.Yy = np.add.reduceat(self.wy * self.y, self.starts)
        self.ndim = self.nB - 1 + len(self.fit_broad)
        self.names = (["a%d" % i for i in range(self.nB)] + ["Bmean"]
                      + list(self.fit_broad))

    # -- parameters ---------------------------------------------------------
    def broad_values(self, theta):
        """Dict rv, vsini, vmac, vb for a parameter vector theta."""
        d = dict(self.fixed_broad)
        for n, v in zip(self.fit_broad, np.atleast_1d(theta)[self.nB - 1:]):
            d[n] = float(v)
        return d

    def _mix(self, a, bp):
        """Concatenated model mix over the fitted pixels."""
        if self.broadeners is None:
            return a @ self.B
        return np.concatenate(
            [self.broadeners[r].observe(a, bp["rv"], bp["vsini"], bp["vmac"],
                                        bp["vb"])[self.idxs[r]]
             for r in range(self.nl)])

    # -- per-line linear solve ----------------------------------------------
    def solve(self, a, bp=None):
        """For filling factors a (nB,) [and broadening dict bp], the per-line
        least-squares (alpha, beta), chi2 at that solution, and the
        normal-matrix pieces."""
        mix = self._mix(a, bp or self.fixed_broad)
        wm = self.w * mix
        S1 = np.add.reduceat(wm, self.starts)
        S2 = np.add.reduceat(wm * mix, self.starts)
        Y1 = np.add.reduceat(self.wy * mix, self.starts)
        det = self.S0 * S2 - S1 * S1
        with np.errstate(all="ignore"):
            alpha = (S2 * self.Y0 - S1 * Y1) / det
            beta = (self.S0 * Y1 - S1 * self.Y0) / det
        chi2 = self.Yy - (alpha * self.Y0 + beta * Y1)
        return alpha, beta, chi2, det, S1, S2

    def fields(self, theta):
        """Filling factors a_0..a_{nB-1} from theta (extra entries ignored)."""
        a = np.empty(self.nB)
        a[1:] = np.asarray(theta)[:self.nB - 1]
        a[0] = 1.0 - a[1:].sum()
        return a

    def _inside(self, theta, a):
        if np.min(a) < 0.0 or np.max(a) > 1.0:
            return False
        for n, v in zip(self.fit_broad, theta[self.nB - 1:]):
            lo, hi = self.bounds[n]
            if not (lo <= v <= hi):
                return False
        return True

    def lnprior(self, theta):
        return 0.0 if self._inside(theta, self.fields(theta)) else -np.inf

    def __call__(self, theta):
        a = self.fields(theta)
        if not self._inside(theta, a):
            return -np.inf
        alpha, beta, chi2, det, S1, S2 = self.solve(a, self.broad_values(theta))
        if not np.all(det > 0.0):
            return -np.inf
        val = np.sum(self.lognorm + np.log(2 * np.pi) - 0.5 * np.log(det)
                     - 0.5 * chi2)
        return val if np.isfinite(val) else -np.inf

    # -- derived -------------------------------------------------------------
    def full_params(self, chain):
        """Rows of chain -> [a_0..a_{nB-1}, Bmean, fitted broadening...]."""
        chain = np.atleast_2d(chain)
        nf = self.nB - 1
        a = np.empty((chain.shape[0], self.nB))
        a[:, 1:] = chain[:, :nf]
        a[:, 0] = 1.0 - chain[:, :nf].sum(axis=1)
        return np.column_stack([a, a @ self.bs, chain[:, nf:]])

    def zero_theta(self, theta):
        """theta with all magnetic fields switched off (a_0 = 1), keeping the
        fitted broadening of theta."""
        t = np.array(theta, float)
        t[:self.nB - 1] = 0.0
        return t

    def line_draws(self, chain, rng):
        """One draw of (c0_r, k_r) for every line, for every row of chain
        (theta vectors). Returns (c0, k), each (nsamp, nl)."""
        n = chain.shape[0]
        c0 = np.empty((n, self.nl))
        k = np.empty((n, self.nl))
        for s in range(n):
            a = self.fields(chain[s])
            alpha, beta, chi2, det, S1, S2 = self.solve(
                a, self.broad_values(chain[s]))
            va = S2 / det                         # covariance = A^-1
            vb = self.S0 / det
            cab = -S1 / det
            z1 = rng.standard_normal(self.nl)
            z2 = rng.standard_normal(self.nl)
            al = alpha + np.sqrt(va) * z1
            be = beta + (cab / np.sqrt(va)) * z1 \
                + np.sqrt(np.maximum(vb - cab ** 2 / va, 0.0)) * z2
            c = al + be
            c0[s] = c
            with np.errstate(all="ignore"):
                k[s] = be / c
        return c0, k

    def init_walkers(self, nwalkers, rng):
        p0 = []
        while len(p0) < nwalkers:
            a = rng.dirichlet(np.ones(self.nB))
            th = list(a[1:])
            for n in self.fit_broad:              # scatter around start value
                v0 = self.fixed_broad[n]
                lo, hi = self.bounds[n]
                th.append(np.clip(v0 + max(0.1 * abs(v0), 0.5)
                                  * rng.standard_normal(), lo, hi))
            th = np.array(th)
            if np.isfinite(self(th)):
                p0.append(th)
        return np.array(p0)

    def model_full(self, a, bp=None):
        """Best-fit (for fields a, broadening dict bp) model over each whole
        region, plus the per-line c0, k and chi2."""
        bp = bp or self.fixed_broad
        alpha, beta, chi2, det, S1, S2 = self.solve(a, bp)
        if self.broadeners is None:
            mixes = [a @ self.bases_full[r] for r in range(self.nl)]
        else:
            mixes = [self.broadeners[r].observe(a, bp["rv"], bp["vsini"],
                                                bp["vmac"], bp["vb"])
                     for r in range(self.nl)]
        models = [alpha[r] + beta[r] * mixes[r] for r in range(self.nl)]
        c0 = alpha + beta
        with np.errstate(all="ignore"):
            k = beta / c0
        return models, c0, k, chi2

_PROB = None
def _lnprob(theta):             # module-level so forked workers can call it
    return _PROB(theta)
def run_mcmc(prob, nwalkers, nsteps, burn_frac, seed, ndraw, progress=False, ncores=1):
    global _PROB
    _PROB = prob
    pool = None
    if ncores > 1:
        print(f'Running with {ncores} cores')
        from multiprocessing import get_context
        pool = get_context("fork").Pool(ncores)
    rng = np.random.default_rng(seed)
    nwalkers = max(int(nwalkers), 2 * prob.ndim + 2)
    p0 = prob.init_walkers(nwalkers, rng)
    # sampler = emcee.EnsembleSampler(nwalkers, prob.ndim, prob)
    sampler = emcee.EnsembleSampler(nwalkers, prob.ndim, _lnprob, pool=pool)
    sampler.random_state = np.random.RandomState(int(seed) % (2 ** 32))
    sampler.run_mcmc(p0, nsteps, progress=progress)
    burn = int(burn_frac * nsteps)
    chain = sampler.get_chain(discard=burn, flat=True)
    lp = sampler.get_log_prob(discard=burn, flat=True)
    allchain = sampler.get_chain(flat=True)
    alllp = sampler.get_log_prob(flat=True)
    imap = int(np.argmax(alllp))
    try:
        tau = float(np.max(sampler.get_autocorr_time(tol=0)))
    except Exception:
        tau = np.nan
    sel = rng.choice(chain.shape[0], size=min(ndraw, chain.shape[0]),
                     replace=False)
    c0, k = prob.line_draws(chain[sel], rng)
    return {
        "chain": chain, "logp": lp, "theta_map": allchain[imap],
        "logp_map": alllp[imap], "tau": tau,
        "acc": float(np.mean(sampler.acceptance_fraction)),
        "c0_draws": c0, "k_draws": k, "draw_index": sel,
        "nwalkers": nwalkers,
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def pct(x, axis=0):
    return np.nanpercentile(x, [16, 50, 84], axis=axis)


def write_results(opath, prob, res):
    full = prob.full_params(res["chain"])
    lo, med, hi = pct(full)
    a_map = prob.fields(res["theta_map"])
    models, c0_map, k_map, chi2_map = prob.model_full(
        a_map, prob.broad_values(res["theta_map"]))
    npts = int(prob.counts.sum())
    nfree = prob.ndim + 2 * prob.nl
    chi2_tot = float(np.sum(chi2_map))
    c0lo, c0med, c0hi = pct(res["c0_draws"])
    klo, kmed, khi = pct(res["k_draws"])
    with open(os.path.join(opath, "shared_results.txt"), "w") as f:
        f.write("# Shared filling factors, %d lines, marginalised over per-line "
                "c0 and k.\n" % prob.nl)
        f.write("# B = %s kG. Columns: median, -err (16%%), +err (84%%), MAP.\n"
                % " ".join("%g" % b for b in prob.bs))
        f.write("# chi2 (profiled, at the MAP fields) = %.3f for %d points, "
                "%d free parameters -> reduced %.3f\n"
                % (chi2_tot, npts, nfree, chi2_tot / max(npts - nfree, 1)))
        f.write("# acceptance %.3f, autocorrelation time (max) %.1f steps, "
                "%d walkers, %d post-burn samples\n"
                % (res["acc"], res["tau"], res["nwalkers"],
                   res["chain"].shape[0]))
        f.write("#\n# --- global ---\n# name median em ep map\n")
        mapfull = prob.full_params(res["theta_map"])[0]
        for j, n in enumerate(prob.names):
            f.write("%s %.5g %.3g %.3g %.5g\n"
                    % (n, med[j], med[j] - lo[j], hi[j] - med[j], mapfull[j]))
        f.write("#\n# --- per line ---\n")
        f.write("# line wlo_nm whi_nm npts c0 c0_em c0_ep k k_em k_ep "
                "chi2_map redchi2_map\n")
        for r in range(prob.nl):
            w = prob.wvls[r]
            f.write("%d %.5f %.5f %d %.5g %.3g %.3g %.5g %.3g %.3g %.3f %.3f\n"
                    % (prob.lines[r], np.nanmin(w), np.nanmax(w),
                       prob.counts[r],
                       c0med[r], c0med[r] - c0lo[r], c0hi[r] - c0med[r],
                       kmed[r], kmed[r] - klo[r], khi[r] - kmed[r],
                       chi2_map[r], chi2_map[r] / max(prob.counts[r] - 2, 1)))
    np.savez_compressed(os.path.join(opath, "shared_samples.npz"),
                        chain=res["chain"], logp=res["logp"],
                        names=np.array(prob.names[:prob.nB]
                                       + prob.names[prob.nB + 1:]),
                        bs=prob.bs,
                        theta_map=res["theta_map"])
    np.savez_compressed(os.path.join(opath, "shared_line_params.npz"),
                        lines=np.array(prob.lines), c0=res["c0_draws"],
                        k=res["k_draws"], draw_index=res["draw_index"])
    np.savez_compressed(os.path.join(opath, "shared_models.npz"),
                        lines=np.array(prob.lines),
                        wvl=np.array([prob.wvls[r] for r in range(prob.nl)]),
                        model_map=np.array(models),
                        c0_map=c0_map, k_map=k_map, chi2_map=chi2_map)
    return models, c0_map, k_map, chi2_map


def plot_results(opath, prob, res, models, fluxes, per_page=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    full = prob.full_params(res["chain"])
    with PdfPages(os.path.join(opath, "shared_fits.pdf")) as pdf:
        fig, axs = plt.subplots(2, int(np.ceil((prob.nB + 1) / 2)),
                                figsize=(11, 5), squeeze=False)
        for ax in axs.ravel():
            ax.axis("off")
        for j, ax in zip(range(prob.nB + 1), axs.ravel()):
            ax.axis("on")
            ax.hist(full[:, j], bins=40, color="C0")
            ax.set_title(prob.names[j], fontsize=9)
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        for p0 in range(0, prob.nl, per_page):
            rr = list(range(p0, min(p0 + per_page, prob.nl)))
            nrow = int(np.ceil(len(rr) / 3))
            fig, axs = plt.subplots(nrow, 3, figsize=(11, 2.6 * nrow),
                                    squeeze=False)
            for ax in axs.ravel():
                ax.axis("off")
            for ax, r in zip(axs.ravel(), rr):
                ax.axis("on")
                ix = prob.idxs[r]
                ax.plot(prob.wvls[r][ix], fluxes[r][ix], ".", color="k", ms=2)
                ax.plot(prob.wvls[r], models[r], color="C3", lw=1)
                ax.set_title("line %d" % prob.lines[r], fontsize=8)
                ax.tick_params(labelsize=6)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Glue to SpectralAnalysis (heavy imports are lazy)
# ---------------------------------------------------------------------------
def _gen(SA, coeffs, T, L, M, A, rv, vsini, vmac, vb):
    return SA.gen_spec(SA.obs_wvl, SA.obs_flux, SA.obs_err, SA.nan_mask,
                       SA.nwvls, SA.grid_n, coeffs, T, L, M, A,
                       SA.teffs, SA.loggs, SA.mhs, SA.alphas, vb,
                       rv, vsini, vmac, SA.veilingFacToFit, SA._T2,
                       SA.fillTeffs)


def require_plain_grid(SA):
    if getattr(SA, "diskIntegrationMode", None) != 0:
        raise NotImplementedError(
            "Fitting the broadening / RV per line is only implemented for "
            "plain interpolated grids (diskIntegrationMode 0); this grid has "
            "mode %s. Fix them instead (--vsini, --rv, ...)."
            % getattr(SA, "diskIntegrationMode", None))
    if SA.logCoeffs:
        raise NotImplementedError("logCoeffs=True is not supported.")


def compute_unbroadened_basis(SA, T, L, M, A):
    """Interpolated, UNBROADENED models for each field strength on the model
    wavelength grid. Returns (nB, nreg, npix_model)."""
    from asap.c_tools.interpolate_4d import wrap_interpolate_4d_opt as interp4d
    out = []
    for i in range(len(SA.bs)):
        _, s = interp4d(T, L, M, A, SA.teffs, SA.loggs, SA.mhs, SA.alphas,
                        SA.grid_n[i], 0)
        out.append(np.array(s, dtype=float))
    return np.array(out)


def fixed_veiling(SA):
    """Veiling on the observed wavelengths for the configured (fixed) values,
    as in gen_spec_int_spectra. Returns (nreg, npix)."""
    from asap.spectral_analysis_pack import veiling_function
    veilingFac = SA.veilingFac
    if SA.fitVeiling:
        if SA.fitBands == "":
            raise Exception("fitBands empty but fitVeiling==True")
        fitveilpos = np.zeros(SA.nbFitVeil, dtype=int)
        for ib, band in enumerate(SA.fitBands):
            fitveilpos[ib] = SA.veilingBands.find(band)
        veilingFac[fitveilpos] = SA.veilingFacToFit
    return np.asarray(veiling_function(veilingFac, SA.obs_wvl,
                                       SA.veilingBands), dtype=float)


def make_broadener(SA, r, basis_m, veil):
    return Broadener(SA.nwvls[r], basis_m[:, r, :], SA.obs_wvl[r],
                     veil[r], SA.vinstru, SA.smoothSpectraVel, SA.vmacMode)


def check_broadening(SA, T, L, M, A, rv, vsini, vmac, vb, ntest=3, seed=1):
    """Compare Broadener.observe with the full gen_spec for random filling
    factors, at the fixed values and at perturbed rv / vsini / vmac / vb."""
    require_plain_grid(SA)
    rng = np.random.default_rng(seed)
    basis_m = compute_unbroadened_basis(SA, T, L, M, A)
    veil = fixed_veiling(SA)
    brs = [make_broadener(SA, r, basis_m, veil) for r in range(len(SA.obs_wvl))]
    nB = basis_m.shape[0]
    tests = [(rv, vsini, vmac, vb),
             (rv + 0.7, vsini + 2.0, vmac + 1.0, vb + 1.5)]
    worst = 0.0
    for (rv_, vs_, vm_, vb_) in tests:
        for _ in range(ntest):
            c = rng.dirichlet(np.ones(nB))
            direct = _gen(SA, c, T, L, M, A, rv_, vs_, vm_, vb_)
            mine = np.array([b.observe(c, rv_, vs_, vm_, vb_) for b in brs])
            d = np.nanmax(np.abs(direct - mine))
            worst = max(worst, d)
            print("  rv=%.2f vsini=%.2f vmac=%.2f vb=%.2f  "
                  "max|gen_spec - Broadener| = %.3e" % (rv_, vs_, vm_, vb_, d))
    return worst


def compute_basis(SA, T, L, M, A, rv, vsini, vmac, vb):
    """(nB, nreg, npix): model for each field strength, one-hot filling
    factors, continuum adjustment off."""
    nB = len(SA.bs)
    out = []
    for i in range(nB):
        c = np.zeros(nB)
        c[i] = 1.0
        out.append(_gen(SA, c, T, L, M, A, rv, vsini, vmac, vb))
    return np.array(out)


def check_linearity(SA, basis, pars, ntest=3, seed=0):
    rng = np.random.default_rng(seed)
    nB = basis.shape[0]
    worst = 0.0
    for _ in range(ntest):
        c = rng.dirichlet(np.ones(nB))
        direct = _gen(SA, c, *pars)
        combo = np.tensordot(c, basis, axes=1)
        d = np.nanmax(np.abs(direct - combo))
        worst = max(worst, d)
        print("  coeffs=%s  max|direct - basis combination| = %.3e"
              % (np.round(c, 3), d))
    return worst


def parse_lines(spec, nreg):
    if spec is None:
        return list(range(nreg))
    sel = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            sel.extend(range(int(a), int(b) + 1))
        elif part:
            sel.append(int(part))
    return [s for s in sel if 0 <= s < nreg]


def build_parser():
    p = argparse.ArgumentParser(
        description="Fit all lines jointly with SHARED magnetic filling "
                    "factors and a separate local continuum level and "
                    "line-depth factor per line (marginalised analytically).")
    p.add_argument("star", type=str)
    p.add_argument("--config", default=None,
                   help="config.ini (default: ./config.ini)")
    p.add_argument("--outdir", default=None)
    p.add_argument("--nwalkers", type=int, default=None,
                   help="default: max(32, 4*ndim)")
    p.add_argument("--nsteps", type=int, default=5000)
    p.add_argument("--burn", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--ndraw", type=int, default=4000,
                   help="posterior draws used for the per-line c0 and k")
    p.add_argument("--lines", default=None,
                   help="subset of regions, e.g. '0,3,10-20'")
    for n in ("teff", "logg", "mh", "alpha", "rv", "vsini", "vmac", "vb"):
        p.add_argument("--" + n, type=float, default=None,
                       help="fixed value (default: from config)")
    p.add_argument("--fit-vsini", action="store_true",
                   help="also fit the rotational broadening (shared by all lines)")
    p.add_argument("--fit-vmac", action="store_true",
                   help="also fit the macroturbulence (shared by all lines)")
    p.add_argument("--fit-vb", action="store_true",
                   help="also fit the Gaussian broadening (shared by all lines)")
    for n, rng_ in (("vsini", (0.0, 100.0)), ("vmac", (0.0, 50.0)),
                    ("vb", (0.0, 50.0))):
        p.add_argument("--%s-range" % n, type=float, nargs=2, default=rng_,
                       metavar=("LO", "HI"),
                       help="flat prior for fitted %s (default %g %g); the "
                            "config / --%s value is the starting point"
                            % ((n,) + rng_ + (n,)))
    p.add_argument("--check-broadening", action="store_true",
                   help="verify the per-step broadening == gen_spec, then exit")
    p.add_argument("--check-linearity", action="store_true",
                   help="verify basis combination == full gen_spec, then exit")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("-c", "--ncores", type=int, default=1)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from asap.SpectralAnalysis import SpectralAnalysis

    star = args.star.strip()
    if star.endswith(".fits"):
        star = star[:-5]
    config_file = args.config or os.path.join(os.getcwd(), "config.ini")
    opath = args.outdir or "linefit_shared_%s/" % star.replace("/", "_")
    os.makedirs(opath, exist_ok=True)
    config_copy = os.path.join(opath, "config_copy.ini")
    if os.path.isfile(config_copy):
        os.chmod(config_copy, 0o644)
    shutil.copy(config_file, config_copy)

    SA = SpectralAnalysis()
    SA.set_opath(opath)
    SA.set_star(star)
    SA.read_config(config_copy)

    infile = None
    for cand in ("%s_templates.fits", "%s.fits", "%s_template.fits"):
        f = SA.pathtodata + cand % star
        if os.path.isfile(f):
            infile = f
            if "_templates" in cand:
                break
    if infile is None:
        raise FileNotFoundError("No observation file for %s in %s"
                                % (star, SA.pathtodata))

    print("---- Loading observations ----")
    med_wvl, med_spectrum, med_err, berv = SA.load_obs(infile)
    obs_wvl, obs_flux, obs_err, nan_mask, regions = SA.create_regions(
        SA.linelist, med_wvl, med_spectrum, med_err, berv)
    print("---- Loading grid ----")
    SA.load_grid(SA.pathtogrid, regions)
    if SA.renorm:
        SA.compute_normFactor(SA.normFactor)
    SA.set_adjcont(False)       # continuum is fitted per line here

    def pick(v, default):
        return default if v is None else v

    T = pick(args.teff, SA._T)
    L = pick(args.logg, SA._L)
    M = pick(args.mh, SA._M)
    A = pick(args.alpha, SA._A)
    rv = pick(args.rv, SA.rv)
    vsini = pick(args.vsini, SA.vsini)
    vmac = pick(args.vmac, SA.vmac)
    vb = pick(args.vb, SA.vb)
    print("Fixed: Teff=%s logg=%s [M/H]=%s [a/Fe]=%s rv=%s vsini=%s vmac=%s vb=%s"
          % (T, L, M, A, rv, vsini, vmac, vb))
    pars = (T, L, M, A, rv, vsini, vmac, vb)
    fit_broad = tuple(n for n in ("vsini", "vmac", "vb")
                      if getattr(args, "fit_" + n))
    if fit_broad or args.check_broadening:
        require_plain_grid(SA)
    if args.check_broadening:
        worst = check_broadening(SA, *pars)
        print("Worst deviation: %.3e" % worst)
        return

    t0 = time.time()
    basis = compute_basis(SA, *pars)               # (nB, nreg, npix)
    print("Basis computed for %d field strengths in %.1f s"
          % (basis.shape[0], time.time() - t0))
    if args.check_linearity:
        worst = check_linearity(SA, basis, pars)
        print("Worst deviation: %.3e" % worst)
        return

    err_scale = np.sqrt(SA.normFactor)
    rows, cols = SA.IDXTOFIT
    sel, wv, fl, er, ix, bl = [], [], [], [], [], []
    for r in parse_lines(args.lines, len(SA.obs_wvl)):
        idx = cols[rows == r]
        if len(idx) < 5:
            print("Line %d: only %d usable points, skipped" % (r, len(idx)))
            continue
        b = basis[:, r, :]
        if not np.all(np.isfinite(b[:, idx])) or np.ptp(b[:, idx]) < 1e-8:
            print("Line %d: model has no usable structure, skipped" % r)
            continue
        sel.append(r)
        wv.append(SA.obs_wvl[r])
        fl.append(SA.obs_flux[r])
        er.append(SA.obs_err[r] * err_scale)
        ix.append(idx)
        bl.append(b)
    broadeners = None
    if fit_broad:
        t0 = time.time()
        basis_m = compute_unbroadened_basis(SA, T, L, M, A)
        veil = fixed_veiling(SA)
        broadeners = [make_broadener(SA, r, basis_m, veil) for r in sel]
        print("Fitting %s (shared by all lines); unbroadened basis in %.1f s"
              % (", ".join(fit_broad), time.time() - t0))
    prob = SharedProblem(
        sel, wv, fl, er, ix, bl, SA.bs, broadeners=broadeners,
        fit_broad=fit_broad,
        fixed_broad=dict(rv=rv, vsini=vsini, vmac=vmac, vb=vb),
        bounds={n: tuple(getattr(args, n + "_range")) for n in fit_broad})

    nwalkers = args.nwalkers or max(32, 4 * prob.ndim)
    print("Fitting %d lines jointly: %d dims (%d filling factors + %d "
          "broadening), %d walkers, %d steps"
          % (prob.nl, prob.ndim, prob.nB - 1, len(fit_broad), nwalkers,
             args.nsteps))
    t0 = time.time()
    res = run_mcmc(prob, nwalkers, args.nsteps, args.burn, args.seed,
                   args.ndraw, progress=args.progress, ncores=args.ncores)
    print("MCMC done in %.1f s" % (time.time() - t0))

    models, c0_map, k_map, chi2_map = write_results(opath, prob, res)
    if args.plot:
        plot_results(opath, prob, res, models, fl)
    print("Results in %s" % opath)

    # --- fixed 0 kG field (a = [1, 0, ...]); fitted broadening is RE-FITTED for this model; per-line c0, k analytically marginalised
    from scipy.optimize import minimize
    th0 = prob.zero_theta(res["theta_map"]); nf = prob.nB - 1
    if prob.fit_broad:      # best broadening for the non-magnetic model (broadening can mimic Zeeman broadening)
        th0[nf:] = minimize(lambda x: -prob(np.r_[th0[:nf], x]), th0[nf:], method="Nelder-Mead").x
        print("0 kG broadening:", dict(zip(prob.fit_broad, np.round(th0[nf:], 3))))
    a0 = prob.fields(th0)
    models0, c0_0, k_0, chi2_0 = prob.model_full(a0, prob.broad_values(th0))
    c0d, kd = prob.line_draws(np.tile(th0, (args.ndraw, 1)), np.random.default_rng(args.seed))
    (c_lo, c_md, c_hi), (k_lo, k_md, k_hi) = pct(c0d), pct(kd)
    np.savetxt(os.path.join(opath, "zero_field_results.txt"), np.column_stack(
        [prob.lines, c_md, c_md - c_lo, c_hi - c_md, k_md, k_md - k_lo, k_hi - k_md, chi2_0]),
        fmt=["%d"] + ["%.5g"] * 7, header="line c0 c0_em c0_ep k k_em k_ep chi2")
    np.savez_compressed(os.path.join(opath, "zero_field_models.npz"), lines=prob.lines, flux=np.array(fl),
                        wvl=np.array(prob.wvls), model=np.array(models0), c0=c0_0, k=k_0, chi2=chi2_0,
                        used=np.array([np.isin(np.arange(len(w)), i) for w, i in zip(prob.wvls, prob.idxs)]))
    print("chi2: 0 kG %.1f, MAP fields %.1f" % (chi2_0.sum(), prob.model_full(prob.fields(res["theta_map"]), prob.broad_values(res["theta_map"]))[3].sum()))

if __name__ == "__main__":
    main()