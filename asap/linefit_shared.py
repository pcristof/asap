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

Teff, log(g), [M/H], [a/Fe], RV and the broadening are FIXED (config.ini or
command line). The per-field models S_{i,r} are computed once through the
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
# Core: no heavy imports, testable on synthetic data
# ---------------------------------------------------------------------------
class SharedProblem:
    """Marginal log-posterior of the shared filling factors."""

    def __init__(self, indices, wvls, fluxes, errs, idxs, bases, bs):
        """Per line r: wvls[r], fluxes[r], errs[r] (npix,), idxs[r] (fitted
        pixels), bases[r] (nB, npix). bs: field strengths in kG."""
        self.lines = [int(i) for i in indices]
        self.nl = len(self.lines)
        self.bs = np.asarray(bs, dtype=float)
        self.nB = len(self.bs)
        self.wvls = [np.asarray(w, float) for w in wvls]
        self.idxs = [np.asarray(i) for i in idxs]
        self.bases_full = [np.asarray(b, float) for b in bases]   # (nB, npix)
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
        self.ndim = self.nB - 1
        self.names = ["a%d" % i for i in range(self.nB)] + ["Bmean"]

    # -- per-line linear solve ----------------------------------------------
    def solve(self, a):
        """For filling factors a (nB,), the per-line least-squares (alpha,
        beta), chi2 at that solution, and the normal-matrix pieces."""
        mix = a @ self.B
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
        a = np.empty(self.nB)
        a[1:] = theta
        a[0] = 1.0 - a[1:].sum()
        return a

    def lnprior(self, theta):
        a = self.fields(theta)
        if np.min(a) < 0.0 or np.max(a) > 1.0:
            return -np.inf
        return 0.0

    def __call__(self, theta):
        a = self.fields(theta)
        if np.min(a) < 0.0 or np.max(a) > 1.0:
            return -np.inf
        alpha, beta, chi2, det, S1, S2 = self.solve(a)
        if not np.all(det > 0.0):
            return -np.inf
        val = np.sum(self.lognorm + np.log(2 * np.pi) - 0.5 * np.log(det)
                     - 0.5 * chi2)
        return val if np.isfinite(val) else -np.inf

    # -- derived -------------------------------------------------------------
    def full_params(self, chain):
        chain = np.atleast_2d(chain)
        a = np.empty((chain.shape[0], self.nB))
        a[:, 1:] = chain
        a[:, 0] = 1.0 - chain.sum(axis=1)
        return np.column_stack([a, a @ self.bs])

    def line_draws(self, chain, rng):
        """One draw of (c0_r, k_r) for every line, for every row of chain
        (filling factors a_1..a_{nB-1}). Returns (c0, k), each (nsamp, nl)."""
        n = chain.shape[0]
        c0 = np.empty((n, self.nl))
        k = np.empty((n, self.nl))
        for s in range(n):
            a = self.fields(chain[s])
            alpha, beta, chi2, det, S1, S2 = self.solve(a)
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
            th = a[1:]
            if np.isfinite(self(th)):
                p0.append(th)
        return np.array(p0)

    def model_full(self, a):
        """Best-fit (for fields a) model over each whole region, plus the
        per-line c0, k and chi2."""
        alpha, beta, chi2, det, S1, S2 = self.solve(a)
        models = [alpha[r] + beta[r] * (a @ self.bases_full[r])
                  for r in range(self.nl)]
        c0 = alpha + beta
        with np.errstate(all="ignore"):
            k = beta / c0
        return models, c0, k, chi2


def run_mcmc(prob, nwalkers, nsteps, burn_frac, seed, ndraw, progress=False):
    rng = np.random.default_rng(seed)
    nwalkers = max(int(nwalkers), 2 * prob.ndim + 2)
    p0 = prob.init_walkers(nwalkers, rng)
    sampler = emcee.EnsembleSampler(nwalkers, prob.ndim, prob)
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
    models, c0_map, k_map, chi2_map = prob.model_full(a_map)
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
                        names=np.array(prob.names[:-1]), bs=prob.bs,
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
    p.add_argument("--check-linearity", action="store_true",
                   help="verify basis combination == full gen_spec, then exit")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--progress", action="store_true")
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
    prob = SharedProblem(sel, wv, fl, er, ix, bl, SA.bs)

    nwalkers = args.nwalkers or max(32, 4 * prob.ndim)
    print("Fitting %d lines jointly: %d shared filling-factor dims, "
          "%d walkers, %d steps" % (prob.nl, prob.ndim, nwalkers, args.nsteps))
    t0 = time.time()
    res = run_mcmc(prob, nwalkers, args.nsteps, args.burn, args.seed,
                   args.ndraw, progress=args.progress)
    print("MCMC done in %.1f s" % (time.time() - t0))

    models, c0_map, k_map, chi2_map = write_results(opath, prob, res)
    if args.plot:
        plot_results(opath, prob, res, models, fl)
    print("Results in %s" % opath)


if __name__ == "__main__":
    main()
