"""Shared-field joint fit: (1) the analytic marginal likelihood equals a
brute-force numerical integral over (alpha, beta); (2) injected shared fields
and per-line c0, k are recovered."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from asap.linefit_shared import SharedProblem, run_mcmc


def make_lines(rng, nl, nB=4, npix=250, a_true=None, noise=0.01):
    bs = np.array([0., 2., 4., 6.][:nB])
    wv, fl, er, ix, bl, truth = [], [], [], [], [], []
    for r in range(nl):
        w = np.linspace(1600 + 5 * r, 1601.5 + 5 * r, npix)
        wc = w.mean()
        wid = 0.03 + 0.01 * rng.random()
        basis = np.array([1 - (0.55 - 0.05 * i) * np.exp(-0.5 * ((w - wc) / (wid + 0.012 * i)) ** 2)
                          for i in range(nB)])
        c0, k = 0.96 + 0.08 * rng.random(), 0.7 + 0.6 * rng.random()
        model = c0 * (1 - k * (1 - a_true @ basis))
        err = np.full(npix, noise)
        wv.append(w); er.append(err); bl.append(basis)
        fl.append(model + err * rng.standard_normal(npix))
        ix.append(np.arange(npix)); truth.append((c0, k))
    return bs, wv, fl, er, ix, bl, truth


def test_marginal_matches_numeric():
    rng = np.random.default_rng(0)
    nB = 3
    a_true = np.array([0.4, 0.4, 0.2])
    bs, wv, fl, er, ix, bl, _ = make_lines(rng, 3, nB=nB, npix=120, a_true=a_true)
    prob = SharedProblem(range(3), wv, fl, er, ix, bl, bs)
    for theta in (np.array([0.35, 0.25]), np.array([0.1, 0.5])):
        a = prob.fields(theta)
        exact = prob(theta)
        total = 0.0
        for r in range(3):
            mix = a @ bl[r]
            y, e = fl[r], er[r]
            al, be, chi2, det, S1, S2 = [v[r] for v in prob.solve(a)]
            sa, sb = np.sqrt(S2 / det), np.sqrt(prob.S0[r] / det)
            A_ = np.linspace(al - 12 * sa, al + 12 * sa, 600)
            B_ = np.linspace(be - 12 * sb, be + 12 * sb, 600)
            AA, BB = np.meshgrid(A_, B_, indexing="ij")
            M = AA[..., None] + BB[..., None] * mix
            ll = -0.5 * np.sum(((y - M) / e) ** 2, axis=-1) - 0.5 * np.sum(np.log(2 * np.pi * e ** 2))
            m = ll.max()
            trap = getattr(np, "trapezoid", None) or np.trapz
            integ = trap(trap(np.exp(ll - m), B_, axis=1), A_)
            total += m + np.log(integ)
        print("theta", theta, " closed form %.6f  numerical %.6f" % (exact, total))
        assert abs(exact - total) < 1e-3


def test_recovery():
    rng = np.random.default_rng(2)
    a_true = np.array([0.3, 0.3, 0.25, 0.15])
    nl = 15
    bs, wv, fl, er, ix, bl, truth = make_lines(rng, nl, a_true=a_true)
    prob = SharedProblem(range(nl), wv, fl, er, ix, bl, bs)
    t0 = time.time()
    res = run_mcmc(prob, 32, 4000, 0.5, 7, 3000)
    print("MCMC %.1fs (%d lines, ndim=%d) tau=%.0f acc=%.2f" % (time.time() - t0, nl, prob.ndim, res["tau"], res["acc"]))
    full = prob.full_params(res["chain"])
    lo, med, hi = np.percentile(full, [16, 50, 84], axis=0)
    tv = list(a_true) + [a_true @ bs]
    ok = True
    for j, n in enumerate(prob.names):
        sig = 0.5 * (hi[j] - lo[j]); z = (med[j] - tv[j]) / sig
        print("  %-6s true %6.3f fit %6.3f +/- %.3f (%.1f sigma)" % (n, tv[j], med[j], sig, z))
        ok &= abs(z) < 4
    c0l, c0m, c0h = np.percentile(res["c0_draws"], [16, 50, 84], axis=0)
    kl, km, kh = np.percentile(res["k_draws"], [16, 50, 84], axis=0)
    zc = [(c0m[r] - truth[r][0]) / (0.5 * (c0h[r] - c0l[r])) for r in range(nl)]
    zk = [(km[r] - truth[r][1]) / (0.5 * (kh[r] - kl[r])) for r in range(nl)]
    print("  per-line c0: rms z = %.2f, k: rms z = %.2f (expect ~1)" % (np.sqrt(np.mean(np.square(zc))), np.sqrt(np.mean(np.square(zk)))))
    print("  line 0: c0 true %.3f fit %.3f +/- %.3f; k true %.3f fit %.3f +/- %.3f" % (
        truth[0][0], c0m[0], 0.5 * (c0h[0] - c0l[0]), truth[0][1], km[0], 0.5 * (kh[0] - kl[0])))
    assert ok and np.sqrt(np.mean(np.square(zc))) < 2 and np.sqrt(np.mean(np.square(zk))) < 2


if __name__ == "__main__":
    test_marginal_matches_numeric()
    test_recovery()
    print("OK")
