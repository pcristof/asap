def main():
    """Usage: python plot_samples.py linefit_<star>/linefit_samples/line_0016.npz [nwalkers]
    (nwalkers is only needed for results from the old version of linefit.py;
    default there: max(32, 4 * number of free parameters), linefit's default.)"""
    import os
    import re
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import corner

    path = sys.argv[1]
    d = np.load(path)
    names = [str(n) for n in d["names"]]          # a0.., k, c0, ..., Bmean
    chain = d["chain"]                            # free parameters only (post burn-in)

    # field strengths: stored in new files, otherwise read from linefit_results.txt
    if "bs" in d.files:
        bs = d["bs"]
    else:
        res = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))),
                        "linefit_results.txt")
        bs = np.array(re.search(r"with B = (.*?) kG", open(res).read()).group(1).split(), float)

    nB = sum(n[0] == "a" and n[1:].isdigit() for n in names)
    a = np.column_stack([1 - chain[:, :nB - 1].sum(axis=1), chain[:, :nB - 1]])
    samples = np.column_stack([a, chain[:, nB - 1:], a @ bs])   # incl. derived a0, <B>

    corner.corner(samples, labels=names, truths=d["map"], quantiles=[0.16, 0.5, 0.84],
                show_titles=True, plot_datapoints=False)
    plt.savefig("corner.png", dpi=120)

    # walker traces: steps x walkers x free parameters
    if "trace" in d.files:
        tr, steps, burn = d["trace"], d["trace_steps"], int(d["burn"])
    else:                                         # old file: post-burn chain only
        nw = int(sys.argv[2]) if len(sys.argv) > 2 else max(32, 4 * chain.shape[1])
        if chain.shape[0] % nw:
            sys.exit("corner.png written; trace skipped: pass the right number of walkers")
        tr = chain.reshape(-1, nw, chain.shape[1])
        steps, burn = np.arange(len(tr)), 0
    fig, axs = plt.subplots(tr.shape[2], 1, sharex=True, figsize=(8, 1.5 * tr.shape[2]))
    for j, ax in enumerate(np.atleast_1d(axs)):
        ax.plot(steps, tr[:, :, j], lw=0.4, alpha=0.5)
        if burn:
            ax.axvline(burn, color="k", ls="--")
        ax.set_ylabel(names[1:-1][j])
    axs[-1].set_xlabel("step")
    opath = path.replace('.npz', '_samples.png')
    fig.savefig(opath, dpi=120, bbox_inches="tight")