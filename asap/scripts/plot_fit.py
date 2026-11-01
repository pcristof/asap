def main():
    """Data, magnetic best fit and (optionally) 0 kG best fit, 12 lines per page.
    Usage: python plot_fits.py linefit_shared_<star> [line numbers ...] [--no-zero]"""
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("lines", nargs="*", type=int, help="line numbers (default: all)")
    ap.add_argument("-m", "--no-zero", action="store_true", help="do not show the 0 kG fit")
    ap.add_argument("-i", "--interactive", action="store_true", help="Interactive figure")
    args = ap.parse_args()

    m = np.load(args.outdir + "/shared_models.npz")       # magnetic best fit
    z = np.load(args.outdir + "/zero_field_models.npz")   # 0 kG best fit + observed flux
    lines = list(m["lines"])
    sel = [i for i, l in enumerate(lines) if not args.lines or l in args.lines]

    with PdfPages(args.outdir+"/fits.pdf") as pdf:
        for p in range(0, len(sel), 12):
            chunk = sel[p:p + 12]
            fig, axs = plt.subplots(-(-len(chunk) // 3), 3, figsize=(15, 3.2 * -(-len(chunk) // 3)),
                                    squeeze=False)
            for ax in axs.ravel()[len(chunk):]:
                ax.axis("off")
            for ax, i in zip(axs.ravel(), chunk):
                ax.plot(z["wvl"][i], z["flux"][i], "k.", ms=2, label="data")
                ax.plot(m["wvl"][i], m["model_map"][i], "r", lw=1, label="magnetic")
                if not args.no_zero:
                    ax.plot(z["wvl"][i], z["model"][i], "b", lw=1, label="0 kG")
                ax.set_title("line %d" % lines[i], fontsize=9)
                if "used" in z.files:   # shade the pixels used in the fit
                    ax.fill_between(z["wvl"][i], 0, 1, where=z["used"][i], lw=0, color="0.88",
                                    transform=ax.get_xaxis_transform(), zorder=0, label="fitted")
            axs[0, 0].legend(fontsize=7)
            fig.tight_layout()
            pdf.savefig(fig)
            if args.interactive:
                plt.show()
            plt.close(fig)

if __name__ == "__main__":
    main()