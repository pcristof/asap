"""Interactive, lightweight editor for the ASAP line-list (region) file.

Shows an observed spectrum (optional) with the regions of the line list as
shaded bands, numbered like the lines of the fit (0, 1, 2, ... in file order),
and lets you add, delete and move region boundaries with the keyboard. Every
change appears immediately. Comments, header (units) and extra columns of the
file are preserved; untouched lines are written back verbatim.

Usage:
    python -m asap.linelist_editor line_list.txt                  # regions only
    python -m asap.linelist_editor line_list.txt template.fits    # + spectrum
    python -m asap.linelist_editor line_list.txt --fit results_dir/   # + data and best fit
    python -m asap.linelist_editor line_list.txt template.fits --rv -12.3 \
            --out new_list.txt --xlim 1600 1620

Keys (cursor must be over the plot):
    a          add: press at the left edge, press again at the right edge
    A          add a region of --width nm centred on the cursor
    d          delete the region under the cursor
    e          grab the edge nearest to the cursor ...
                  ... then 'e' again moves it to the cursor, or
                  left/right arrows nudge it by one pixel (shift: 10 pixels);
                  Enter/Esc releases it
    u          undo
    w          write the file (a .bak copy is made first)
    m / z      show / hide the best-fit model / the 0 kG model (with --fit)
    ] / [      jump to the next / previous region
    =  /  -    zoom in / out on the x axis (around the cursor)
    +  /  _    zoom in / out on the y axis (around the cursor)
    .  /  ,    move right / left
    >  /  <    move up / down
    0          back to automatic y range (it is automatic until you use + _ > <)
    scroll     zoom in x around the cursor      (toolbar zoom/pan also work)
    Esc        cancel the pending action
    ?          print this help

Fit quality (with --fit): every region that can be matched to a fit window shows
its reduced chi2 in a coloured badge (green near 1, red high). The slider and the
button at the bottom flag regions by chi2: drag the slider to set the threshold,
click the button to cycle  above -> below -> off (default: above).  Flagged regions turn red, and
when the list is written ('w') they are COMMENTED OUT (kept, with a note), not
removed. chi2 comes from the fit run (linefit_shared / linefit: reduced chi2 of the
MAP; original ASAP: chi2 / number of fitted pixels). New or moved regions keep the
chi2 of the fit window they came from.

--fit takes the output folder of a run and finds, by itself, fit-data.fits (the
original ASAP code: data, best fit, no-field fit), shared_models.npz (+
zero_field_models.npz) of asap.linefit_shared, or linefit_models.npz of
asap.linefit. The models are those of the regions at the time of the fit: after
you edit the list they do not change, they just stay where they were computed.

Region numbers are the order in the file. Deleting a region renumbers the
following ones; new regions are appended at the end of the file.
Needs numpy and matplotlib (astropy only to read .fits).
"""
import os
import sys
import copy
import shutil
import argparse
import numpy as np

_C_KMS = 299792.458
HELP = __doc__.split("Keys (cursor must be over the plot):")[1].split("Region numbers")[0]


# ---------------------------------------------------------------------------
# File model (no matplotlib)
# ---------------------------------------------------------------------------
class Region:
    """One line of the list. c, lo, hi in nm. raw: the original text (None
    once modified, or for new regions)."""
    __slots__ = ("c", "lo", "hi", "extra", "raw", "chi2", "flag")

    def __init__(self, c, lo, hi, extra=(), raw=None, chi2=None, flag=False):
        self.c, self.lo, self.hi = float(c), float(lo), float(hi)
        self.extra = list(extra)
        self.raw = raw
        self.chi2 = chi2        # reduced chi2 of the fit window it sits in (or None)
        self.flag = flag        # flagged by the chi2 threshold -> commented out on save

    def copy(self):
        return Region(self.c, self.lo, self.hi, self.extra, self.raw,
                      self.chi2, self.flag)


class LineList:
    def __init__(self, path=None):
        self.path = path
        self.items = []             # str (comment/blank line) or Region
        self.factor = 1.0           # nm per file unit
        self.history = []
        self.flag_note = ""         # added to lines commented out on save
        self.modified = False
        self._backed_up = False
        if path is not None:
            self._read(path)

    # -- reading -------------------------------------------------------------
    def _read(self, path):
        header, in_header = {}, True
        with open(path) as f:
            lines = f.read().splitlines()
        parsed = []
        for line in lines:
            s = line.strip()
            if s == "" or s[0] == "#":
                if in_header and s[:1] == "#":
                    for tok in s.split():
                        for sep in "=:":
                            if sep in tok:
                                k, v = tok.split(sep, 1)
                                header[k] = v
                                break
                parsed.append(line)
            else:
                in_header = False
                parsed.append((line, s.split()))
        units = str(header.get("unit", header.get("units", "nm"))).lower()
        self.factor = 0.1 if units in ("a", "angs", "angstrom") else 1.0
        for p in parsed:
            if isinstance(p, str):
                self.items.append(p)
            else:
                line, t = p
                c, lo, hi = (float(x) * self.factor for x in t[:3])
                self.items.append(Region(c, lo, hi, t[3:], raw=line))

    @property
    def regions(self):
        return [i for i in self.items if isinstance(i, Region)]

    # -- editing -------------------------------------------------------------
    def key(self):
        return tuple((r.lo, r.hi, r.c) for r in self.regions)

    def snapshot(self):
        self.history.append([i.copy() if isinstance(i, Region) else i
                             for i in self.items])
        if len(self.history) > 200:
            self.history.pop(0)

    def undo(self):
        """Back to the last snapshot that differs from the current state."""
        cur = self.key()
        while self.history:
            self.items = self.history.pop()
            if self.key() != cur:
                self.modified = True
                return True
        return False

    def add(self, lo, hi):
        lo, hi = sorted((float(lo), float(hi)))
        if hi - lo <= 0:
            return None
        regs = self.regions
        extra = ["Uk", "1", "0"] if any(r.extra for r in regs) else []
        self.snapshot()
        r = Region(0.5 * (lo + hi), lo, hi, extra)
        self.items.append(r)
        self.modified = True
        return r

    def delete(self, region):
        self.snapshot()
        self.items = [i for i in self.items if i is not region]
        self.modified = True

    def set_edge(self, region, which, x):
        lo, hi = (x, region.hi) if which == "lo" else (region.lo, x)
        if not hi > lo:
            return False
        region.lo, region.hi = lo, hi
        if not lo <= region.c <= hi:
            region.c = 0.5 * (lo + hi)
        region.raw = None
        self.modified = True
        return True

    # -- writing -------------------------------------------------------------
    def save(self, path=None, backup=True):
        path = path or self.path
        if backup and os.path.isfile(path) and not self._backed_up:
            bak, n = path + ".bak", 1
            while os.path.exists(bak):
                n += 1
                bak = "%s.bak%d" % (path, n)
            shutil.copy(path, bak)
            self._backed_up = True
        ## Sort the list by lower wavelength
        it = iter(sorted(self.regions, key=lambda r: r.lo))      # sort by wavelength,
        self.items = [next(it) if isinstance(i, Region) else i    # comments stay in place
                      for i in self.items]
        ##
        f = self.factor
        out = []
        for i in self.items:
            if isinstance(i, str):
                out.append(i)
            elif i.raw is not None:
                out.append(i.raw)
            else:
                out.append(" ".join(["%.5f" % (i.c / f), "%.5f" % (i.lo / f),
                                     "%.5f" % (i.hi / f)] + i.extra))
            if not isinstance(i, str) and i.flag:       # flagged: comment out, keep
                out[-1] = "# %s   # flagged: chi2r=%.3f %s" % (
                    out[-1], i.chi2, self.flag_note)
        with open(path, "w") as fh:
            fh.write("\n".join(out) + "\n")
        self.modified = False
        return path


# ---------------------------------------------------------------------------
# Observation (optional)
# ---------------------------------------------------------------------------
def doppler(v):
    return np.sqrt((1.0 + v / _C_KMS) / (1.0 - v / _C_KMS))


def load_obs(path, rv=0.0):
    """List of (wavelength_nm, flux) per order, wavelengths shifted by rv
    (km/s) like SpectralAnalysis.load_obs. Reads the .fits layouts used by ASAP
    ('WaveAB'/'StokesI', or HDU1 = wavelength, HDU2 = flux) or a two-column
    text file (nm, flux)."""
    if os.path.splitext(path)[1].lower() in (".fits", ".fit", ".fz"):
        from astropy.io import fits
        with fits.open(path) as h:
            names = [x.name for x in h]
            if "WaveAB" in names and "StokesI" in names:
                w, fl = h["WaveAB"].data, h["StokesI"].data
            else:
                w, fl = h[1].data, h[2].data
            w, fl = np.array(w, float), np.array(fl, float)
    else:
        d = np.loadtxt(path)
        w, fl = d[:, 0], d[:, 1]
    w, fl = np.atleast_2d(w) * doppler(rv), np.atleast_2d(fl)
    if w.shape != fl.shape:
        raise ValueError("wavelength and flux arrays differ in shape: %s vs %s"
                         % (w.shape, fl.shape))
    orders = []
    for wo, fo in zip(w, fl):
        ok = np.isfinite(wo)
        wo, fo = wo[ok], fo[ok]
        if len(wo) < 2:
            continue
        srt = np.argsort(wo)
        orders.append((wo[srt], fo[srt]))
    return orders


def _segments(w, f, to_nm=True):
    """2-D (nreg, npix) arrays -> list of (wavelength_nm, values) per region,
    sorted, NaN wavelengths dropped."""
    w, f = np.atleast_2d(np.asarray(w, float)), np.atleast_2d(np.asarray(f, float))
    out = []
    for wo, fo in zip(w, f):
        ok = np.isfinite(wo)
        wo, fo = wo[ok], fo[ok]
        if len(wo) < 2:
            out.append(None)
            continue
        if to_nm and np.nanmedian(wo) > 3000:         # Angstrom -> nm
            wo = wo / 10.0
        srt = np.argsort(wo)
        out.append((wo[srt], fo[srt]))
    return out


def _window(seg):
    return (np.nan, np.nan) if seg is None else (seg[0][0], seg[0][-1])


def _table_rows(path, minc):
    """Numeric rows (>= minc columns) of a results text file."""
    rows = []
    if os.path.isfile(path):
        for line in open(path):
            if line.strip() and not line.lstrip().startswith("#"):
                try:
                    v = [float(t) for t in line.split()]
                except ValueError:
                    continue
                if len(v) >= minc:
                    rows.append(v)
    return rows


def _chi2_from_rows(rows, iw0, ichi):
    out = []
    for v in rows:
        lo, hi = v[iw0], v[iw0 + 1]
        if max(lo, hi) > 3000:                    # Angstrom -> nm
            lo, hi = lo / 10.0, hi / 10.0
        out.append((lo, hi, v[ichi]))
    return out


def load_fit(path):
    """Best fit of an ASAP run, from either
       - the original ASAP output folder  (fit-data.fits), or
       - asap.linefit_shared output       (shared_models.npz [+ zero_field_models.npz]), or
       - asap.linefit output              (linefit_models.npz).
    Returns dict: kind, data (list of (w_nm, flux) per region or None),
    fit and zero (same, or None)."""
    d = path if os.path.isdir(path) else os.path.dirname(path) or "."
    f_fits, f_shared = (os.path.join(d, n) for n in ("fit-data.fits", "shared_models.npz"))
    f_zero, f_line = (os.path.join(d, n) for n in ("zero_field_models.npz", "linefit_models.npz"))
    if os.path.isfile(f_shared):
        m = np.load(f_shared)
        data = None
        if "flux" in m.files:
            data = _segments(m["wvl"], m["flux"])
        fit = _segments(m["wvl"], m["model_map"])
        zero = None
        if os.path.isfile(f_zero):
            z = np.load(f_zero)
            zero = _segments(z["wvl"], z["model"])
            if data is None and "flux" in z.files:
                data = _segments(z["wvl"], z["flux"])
        # per-line table of shared_results.txt: line wlo whi npts c0 ... chi2 redchi2
        chi = _chi2_from_rows(_table_rows(os.path.join(d, "shared_results.txt"), 12), 1, -1)
        return dict(kind="linefit_shared", data=data, fit=fit, zero=zero, chi2=chi,
                    index=[int(i) for i in m["lines"]])
    if os.path.isfile(f_line):
        m = np.load(f_line)
        # linefit_results.txt: line wlo whi npts chi2_map redchi2_map ...
        chi = _chi2_from_rows(_table_rows(os.path.join(d, "linefit_results.txt"), 6), 1, 5)
        return dict(kind="linefit", data=_segments(m["wvl"], m["flux"]),
                    fit=_segments(m["wvl"], m["model_map"]), zero=None, chi2=chi,
                    index=[int(i) for i in m["lines"]])
    if os.path.isfile(f_fits):
        from astropy.io import fits
        with fits.open(f_fits) as h:
            w, fl, fi = h["WVL"].data, h["FLUX"].data, h["FIT"].data
            nm = h["FITNOMAG"].data if "FITNOMAG" in [x.name for x in h] else None
            chi = []
            try:       # chi2 as the likelihood computes it: on IDXTOFIT, errors * sqrt(NORMFAC)
                ffit, err = np.atleast_2d(h["FLUXFIT"].data), np.atleast_2d(h["ERROR"].data)
                rows, cols = np.asarray(h["IDXTOFIT"].data)
                nf = float(h[0].header.get("NORMFAC", 1.0))
                wv = np.atleast_2d(np.asarray(w, float))
                for r in range(wv.shape[0]):
                    cc = cols[rows == r]
                    lo, hi = np.nanmin(wv[r]), np.nanmax(wv[r])
                    if hi > 3000:
                        lo, hi = lo / 10.0, hi / 10.0
                    c2 = np.sum(((ffit[r, cc] - np.atleast_2d(fi)[r, cc])
                                 / (err[r, cc] * np.sqrt(nf))) ** 2)
                    chi.append((lo, hi, c2 / len(cc) if len(cc) else np.nan))
            except KeyError:
                pass
        return dict(kind="ASAP", data=_segments(w, fl), fit=_segments(w, fi),
                    zero=None if nm is None else _segments(w, nm), chi2=chi,
                    index=list(range(len(np.atleast_2d(w)))))
    raise FileNotFoundError("no fit-data.fits, shared_models.npz or "
                            "linefit_models.npz in %s" % d)


def decimate(x, y, maxpts):
    """At most ~maxpts points, keeping the min and max of each block."""
    n = len(x)
    if n <= maxpts:
        return x, y
    step = max(n // (maxpts // 2), 2)
    m = n // step * step
    yr = y[:m].reshape(-1, step)
    nan = np.isnan(yr)
    lo = np.where(nan, np.inf, yr).min(axis=1)
    hi = np.where(nan, -np.inf, yr).max(axis=1)
    bad = ~np.isfinite(lo)
    lo[bad] = hi[bad] = np.nan
    xs = x[:m:step]
    return np.repeat(xs, 2), np.column_stack([lo, hi]).ravel()


# ---------------------------------------------------------------------------
# Viewer / controller
# ---------------------------------------------------------------------------
class Editor:
    MAXPTS = 4000          # max points drawn per order ...
    BUDGET = 40000         # ... and in total (wide views of many orders)

    def __init__(self, ll, orders=(), outpath=None, width=0.1, fig=None,
                 xlim=None, fit=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import blended_transform_factory
        self._Rectangle = Rectangle
        self.plt = plt
        self.ll = ll
        self.fitinfo = fit
        orders = list(orders)
        if not orders and fit is not None and fit["data"] is not None:
            orders = [sg for sg in fit["data"] if sg is not None]
        self.orders = orders
        self.show = {"data": True, "fit": True, "zero": True}
        self.yfix = False              # True once the y range was set by hand
        self.mode = "above"            # chi2 flagging: "above" | "below" | "off"
        self.thr = 0.0                 # chi2 threshold
        self.has_chi2 = False
        self.outpath = outpath or ll.path
        self.width = width
        self.pending = None            # ("add", x0) | ("edge",)
        self.sel = None                # (region, "lo"/"hi")
        self.art = {}                  # id(region) -> artists
        self.fig = fig or plt.figure(figsize=(12, 5))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("wavelength (nm)")
        self.ax.set_ylabel("flux")
        self.tr = blended_transform_factory(self.ax.transData,
                                            self.ax.transAxes)
        style = {"data": dict(color="k", lw=0.6),
                 "fit": dict(color="C3", lw=1.1),
                 "zero": dict(color="C0", lw=1.0, ls="--")}
        self.series = []                 # (Line2D, w, f, role)
        for w, f in self.orders:
            self.series.append((self.ax.plot([], [], **style["data"])[0], w, f, "data"))
        if fit is not None:
            for role in ("fit", "zero"):
                for sg in (fit[role] or []):
                    if sg is not None:
                        self.series.append((self.ax.plot([], [], **style[role])[0],
                                            sg[0], sg[1], role))
            names = {"fit": "best fit", "zero": "0 kG"}
            for role in ("fit", "zero"):
                if fit[role] is not None:
                    self.ax.plot([], [], label=names[role], **style[role])
            self.ax.plot([], [], label="data (%s)" % fit["kind"], **style["data"])
            self.ax.legend(loc="lower left", fontsize=7, framealpha=0.7)
        self.vline = self.ax.axvline(np.nan, color="C1", lw=1, ls="--")
        self.selline = self.ax.axvline(np.nan, color="C3", lw=1.8)
        self.status = self.fig.text(0.01, 0.01, "", fontsize=8, va="bottom")
        self.fig.subplots_adjust(bottom=0.12)
        if fit is not None and fit.get("chi2"):
            self.setup_chi2(fit["chi2"])
        if xlim is None:
            regs = ll.regions
            if regs:
                xlim = (min(r.lo for r in regs), max(r.hi for r in regs))
            elif self.orders:
                xlim = (min(o[0][0] for o in self.orders),
                        max(o[0][-1] for o in self.orders))
            else:
                xlim = (0.0, 1.0)
            span = xlim[1] - xlim[0]
            if len(regs) > 1 and span > 0:          # open on the first region
                r0 = regs[0]
                half = 5 * (r0.hi - r0.lo)
                xlim = (r0.c - half, r0.c + half)
        self.ax.set_xlim(*xlim)
        self.ax.callbacks.connect("xlim_changed", lambda ax: self.update_view())
        self.cids = [
            self.fig.canvas.mpl_connect("key_press_event", self.on_key),
            self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion),
            self.fig.canvas.mpl_connect("scroll_event", self.on_scroll),
            self.fig.canvas.mpl_connect("close_event", self.on_close),
        ]
        self.update_view()
        self.sync()

    # -- drawing -------------------------------------------------------------
    def update_view(self):
        x0, x1 = self.ax.get_xlim()
        ys, vis = [], []
        for ln, w, f, role in self.series:
            i0, i1 = np.searchsorted(w, [x0, x1])
            i0, i1 = max(i0 - 1, 0), min(i1 + 1, len(w))
            ok = (i1 - i0 >= 2) and self.show[role]
            vis.append((ln, w, f, i0, i1, role) if ok else (ln, None, None, 0, 0, role))
        nvis = sum(v[1] is not None for v in vis)
        maxpts = int(min(self.MAXPTS, max(600, self.BUDGET // max(nvis, 1))))
        for ln, w, f, i0, i1, role in vis:
            ln.set_visible(w is not None)
            if w is None:
                continue
            xd, yd = decimate(w[i0:i1], f[i0:i1], maxpts)
            ln.set_data(xd, yd)
            fin = yd[np.isfinite(yd)]
            if len(fin) and role == "data":
                ys.append(np.percentile(fin, [0.5, 99.5]))
        if ys and not self.yfix:
            ys = np.array(ys)
            lo, hi = ys[:, 0].min(), ys[:, 1].max()
            pad = 0.08 * (hi - lo if hi > lo else 1.0)
            self.ax.set_ylim(lo - pad, hi + pad)
        self.cull()
        self.fig.canvas.draw_idle()

    def cull(self):
        """Draw only the regions in view; drop numbers/ticks when crowded."""
        x0, x1 = self.ax.get_xlim()
        inview = {id(r) for r in self.ll.regions if r.hi >= x0 and r.lo <= x1}
        detail = len(inview) <= 80
        for k, (rect, tick, txt) in self.art.items():
            v = k in inview
            rect.set_visible(v)
            tick.set_visible(v and detail)
            txt.set_visible(v and detail)

    def sync(self):
        """Create / remove / move the region artists to match the list."""
        regs = self.ll.regions
        self.compute_flags()
        live = {id(r) for r in regs}
        for k in [k for k in self.art if k not in live]:
            for artist in self.art.pop(k):
                artist.remove()
        for n, r in enumerate(regs):
            a = self.art.get(id(r))
            if a is None:
                rect = self._Rectangle((r.lo, 0), r.hi - r.lo, 1,
                                       transform=self.tr, color="C0",
                                       alpha=0.22, lw=0)
                self.ax.add_patch(rect)
                tick, = self.ax.plot([r.c, r.c], [0, 0.05], color="C0", lw=1,
                                     transform=self.tr)
                txt = self.ax.text(r.c, 0.985, "", transform=self.tr,
                                   ha="center", va="top", fontsize=7.5,
                                   color="C0", clip_on=True)
                a = self.art[id(r)] = (rect, tick, txt)
            rect, tick, txt = a
            rect.set_x(r.lo)
            rect.set_width(r.hi - r.lo)
            tick.set_xdata([r.c, r.c])
            txt.set_x(0.5 * (r.lo + r.hi))
            col = "C3" if r.flag else "C0"
            rect.set_color(col)
            rect.set_alpha(0.30 if r.flag else 0.22)
            tick.set_color(col)
            txt.set_color(col)
            if r.chi2 is not None:
                txt.set_text(r"%d" "\n" r"$\chi^2_r$ %.2f" % (n, r.chi2))
                txt.set_bbox(dict(boxstyle="round,pad=0.25", fc="white", alpha=0.88,
                                  ec=self.badge_color(r.chi2), lw=1.2))
            else:
                txt.set_text(str(n))
        self.cull()
        self.refresh_title()
        # self.sync()
        self.fig.canvas.draw_idle()

    # -- chi2 quality and flagging ---------------------------------------------
    def setup_chi2(self, windows):
        """Give each region the reduced chi2 of the fit window containing its
        centre, and build the threshold slider and mode button."""
        from matplotlib.widgets import Slider, Button
        win = np.array([w for w in windows if np.all(np.isfinite(w))], float)
        if len(win) == 0:
            return
        for r in self.ll.regions:
            ok = np.where((win[:, 0] <= r.c) & (r.c <= win[:, 1]))[0]
            if len(ok):
                j = ok[np.argmin(np.abs(0.5 * (win[ok, 0] + win[ok, 1]) - r.c))]
                r.chi2 = float(win[j, 2])
        vals = np.array([r.chi2 for r in self.ll.regions if r.chi2 is not None])
        if len(vals) == 0:
            return
        self.has_chi2 = True
        lo = max(vals.min(), 1e-3) * 0.9
        hi = max(vals.max(), lo * 1.5) * 1.1
        self.fig.subplots_adjust(bottom=0.27)
        self.status.set_y(0.01)
        ax_s = self.fig.add_axes([0.14, 0.085, 0.45, 0.03])
        self.slider = Slider(ax_s, r"$\chi^2_r$ threshold", np.log10(lo), np.log10(hi),
                             valinit=np.log10(hi), color="C3")
        self.slider.valtext.set_text("%.2f" % hi)
        self.slider.on_changed(self._on_slider)
        ax_b = self.fig.add_axes([0.66, 0.07, 0.24, 0.055])
        self.button = Button(ax_b, "", hovercolor="0.9")
        self.button.on_clicked(lambda ev: self.cycle_mode())
        self.thr = hi             # starts above every chi2: nothing flagged yet

    def _on_slider(self, v):
        self.thr = 10.0 ** v
        self.slider.valtext.set_text("%.2f" % self.thr)
        self.sync()

    def cycle_mode(self):
        self.mode = {"above": "below", "below": "off", "off": "above"}[self.mode]
        self.sync()

    def set_threshold(self, value):
        """Set the chi2 threshold (also moves the slider)."""
        self.thr = float(value)
        if hasattr(self, "slider"):
            self.slider.eventson = False
            self.slider.set_val(np.log10(self.thr))
            self.slider.eventson = True
            self.slider.valtext.set_text("%.2f" % self.thr)
        self.sync()

    def compute_flags(self):
        n = 0
        for r in self.ll.regions:
            r.flag = bool(self.has_chi2 and self.mode != "off" and r.chi2 is not None
                          and ((r.chi2 < self.thr) if self.mode == "below"
                               else (r.chi2 > self.thr)))
            n += r.flag
        sym = {"below": "<", "above": ">", "off": "off"}[self.mode]
        self.ll.flag_note = "(%s %.3g)" % (sym, self.thr) if self.mode != "off" else ""
        if self.has_chi2:
            tot = sum(r.chi2 is not None for r in self.ll.regions)
            self.button.label.set_text(
                "flag" if self.mode == "off" else
                r"flag $\chi^2_r$ %s %.2f" % (sym, self.thr))
            self.button.label.set_text(
                self.button.label.get_text() + "   (%d / %d)" % (n, tot))
        return n

    @staticmethod
    def badge_color(chi):
        """Green near 1 -> red when large (log scale)."""
        import matplotlib.pyplot as plt
        return plt.cm.RdYlGn_r(float(np.clip((np.log10(max(chi, 1e-3)) + 0.3) / 1.3, 0, 1)))

    def refresh_title(self):
        self.ax.set_title("%s   %d regions%s   [a add  d delete  e edge  u undo"
                          "  w write  ? help]"
                          % (os.path.basename(self.ll.path or "(new)"),
                             len(self.ll.regions),
                             "  *UNSAVED*" if self.ll.modified else ""),
                          fontsize=9)

    def say(self, msg):
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()

    # -- helpers ---------------------------------------------------------------
    def nearest_edge(self, x):
        best = None
        for r in self.ll.regions:
            for which in ("lo", "hi"):
                d = abs(getattr(r, which) - x)
                if best is None or d < best[0]:
                    best = (d, r, which)
        return None if best is None else (best[1], best[2])

    def pixel(self, x):
        """Typical sample spacing of the spectrum near x (nm)."""
        for w, _ in self.orders:
            if w[0] <= x <= w[-1]:
                i = int(np.clip(np.searchsorted(w, x), 1, len(w) - 1))
                return float(w[i] - w[i - 1])
        x0, x1 = self.ax.get_xlim()
        return (x1 - x0) / 500.0

    def set_sel(self, sel):
        self.sel = sel
        self.selline.set_xdata([np.nan, np.nan] if sel is None else
                               [getattr(sel[0], sel[1])] * 2)
        self.fig.canvas.draw_idle()

    def cancel(self):
        self.pending = None
        self.vline.set_xdata([np.nan, np.nan])
        self.set_sel(None)

    # -- events ----------------------------------------------------------------
    def on_motion(self, ev):
        if self.pending is not None and ev.inaxes is self.ax:
            self.vline.set_xdata([ev.xdata, ev.xdata])
            self.fig.canvas.draw_idle()

    # -- view controls -------------------------------------------------------
    def zoom(self, axis, factor, center=None):
        lim = self.ax.get_xlim() if axis == "x" else self.ax.get_ylim()
        c = 0.5 * (lim[0] + lim[1]) if center is None else center
        new = (c - (c - lim[0]) * factor, c + (lim[1] - c) * factor)
        if axis == "x":
            self.ax.set_xlim(*new)          # update_view redraws
        else:
            self.yfix = True
            self.ax.set_ylim(*new)
            self.fig.canvas.draw_idle()

    def pan(self, axis, frac):
        lim = self.ax.get_xlim() if axis == "x" else self.ax.get_ylim()
        d = frac * (lim[1] - lim[0])
        if axis == "x":
            self.ax.set_xlim(lim[0] + d, lim[1] + d)
        else:
            self.yfix = True
            self.ax.set_ylim(lim[0] + d, lim[1] + d)
            self.fig.canvas.draw_idle()

    def on_scroll(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        s = 0.8 if ev.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim()
        self.ax.set_xlim(ev.xdata - (ev.xdata - x0) * s,
                         ev.xdata + (x1 - ev.xdata) * s)

    def on_close(self, ev):
        if self.ll.modified:
            print("linelist_editor: closed with UNSAVED changes (not written).")

    def on_key(self, ev):
        k = ev.key
        x = ev.xdata if ev.inaxes is self.ax else None      # not over the slider / button
        ll = self.ll
        if k == "?":
            print(HELP)
        elif k == "escape" or k == "enter":
            self.cancel()
            self.say("")
        elif k == "u":
            self.cancel()
            self.say("undone" if ll.undo() else "nothing to undo")
            self.sync()
        elif k == "w":
            p = ll.save(self.outpath)
            nfl = sum(r.flag for r in ll.regions)
            self.say("wrote %d regions to %s%s" % (len(ll.regions), p,
                     "  (%d commented out)" % nfl if nfl else ""))
            self.refresh_title()
        elif k in ("=", "-"):
            self.zoom("x", 0.8 if k == "=" else 1.25, x)
        elif k in ("+", "_"):
            self.zoom("y", 0.8 if k == "+" else 1.25,
                      ev.ydata if ev.inaxes is self.ax else None)
        elif k in ("]", "["):
            self.pan("x", 0.2 if k == "]" else -0.2)
        elif k in ("}", "{"):
            self.pan("y", 0.2 if k == "}" else -0.2)
        elif k == "0":
            self.yfix = False
            self.update_view()
        elif k in ("m", "z"):
            role = "fit" if k == "m" else "zero"
            self.show[role] = not self.show[role]
            self.say("%s %s" % ({"fit": "best fit", "zero": "0 kG model"}[role],
                                "shown" if self.show[role] else "hidden"))
            self.update_view()
        elif k in (".", ","):
            self.jump(+1 if k == "." else -1)
        elif k in ("left", "right", "shift+left", "shift+right"):
            self.nudge(k)
        elif x is None:
            self.say("put the cursor over the plot")
        elif k == "a":
            if self.pending and self.pending[0] == "add":
                x0 = self.pending[1]
                r = ll.add(x0, x)
                self.cancel()
                self.say("added region %d" % (len(ll.regions) - 1) if r else
                         "empty region ignored")
                self.sync()
            else:
                self.cancel()
                self.pending = ("add", x)
                self.vline.set_xdata([x, x])
                self.say("add: move to the other edge and press 'a' (Esc cancels)")
        elif k == "A":
            r = ll.add(x - 0.5 * self.width, x + 0.5 * self.width)
            self.say("added region %d" % (len(ll.regions) - 1))
            self.sync()
        elif k == "d":
            hit = [r for r in ll.regions if r.lo <= x <= r.hi]
            if not hit:
                self.say("no region under the cursor")
            else:
                r = min(hit, key=lambda r: abs(0.5 * (r.lo + r.hi) - x))
                n = ll.regions.index(r)
                ll.delete(r)
                self.cancel()
                self.say("deleted region %d (later regions renumbered)" % n)
                self.sync()
        elif k == "e":
            if self.sel is None:
                e = self.nearest_edge(x)
                if e is None:
                    self.say("no regions")
                    return
                ll.snapshot()
                self.pending = ("edge",)
                self.set_sel(e)
                self.say("edge of region %d grabbed: 'e' moves it to the cursor, "
                         "arrows nudge, Enter releases"
                         % ll.regions.index(e[0]))
            else:
                r, which = self.sel
                if ll.set_edge(r, which, x):
                    self.set_sel((r, which))
                    self.say("moved")
                else:
                    self.say("edge cannot cross the other one")
                self.sync()

    def nudge(self, k):
        if self.sel is None:
            self.say("grab an edge first with 'e'")
            return
        r, which = self.sel
        step = self.pixel(getattr(r, which)) * (10 if k.startswith("shift") else 1)
        step = -step if k.endswith("left") else step
        if self.ll.set_edge(r, which, getattr(r, which) + step):
            self.set_sel((r, which))
            self.sync()
        else:
            self.say("edge cannot cross the other one")

    def jump(self, direction):
        regs = sorted(self.ll.regions, key=lambda r: r.lo)
        if not regs:
            return
        x0, x1 = self.ax.get_xlim()
        mid = 0.5 * (x0 + x1)
        cand = [r for r in regs if (0.5 * (r.lo + r.hi) - mid) * direction > 1e-9 * abs(mid)]
        if not cand:
            self.say("no further regions")
            return
        r = cand[0] if direction > 0 else cand[-1]
        half = max(0.5 * (x1 - x0), 2 * (r.hi - r.lo))
        c = 0.5 * (r.lo + r.hi)
        self.ax.set_xlim(c - half, c + half)
        self.say("region %d: %.4f - %.4f nm" % (self.ll.regions.index(r), r.lo, r.hi))


# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Interactive editor for the ASAP line-list regions. "
                    "Press ? in the window for the keys.")
    p.add_argument("linelist")
    p.add_argument("obs", nargs="?", default=None,
                   help="observation (.fits as read by ASAP, or 2-column text "
                        "nm/flux); optional")
    p.add_argument("--fit", default=None, metavar="DIR",
                   help="output folder of a fit: shows data + best fit (+ 0 kG). "
                        "Original ASAP (fit-data.fits), asap.linefit_shared or "
                        "asap.linefit. Without OBS, the data come from it.")
    p.add_argument("--rv", type=float, default=0.0,
                   help="shift the spectrum by this RV (km/s), as ASAP does "
                        "with the guessed RV, so that it lines up with the list")
    p.add_argument("--out", default=None,
                   help="write here instead of overwriting the input list")
    p.add_argument("--width", type=float, default=0.1,
                   help="width in nm of regions added with 'A' (default 0.1)")
    p.add_argument("--xlim", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"), help="initial wavelength range, nm")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    import matplotlib.pyplot as plt
    for name, drop in (("keymap.back", ("left",)), ("keymap.forward", ("right",))):
        plt.rcParams[name] = [k for k in plt.rcParams[name] if k not in drop]
    ll = LineList(args.linelist)
    orders = load_obs(args.obs, args.rv) if args.obs else []
    print("%d regions. Press ? in the window for the keys." % len(ll.regions))
    fit = load_fit(args.fit) if args.fit else None
    if fit is not None:
        print("Best fit from %s output (%d regions)." % (fit["kind"], len(fit["fit"])))
    ed = Editor(ll, orders, args.out, args.width, xlim=args.xlim, fit=fit)
    plt.show()


if __name__ == "__main__":
    main()