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

Non-interactive (no window): filter the list by chi2 and save it
    python -m asap.linelist_editor line_list.txt --fit results_dir/ --chi2-max 25 \\
            --out filtered_list.txt [--tell]
Regions whose chi2 is ABOVE --chi2-max (BELOW --chi2-min) are commented out, not
removed, exactly as with the slider and 'w' in the window.

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
    t          show / hide the telluric candidates (only with --tell)
    n  /  N    jump to the next / previous candidate pixel (only with --tell)
    i          show / hide the VALD line identification (with --fit)
    up / down  raise / lower the minimum VALD line depth (shift: 5 times faster)
    x          apply / ignore the species exclusion of --vald-exclude
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
MAP; original ASAP: mean squared residual of the fitted pixels inside each region, errors without
the normFactor, and WITHOUT the pixels flagged by --tell). Moved regions keep the
chi2 of the fit window they came from (linefit outputs).

VALD identification (with --fit): lines of the VALD lists in asap/support_data
are drawn as ticks (taller = deeper) with their species and wavelength. The
file is chosen by the effective temperature of the results (results.txt in the
--fit folder; or --vald-teff), among the vald-lines_*_<T>.txt files; --vald
gives a file explicitly. Wavelengths are converted to the units and medium the
spectrum uses (nm, vacuum: the file's header says A or nm, air or vacuum) and
shifted by the FITTED radial velocity of the results (the 'rv' of results.txt,
never the guessed one; --vald-rv overrides it, and is needed for folders that
have no results.txt, such as asap.linefit_shared). A slider sets the minimum
line depth shown, so that only the deepest lines remain (it can also be moved with
the up / down keys). --vald-exclude TiO [OH "Fe 1" ...] hides species: a bare name
(TiO) hides every ionisation stage, a name with a stage ("Fe 1") only that one;
x switches the exclusion off and on.

Telluric candidates (option --tell, with --fit): pixels where the data fall below the best fit
by a lot, as a narrow spike that a wrong line model would not produce. For each
fit window the residual (data - best fit) is taken over the pixels that were fitted
(all pixels inside the regions of the list when the run did not save them),
scaled by its robust scatter (1.4826 MAD), and a pixel is a candidate when it
is below -SIGMA, belongs to a run of at most WIDTH adjacent pixels beyond SIGMA/2,
and is among the worst FRAC of the window's pixels (at least one). Candidates are
red crosses; a candidate that no region of the list contains any more (because
you moved an edge or deleted the region) turns into a grey circle, so you can see
at once that it is excluded. The badge of a region shows the number inside it.
Off unless you pass --tell. Tune with --tell-sigma, --tell-width, --tell-frac,
--tell-sign.

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
        used = _segments(m["wvl"], m["used"].astype(float)) if "used" in m.files else None
        return dict(kind="linefit_shared", data=data, fit=fit, zero=zero, chi2=chi,
                    used=used, tdata=data, index=[int(i) for i in m["lines"]])
    if os.path.isfile(f_line):
        m = np.load(f_line)
        # linefit_results.txt: line wlo whi npts chi2_map redchi2_map ...
        chi = _chi2_from_rows(_table_rows(os.path.join(d, "linefit_results.txt"), 6), 1, 5)
        dat = _segments(m["wvl"], m["flux"])
        return dict(kind="linefit", data=dat,
                    fit=_segments(m["wvl"], m["model_map"]), zero=None, chi2=chi,
                    used=None, tdata=dat, index=[int(i) for i in m["lines"]])
    if os.path.isfile(f_fits):
        from astropy.io import fits
        with fits.open(f_fits) as h:
            w, fl, fi = h["WVL"].data, h["FLUX"].data, h["FIT"].data
            nm = h["FITNOMAG"].data if "FITNOMAG" in [x.name for x in h] else None
            chi, used, tdata, pix = [], None, None, None
            pix = []
            try:       # chi2 on IDXTOFIT pixels; the saved ERROR already holds the normFactor, removed here
                nf = float(h[0].header.get("NORMFAC", 1.0))
                ffit, err = np.atleast_2d(h["FLUXFIT"].data), np.atleast_2d(h["ERROR"].data)
                err = err #/ np.sqrt(nf)           # de-apply the normFactor: raw errors
                rows, cols = np.asarray(h["IDXTOFIT"].data)
                wv = np.atleast_2d(np.asarray(w, float))
                um = np.zeros(wv.shape)
                um[rows.astype(int), cols.astype(int)] = 1.0
                used = _segments(wv, um)
                tdata = _segments(wv, ffit)
                # squared normalised residual of every fitted pixel (nan elsewhere): the
                # editor averages it over each region, so every region has its own chi2
                r2 = np.full(wv.shape, np.nan)
                ri, ci = rows.astype(int), cols.astype(int)
                r2[ri, ci] = ((ffit[ri, ci] - np.atleast_2d(fi)[ri, ci])
                              / err[ri, ci]) ** 2
                for r in range(wv.shape[0]):
                    x = wv[r] / 10.0 if np.nanmax(wv[r]) > 3000 else wv[r]
                    o = np.argsort(x)
                    pix.append((x[o], r2[r][o]))
            except KeyError:
                pass
        return dict(kind="ASAP", data=_segments(w, fl), fit=_segments(w, fi),
                    zero=None if nm is None else _segments(w, nm), chi2=chi,
                    used=used, tdata=tdata or _segments(w, fl), pix=pix or None,
                    index=list(range(len(np.atleast_2d(w)))))
    raise FileNotFoundError("no fit-data.fits, shared_models.npz or "
                            "linefit_models.npz in %s" % d)


def read_results_scalar(folder, key):
    """A 'flt:<key>:' or 'cst:<key>:' value of results.txt (the ASAP v2 format
    read by asap.reader.read_res_v2), or None."""
    path = os.path.join(folder, "results.txt")
    if not os.path.isfile(path):
        return None
    for line in open(path):
        sl = line.split(":")
        if len(sl) == 3 and sl[0].strip() in ("flt", "cst") and sl[1].strip() == key:
            try:
                return float(sl[2].split()[0])
            except (ValueError, IndexError):
                return None
    return None


def air_to_vacuum(wa):
    """Air -> vacuum wavelengths (Angstrom in, Angstrom out): the VALD formula
    used by asap.analysis_tools.convert_lambda_in_vacuum (method 1)."""
    wa = np.asarray(wa, float)
    s = 1e4 / wa
    n = (1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s ** 2)
         + 0.0001599740894897 / (38.92568793293 - s ** 2))
    return n * wa


def vald_files(directory=None):
    """{model temperature: path} of the vald-lines_*_<T>.txt files."""
    import glob
    import re
    if directory is None:
        try:
            from importlib.resources import files
            directory = str(files("asap.support_data"))
        except Exception:
            directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "support_data")
    out = {}
    for f in glob.glob(os.path.join(directory, "vald-lines_*.txt")):
        m = re.search(r"_(\d+)\.txt$", f)
        if m:
            out[int(m.group(1))] = f
    return out


def species_match(species, names):
    """Boolean array: which entries of `species` ('Fe 1', 'TiO 1', ...) are
    named in `names`. 'TiO' matches every stage of TiO, 'Fe 1' only Fe 1."""
    sp = np.array([" ".join(x.lower().split()) for x in species])
    out = np.zeros(len(sp), bool)
    for n in names:
        n = " ".join(str(n).lower().split())
        if not n:
            continue
        if " " in n:
            out |= sp == n
        else:
            out |= np.array([x.split(" ")[0] == n for x in sp], bool)
    return out


def read_vald_file(path, wlo_nm, whi_nm):
    """Lines of a VALD 'extract stellar' file inside [wlo_nm, whi_nm], vacuum nm.
    Returns (wavelength_nm sorted, depth, species). Units (A / nm) and medium
    (air / vacuum) are read from the header, as asap.mask_tools does."""
    medium = unit = None
    w, dep, sp = [], [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < 4 and medium is None:
                for tag, med in (("WL_vac(", "vac"), ("WL_air(", "air")):
                    if tag in line:
                        medium, unit = med, line.split(tag)[1][0].lower()
            if line[:1] != "'":
                if medium is not None and i > 3 and len(line.split(",")) <= 5:
                    break
                continue
            if medium is None:
                raise ValueError("%s: cannot tell air from vacuum / the units" % path)
            c = line.split(",")
            if len(c) < 10:                              # e.g. the model-atmosphere line
                continue
            x = float(c[1])
            if unit == "a":
                x /= 10.0                               # -> nm
            if medium == "air":
                x = air_to_vacuum(x * 10.0) / 10.0      # -> vacuum
            if wlo_nm <= x <= whi_nm:
                w.append(x)
                dep.append(float(c[9]))
                sp.append(c[0].strip().strip("'"))
    w, dep, sp = np.array(w), np.array(dep), np.array(sp)
    o = np.argsort(w)
    return w[o], dep[o], sp[o]


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


def find_outliers(d, m, cand, sigma=4.0, width=3, frac=0.01, sign="neg"):
    """Indices of isolated spikes in the residual d - m among the candidate
    pixels `cand` (boolean). d, m are in pixel order of one window.
    A pixel is flagged when its robust z-score is beyond `sigma` (below -sigma
    for sign="neg", i.e. absorption; either side for "both"), it sits in a run
    of at most `width` adjacent pixels beyond sigma/2 with the same sign, and the
    total stays within max(1, frac * number of candidates), worst first."""
    d, m, cand = np.asarray(d, float), np.asarray(m, float), np.asarray(cand, bool)
    r = d - m
    ok = cand & np.isfinite(r)
    if ok.sum() < 10:
        return np.array([], dtype=int)
    med = np.median(r[ok])
    scale = 1.4826 * np.median(np.abs(r[ok] - med))
    if not scale > 0:
        scale = np.std(r[ok])
    if not scale > 0:
        return np.array([], dtype=int)
    z = np.where(ok, (r - med) / scale, 0.0)
    sig = -z if sign == "neg" else np.abs(z)
    hot = ok & (sig > 0.5 * sigma)
    runs, i, n = [], 0, len(z)
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and hot[j + 1] and np.sign(z[j + 1]) == np.sign(z[i]):
            j += 1                                     # adjacent, same side
        top = sig[i:j + 1].max()
        if j - i + 1 <= width and top > sigma:         # narrow and strong
            runs.append((top, i, j))
        i = j + 1
    runs.sort(reverse=True)                            # worst first
    budget = max(1, int(np.ceil(frac * ok.sum())))
    out = []
    for top, i, j in runs:
        if len(out) >= budget:
            break
        out.extend(range(i, j + 1))                    # never split a spike
    return np.array(sorted(out), dtype=int)


def tell_candidates(fit, regions, sigma=4.0, width=3, frac=0.01, sign="neg"):
    """Telluric candidate pixels of every fit window: dict(w, f) (nm, flux), sorted.
    See find_outliers. Needs fit['tdata'|'data'] and fit['fit']."""
    out = dict(w=np.zeros(0), f=np.zeros(0))
    data = fit.get("tdata") or fit.get("data")
    model = fit.get("fit")
    if data is None or model is None:
        return out
    used = fit.get("used")
    W, F = [], []
    for j, (dj, mj) in enumerate(zip(data, model)):
        if dj is None or mj is None or len(dj[0]) != len(mj[0]):
            continue
        w, d, m = dj[0], dj[1], mj[1]
        if used is not None and used[j] is not None:
            cand = used[j][1] > 0.5
        else:                              # not saved: pixels in today's regions
            cand = np.zeros(len(w), bool)
            for r in regions:
                cand |= (w >= r.lo) & (w <= r.hi)
        ix = find_outliers(d, m, cand, sigma, width, frac, sign)
        W.extend(w[ix])
        F.extend(d[ix])
    if W:
        W, u = np.unique(np.round(np.array(W), 7), return_index=True)   # overlapping windows
        F = np.array(F)[u]
        srt = np.argsort(W)
        out = dict(w=W[srt], f=F[srt])
    return out


def pixel_chi2(regions, pix, tell_w=()):
    """Original ASAP: chi2 of each region = mean squared normalised residual of the
    fitted pixels inside [lo, hi] (all orders), without the pixels at tell_w."""
    tw = np.asarray(tell_w, float)
    for r in regions:
        tot, n = 0.0, 0
        for x, r2 in pix:
            i0, i1 = np.searchsorted(x, (r.lo, r.hi))
            v, xs = r2[i0:i1], x[i0:i1]
            ok = np.isfinite(v)
            if len(tw) and len(xs):             # pixels flagged by --tell do not count
                j = np.clip(np.searchsorted(tw, xs), 1, len(tw) - 1) if len(tw) > 1 else np.zeros(len(xs), int)
                near = np.minimum(np.abs(tw[j] - xs), np.abs(tw[np.maximum(j - 1, 0)] - xs))
                ok &= near > 1e-6
            tot += v[ok].sum(); n += int(ok.sum())
        r.chi2 = tot / n if n else None


def window_chi2(regions, windows):
    """linefit_shared / linefit: each region gets the chi2 of the fit window that
    contains its centre (the nearest window centre if several)."""
    win = np.array([w for w in windows if np.all(np.isfinite(w))], float)
    if len(win) == 0:
        return
    for r in regions:
        ok = np.where((win[:, 0] <= r.c) & (r.c <= win[:, 1]))[0]
        if len(ok):
            j = ok[np.argmin(np.abs(0.5 * (win[ok, 0] + win[ok, 1]) - r.c))]
            r.chi2 = float(win[j, 2])


def filter_linelist(ll, fit, chi2_max, tell=None, mode="above"):
    """No GUI: give every region its chi2, flag those ABOVE chi2_max (or below, with
    mode='below'); save() then comments them out. Returns (n_flagged, n_with_chi2,
    n_tell_pixels). tell = dict(sigma, width, frac, sign) removes telluric pixels
    from the chi2 (original ASAP output only)."""
    regs = ll.regions
    tw = np.zeros(0)
    if tell is not None:
        tw = tell_candidates(fit, regs, **tell)["w"]
    if fit.get("pix"):
        pixel_chi2(regs, fit["pix"], tw)
    else:
        window_chi2(regs, fit["chi2"])
    n = 0
    for r in regs:
        r.flag = bool(r.chi2 is not None and ((r.chi2 > chi2_max) if mode == "above"
                                              else (r.chi2 < chi2_max)))
        n += r.flag
    ll.flag_note = "(%s %.3g)" % (">" if mode == "above" else "<", chi2_max)
    return n, sum(r.chi2 is not None for r in regs), len(tw)


# ---------------------------------------------------------------------------
# Viewer / controller
# ---------------------------------------------------------------------------
class Editor:
    MAXPTS = 4000          # max points drawn per order ...
    BUDGET = 40000         # ... and in total (wide views of many orders)

    def __init__(self, ll, orders=(), outpath=None, width=0.1, fig=None,
                 xlim=None, fit=None, tell=None, vald=None):
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
        self.nrows = 0                 # rows of widgets under the axes
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
        self.fit = fit
        self.tell_show = True
        self.tell_enabled = fit is not None and tell is not None
        self.tell = dict(w=np.zeros(0), f=np.zeros(0))
        self.tell_on, = self.ax.plot([], [], "x", color="C3", ms=7, mew=1.8, zorder=5)
        self.tell_off, = self.ax.plot([], [], "o", mfc="none", mec="0.5", ms=6, mew=1.0, zorder=4)
        if self.tell_enabled:
            self.setup_tell(fit, **tell)
        if fit is not None and (fit.get("chi2") or fit.get("pix")):
            self.setup_chi2(fit["chi2"])          # after --tell: its pixels do not count
        self.vald = None
        if vald is not None:
            self.setup_vald(**vald)
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
        self.update_vald()
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
            nt = self.tell_count(r) if self.tell_show else 0
            lines = [str(n)]
            if r.chi2 is not None:
                lines.append(r"$\chi^2_r$ %.2f" % r.chi2)
            if nt:
                lines.append(u"\u26a0 %d" % nt)
            txt.set_text("\n".join(lines))
            if len(lines) > 1:
                ec = self.badge_color(r.chi2) if r.chi2 is not None else "C3"
                txt.set_bbox(dict(boxstyle="round,pad=0.25", fc="white", alpha=0.88,
                                  ec=ec, lw=1.2))
            else:
                txt.set_bbox(None)
        self.update_tell()
        self.cull()
        self.refresh_title()
        # self.sync()
        self.fig.canvas.draw_idle()

    def widget_row(self):
        """y of the next row of widgets under the axes; makes room for it."""
        y = 0.08 + 0.06 * self.nrows
        self.nrows += 1
        self.fig.subplots_adjust(bottom=0.21 + 0.06 * self.nrows)
        self.status.set_y(0.01)
        return y

    # -- VALD line identification ------------------------------------------------
    def setup_vald(self, path, rv=0.0, depth=0.2, nlabels=30, exclude=()):
        """Load the VALD lines covering the data / regions, shift them by the
        fitted RV and add the minimum-depth slider."""
        from matplotlib.collections import LineCollection
        from matplotlib.widgets import Slider
        spans = [(r.lo, r.hi) for r in self.ll.regions]
        for w, _ in self.orders:
            spans.append((w[0], w[-1]))
        if self.fitinfo is not None:
            for key in ("fit", "data"):
                spans += [(sg[0][0], sg[0][-1]) for sg in (self.fitinfo.get(key) or []) if sg is not None]
        if not spans:
            return
        lo, hi = min(a for a, b in spans) - 0.5, max(b for a, b in spans) + 0.5
        w, dep, sp = read_vald_file(path, lo / doppler(rv), hi / doppler(rv))
        if len(w) == 0:
            print("VALD: no lines of %s in the plotted range" % os.path.basename(path))
            return
        self.vw, self.vdep, self.vsp = w * doppler(rv), dep, sp      # shifted by the FITTED rv
        self.vthr, self.vshow, self.nlabels = float(depth), True, nlabels
        self.vexcl = species_match(sp, exclude)       # True = hidden when the exclusion is on
        self.vexcl_on = bool(len(exclude))
        self.vcoll = LineCollection([], colors="C1", linewidths=1.0, alpha=0.8,
                                    transform=self.tr, zorder=3)
        self.ax.add_collection(self.vcoll)
        self.vtexts = []
        y = self.widget_row()
        ax_s = self.fig.add_axes([0.14, y, 0.45, 0.03])
        dmax = self.vdepth_max()                     # slider spans the depths present
        self.vthr = min(self.vthr, dmax)
        self.vslider = Slider(ax_s, "min line depth", 0.0, dmax, valinit=self.vthr, color="C1")
        self.vslider.on_changed(self._on_vslider)
        self.vald = dict(path=path, rv=rv)
        print("VALD: %s, %d lines (%d excluded by species), shifted by RV %.3f km/s"
              % (os.path.basename(path), len(w), int(self.vexcl.sum()), rv))
        self.update_vald()

    def vdepth_max(self):
        d = self.vdep[~self.vexcl] if self.vexcl_on else self.vdep
        return max(float(d.max()) if len(d) else 0.0, 0.05)

    def vkeep(self):
        return ~self.vexcl if self.vexcl_on else np.ones(len(self.vw), bool)

    def step_vdepth(self, nsteps):
        """Move the minimum depth by nsteps of 2% of the slider range."""
        dmax = self.vdepth_max()
        self.vslider.set_val(float(np.clip(self.vthr + nsteps * dmax / 50.0, 0.0, dmax)))

    def toggle_vexcl(self):
        self.vexcl_on = not self.vexcl_on
        dmax = self.vdepth_max()                      # the slider range follows the visible lines
        self.vslider.valmax = dmax
        self.vslider.ax.set_xlim(0.0, dmax)
        self.vslider.set_val(min(self.vthr, dmax))
        self.update_vald()
        self.fig.canvas.draw_idle()

    def _on_vslider(self, v):
        self.vthr = float(v)
        self.update_vald()
        self.fig.canvas.draw_idle()

    def update_vald(self):
        if self.vald is None:
            return
        x0, x1 = self.ax.get_xlim()
        i0, i1 = np.searchsorted(self.vw, [x0, x1])
        ok = (self.vdep[i0:i1] >= self.vthr) & self.vkeep()[i0:i1]
        sel = np.flatnonzero(ok) + i0
        if not self.vshow:
            sel = sel[:0]
        w, d = self.vw[sel], self.vdep[sel]
        top = 0.08 + 0.30 * d                            # taller tick = deeper line
        self.vcoll.set_segments([[(x, 0.0), (x, t)] for x, t in zip(w, top)])
        keep = np.argsort(-d)[:self.nlabels] if len(d) > self.nlabels else np.arange(len(d))
        while len(self.vtexts) < len(keep):
            self.vtexts.append(self.ax.text(0, 0, "", transform=self.tr, rotation=90,
                                            ha="center", va="bottom", fontsize=6.5,
                                            color="C1", clip_on=True, zorder=6))
        for t in self.vtexts:
            t.set_visible(False)
        for t, j in zip(self.vtexts, keep):
            t.set_position((w[j], top[j] + 0.01))
            t.set_text("%s  %.3f" % (self.vsp[sel[j]], w[j]))
            t.set_visible(True)

    # -- telluric candidates -----------------------------------------------------
    def setup_tell(self, fit, sigma=4.0, width=3, frac=0.01, sign="neg"):
        """Find isolated absorption spikes (data below the best fit) in every fit
        window; see find_outliers."""
        self.tell = tell_candidates(fit, self.ll.regions, sigma, width, frac, sign)
        print("telluric candidates: %d pixel(s)" % len(self.tell["w"]))

    def tell_inside(self):
        """Boolean array: candidate lies inside some region of the (edited) list."""
        w = self.tell["w"]
        ins = np.zeros(len(w), bool)
        for r in self.ll.regions:
            ins |= (w >= r.lo) & (w <= r.hi)
        return ins

    def update_tell(self):
        t = self.tell
        ins = self.tell_inside()
        on, off = self.tell_show and ins, self.tell_show and ~ins
        self.tell_on.set_data(t["w"][on], t["f"][on])
        self.tell_off.set_data(t["w"][off], t["f"][off])

    def tell_count(self, r):
        w = self.tell["w"]
        if len(w) == 0:
            return 0
        return int(np.searchsorted(w, r.hi, side="right") - np.searchsorted(w, r.lo, side="left"))

    def next_tell(self, direction):
        w = self.tell["w"][self.tell_inside()]
        if len(w) == 0:
            self.say("no telluric candidate inside the regions")
            return
        x0, x1 = self.ax.get_xlim()
        mid = 0.5 * (x0 + x1)
        cand = w[(w - mid) * direction > 1e-9 * abs(mid)]
        if len(cand) == 0:
            self.say("no further candidates")
            return
        x = cand[0] if direction > 0 else cand[-1]
        half = 0.5 * (x1 - x0)
        self.ax.set_xlim(x - half, x + half)
        self.say("telluric candidate at %.4f nm" % x)

    # -- chi2 quality and flagging ---------------------------------------------
    def region_chi2(self):
        tw = self.tell["w"] if getattr(self, "tell_enabled", False) else ()
        pixel_chi2(self.ll.regions, self.pix, tw)

    def setup_chi2(self, windows):
        """Give each region the reduced chi2 of the fit window containing its
        centre, and build the threshold slider and mode button."""
        from matplotlib.widgets import Slider, Button
        self.pix = self.fit.get("pix") if isinstance(getattr(self, "fit", None), dict) else None
        if self.pix:
            self.region_chi2()
        else:
            window_chi2(self.ll.regions, windows)
        vals = np.array([r.chi2 for r in self.ll.regions if r.chi2 is not None])
        if len(vals) == 0:
            return
        self.has_chi2 = True
        lo = max(vals.min(), 1e-3) * 0.9
        hi = max(vals.max(), lo * 1.5) * 1.1
        y = self.widget_row()
        ax_s = self.fig.add_axes([0.14, y, 0.45, 0.03])
        self.slider = Slider(ax_s, r"$\chi^2_r$ threshold", np.log10(lo), np.log10(hi),
                             valinit=np.log10(hi), color="C3")
        self.slider.valtext.set_text("%.2f" % hi)
        self.slider.on_changed(self._on_slider)
        ax_b = self.fig.add_axes([0.66, y - 0.012, 0.24, 0.055])
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
        if getattr(self, "pix", None):
            self.region_chi2()                  # regions may have been added / moved
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
        elif k == "i":
            if self.vald is None:
                self.say("no VALD identification (needs --fit with a results folder)")
            else:
                self.vshow = not self.vshow
                self.say("VALD lines %s" % ("shown" if self.vshow else "hidden"))
                self.update_vald()
                self.fig.canvas.draw_idle()
        elif k in ("up", "down", "shift+up", "shift+down"):
            if self.vald is None:
                self.say("no VALD identification (needs --fit with a results folder)")
            else:
                self.step_vdepth((1 if k.endswith("up") else -1) * (5 if k.startswith("shift") else 1))
                self.say("minimum line depth %.2f" % self.vthr)
        elif k == "x":
            if self.vald is None or not self.vexcl.any():
                self.say("no species excluded (use --vald-exclude TiO ...)")
            else:
                self.toggle_vexcl()
                self.say("excluded species %s" % ("hidden" if self.vexcl_on else "shown"))
        elif k in ("t", "n", "N") and not self.tell_enabled:
            self.say("telluric detection is off (start with --tell)")
        elif k == "t":
            self.tell_show = not self.tell_show
            self.say("telluric candidates %s" % ("shown" if self.tell_show else "hidden"))
            self.sync()
        elif k in ("n", "N"):
            self.next_tell(+1 if k == "n" else -1)
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
    p.add_argument("--tell", action="store_true",
                   help="detect telluric candidates (isolated absorption spikes of "
                        "the data below the best fit); needs --fit")
    p.add_argument("--vald", default=None, metavar="FILE",
                   help="VALD line list to identify lines (default: the one of "
                        "asap/support_data closest to the results' Teff)")
    p.add_argument("--no-vald", action="store_true",
                   help="do not draw the VALD identification")
    p.add_argument("--vald-rv", type=float, default=None,
                   help="RV (km/s) to shift the VALD lines by; default: the fitted "
                        "'rv' of results.txt in the --fit folder (not the guessed RV)")
    p.add_argument("--vald-exclude", nargs="+", default=[], metavar="SPECIES",
                   help="species to hide, e.g. TiO OH 'Fe 1' (a bare name hides all "
                        "ionisation stages; key x toggles it)")
    p.add_argument("--vald-teff", type=float, default=None,
                   help="Teff selecting the VALD file (default: results.txt, else 4000)")
    p.add_argument("--vald-depth", type=float, default=0.2,
                   help="initial minimum line depth shown (0-1, default 0.2)")
    p.add_argument("--chi2-max", type=float, default=None, metavar="X",
                   help="NON-INTERACTIVE: no window. Comment out (keep, with a note) every "
                        "region whose chi2 is ABOVE X and write the list to --out (or over "
                        "the input, with a .bak copy). Needs --fit; --tell removes the "
                        "telluric pixels from the chi2 first")
    p.add_argument("--chi2-min", type=float, default=None, metavar="X",
                   help="NON-INTERACTIVE like --chi2-max, but comments out chi2 BELOW X")
    p.add_argument("--tell-sigma", type=float, default=4.0,
                   help="telluric candidates: deviation threshold in robust sigmas (4)")
    p.add_argument("--tell-width", type=int, default=3,
                   help="telluric candidates: max adjacent pixels in a spike (3)")
    p.add_argument("--tell-frac", type=float, default=0.01,
                   help="telluric candidates: at most this fraction of a window's pixels (0.01)")
    p.add_argument("--tell-sign", choices=["neg", "both"], default="neg",
                   help="neg: only data below the fit (absorption); both: either side")
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


def batch(args):
    """--chi2-max / --chi2-min: filter the list without opening a window."""
    if args.fit is None:
        sys.exit("--chi2-max / --chi2-min need --fit DIR")
    if args.chi2_max is not None and args.chi2_min is not None:
        sys.exit("use only one of --chi2-max and --chi2-min")
    ll = LineList(args.linelist)
    fit = load_fit(args.fit)
    tell = None
    if args.tell:
        tell = dict(sigma=args.tell_sigma, width=args.tell_width,
                    frac=args.tell_frac, sign=args.tell_sign)
    mode, thr = ("above", args.chi2_max) if args.chi2_max is not None else ("below", args.chi2_min)
    nfl, nchi, ntell = filter_linelist(ll, fit, thr, tell, mode)
    if nchi == 0:
        sys.exit("no region could be matched to the fit: nothing written")
    out = ll.save(args.out or args.linelist)
    chi = np.array([r.chi2 for r in ll.regions if r.chi2 is not None])
    print("%s output, %d regions (%d with a chi2; median %.2f, max %.2f)"
          % (fit["kind"], len(ll.regions), nchi, np.median(chi), chi.max()))
    if tell is not None:
        print("%d telluric pixel(s) left out of the chi2" % ntell)
    print("%d region(s) with chi2 %s %g commented out; wrote %s"
          % (nfl, ">" if mode == "above" else "<", thr, out))
    for r in ll.regions:
        if r.flag:
            print("   %.5f nm  chi2 %.2f" % (r.c, r.chi2))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.chi2_max is not None or args.chi2_min is not None:
        return batch(args)
    import matplotlib.pyplot as plt
    for name, drop in (("keymap.back", ("left",)), ("keymap.forward", ("right",))):
        plt.rcParams[name] = [k for k in plt.rcParams[name] if k not in drop]
    ll = LineList(args.linelist)
    orders = load_obs(args.obs, args.rv) if args.obs else []
    print("%d regions. Press ? in the window for the keys." % len(ll.regions))
    fit = load_fit(args.fit) if args.fit else None
    if fit is not None:
        print("Best fit from %s output (%d regions)." % (fit["kind"], len(fit["fit"])))
    tell = None
    if args.tell:
        tell = dict(sigma=args.tell_sigma, width=args.tell_width,
                    frac=args.tell_frac, sign=args.tell_sign)
    vald = None
    if fit is not None and not args.no_vald:
        folder = args.fit if os.path.isdir(args.fit) else os.path.dirname(args.fit) or "."
        rv = args.vald_rv if args.vald_rv is not None else read_results_scalar(folder, "rv")
        if rv is None:
            print("VALD: no 'rv' in %s/results.txt: lines are NOT shifted (use --vald-rv)" % folder)
            rv = 0.0
        path = args.vald
        if path is None:
            teff = args.vald_teff if args.vald_teff is not None else read_results_scalar(folder, "teff")
            files_ = vald_files()
            if files_:
                t = min(files_, key=lambda t: abs(t - (4000.0 if teff is None else teff)))
                path = files_[t]
            else:
                print("VALD: no vald-lines_*.txt in asap/support_data (use --vald FILE)")
        if path:
            excl = [n for item in args.vald_exclude for n in item.split(",") if n.strip()]
            vald = dict(path=path, rv=rv, depth=args.vald_depth, exclude=excl)
    ed = Editor(ll, orders, args.out, args.width, xlim=args.xlim, fit=fit,
                tell=tell, vald=vald)
    plt.show()


if __name__ == "__main__":
    main()