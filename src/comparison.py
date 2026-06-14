import numpy as np
import pandas as pd


def load_poggio_catalog(path):
    """
    Load the Poggio-style young giant catalog.
    """
    colnames = [
        "source_id", "glon_deg", "glat_deg",
        "dist_kpc", "edist_kpc", "plxcor_mas",
        "q_dist", "gmag"
    ]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=colnames, comment="#")

    m = (
        np.isfinite(df["glon_deg"]) &
        np.isfinite(df["glat_deg"]) &
        np.isfinite(df["dist_kpc"]) &
        (df["dist_kpc"] > 0)
    )
    df = df.loc[m].copy()
    return df


def galactic_to_suncentered_xyz(glon_deg, glat_deg, dist_kpc):
    """
    Convert Galactic (l,b,d) to Sun-centered Cartesian coordinates.
    Sun is at (0,0,0).
    """
    l = np.deg2rad(np.asarray(glon_deg, float))
    b = np.deg2rad(np.asarray(glat_deg, float))
    d = np.asarray(dist_kpc, float)

    X = d * np.cos(b) * np.cos(l)
    Y = d * np.cos(b) * np.sin(l)
    Z = d * np.sin(b)

    return X, Y, Z


def binned_median_xy(x, y, z, xedges, yedges, min_count=5):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[ok]
    y = y[ok]
    z = z[ok]

    nx = len(xedges) - 1
    ny = len(yedges) - 1

    ix = np.searchsorted(xedges, x, side="right") - 1
    iy = np.searchsorted(yedges, y, side="right") - 1
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    ix = ix[inside]
    iy = iy[inside]
    z = z[inside]

    N = np.zeros((ny, nx), dtype=int)
    np.add.at(N, (iy, ix), 1)

    Zmed = np.full((ny, nx), np.nan, float)
    bins = [[[] for _ in range(nx)] for __ in range(ny)]
    for j, i, val in zip(iy, ix, z):
        bins[j][i].append(val)

    for j in range(ny):
        for i in range(nx):
            if N[j, i] >= min_count:
                Zmed[j, i] = np.median(bins[j][i])

    return Zmed, N


def build_poggio_xy_medianZ_map(path, bin_kpc=0.25, min_per_bin=2, xy_lim=16.0):
    """
    Full pipeline for Poggio catalog:
      load catalog -> convert to XYZ -> compute median Z map.
    """
    df = load_poggio_catalog(path)

    X, Y, Z = galactic_to_suncentered_xyz(
        df["glon_deg"].to_numpy(),
        df["glat_deg"].to_numpy(),
        df["dist_kpc"].to_numpy()
    )

    xedges = np.arange(-xy_lim, xy_lim + bin_kpc, bin_kpc)
    yedges = np.arange(-xy_lim, xy_lim + bin_kpc, bin_kpc)

    Zmed, Nbin = binned_median_xy(X, Y, Z, xedges, yedges, min_count=min_per_bin)

    diagnostics = {
        "n_stars_used": len(df),
        "n_bins_kept": int(np.sum(np.isfinite(Zmed))),
    }

    result = {
        "df": df,
        "X": X,
        "Y": Y,
        "Z": Z,
        "Zmed": Zmed,
        "Nbin": Nbin,
        "xedges": xedges,
        "yedges": yedges,
    }

    return result, diagnostics

import numpy as np

from .comparison import load_poggio_catalog, galactic_to_suncentered_xyz


def binned_median_polar(r, th, val, r_edges, th_edges, min_count=5):
    r = np.asarray(r)
    th = np.asarray(th)
    val = np.asarray(val)

    ok = np.isfinite(r) & np.isfinite(th) & np.isfinite(val)
    r = r[ok]
    th = th[ok]
    val = val[ok]

    nr = len(r_edges) - 1
    nth = len(th_edges) - 1

    ir = np.searchsorted(r_edges, r, side="right") - 1
    it = np.searchsorted(th_edges, th, side="right") - 1

    inside = (ir >= 0) & (ir < nr) & (it >= 0) & (it < nth)
    ir = ir[inside]
    it = it[inside]
    val = val[inside]

    N = np.zeros((nr, nth), dtype=int)
    np.add.at(N, (ir, it), 1)

    bins = [[[] for _ in range(nth)] for __ in range(nr)]
    for rr, tt, vv in zip(ir, it, val):
        bins[rr][tt].append(vv)

    Vmed = np.full((nr, nth), np.nan, float)
    for rr in range(nr):
        for tt in range(nth):
            if N[rr, tt] >= min_count:
                Vmed[rr, tt] = np.median(bins[rr][tt])

    return Vmed, N


def build_poggio_polar_medianZ_map(path, r_edges, th_edges,
                                   R0=8.2,
                                   min_per_bin=5):
    """
    Load Poggio catalog and rebin median Z into the user's polar bins.
    """
    df = load_poggio_catalog(path)

    X, Y, Z = galactic_to_suncentered_xyz(
        df["glon_deg"].to_numpy(),
        df["glat_deg"].to_numpy(),
        df["dist_kpc"].to_numpy()
    )

    # Sun-centered -> Galactocentric-centered XY
    Xgc = X - R0
    Ygc = Y

    R = np.sqrt(Xgc**2 + Ygc**2)
    phi = np.mod(np.arctan2(Ygc, Xgc), 2*np.pi)

    Zmed_polar, N_polar = binned_median_polar(
        R, phi, Z,
        r_edges=r_edges,
        th_edges=th_edges,
        min_count=min_per_bin
    )

    result = {
        "df": df,
        "X": X,
        "Y": Y,
        "Z": Z,
        "Xgc": Xgc,
        "Ygc": Ygc,
        "R": R,
        "phi": phi,
        "Zmed_polar": Zmed_polar,
        "N_polar": N_polar,
    }

    diagnostics = {
        "n_stars_used": len(df),
        "n_bins_kept": int(np.sum(np.isfinite(Zmed_polar))),
    }

    return result, diagnostics

