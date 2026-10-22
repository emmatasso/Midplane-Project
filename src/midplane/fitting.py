import numpy as np

from scipy.interpolate import make_splrep
from scipy.optimize import minimize_scalar


def spline_fit_on_trend_splrep(
    z_centers,
    trend,
    cnt,
    sem=None,
    zmin=-1.5,
    zmax=1.5,
    min_cnt=5,
    min_bins=8,
    spline_k=2,
    spline_s=0.002,
    n_eval=400,
    max_allowed_minima=1,
    edge_buffer=0.15,
):
    """
    Same logic as spline_fit_on_trend, but uses make_splrep instead
    of UnivariateSpline.
    """

    zc = np.asarray(z_centers, float)
    tr = np.asarray(trend, float)
    cnt = np.asarray(cnt, int)

    if sem is None:
        use = (
            (zc >= zmin) & (zc <= zmax) &
            np.isfinite(tr) &
            (cnt >= min_cnt)
        )
        w = None
    else:
        sem = np.asarray(sem, float)
        use = (
            (zc >= zmin) & (zc <= zmax) &
            np.isfinite(tr) &
            (cnt >= min_cnt) &
            np.isfinite(sem) & (sem > 0)
        )

    if np.sum(use) < min_bins:
        return None, "too_few_bins", np.nan

    x = zc[use]
    y = tr[use]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if sem is not None:
        s_arr = sem[use][order]
        pos = s_arr[np.isfinite(s_arr) & (s_arr > 0)]

        if len(pos) == 0:
            return None, "fit_failed", np.nan

        # prevent extreme weights
        s_floor = max(np.nanpercentile(pos, 10) * 0.5, 1e-4)
        s_arr = np.maximum(s_arr, s_floor)

        w = 1.0 / s_arr
    else:
        w = None

    try:
        spl = make_splrep(x, y, w=w, k=spline_k, s=spline_s)
    except Exception:
        return None, "fit_failed", np.nan

    zgrid = np.linspace(zmin, zmax, n_eval)
    ygrid = spl(zgrid)

    # count minima
    is_local_min = (
        (ygrid[1:-1] < ygrid[:-2]) &
        (ygrid[1:-1] < ygrid[2:])
    )
    idx_local_min = np.where(is_local_min)[0] + 1
    n_local_min = len(idx_local_min)

    if n_local_min > max_allowed_minima:
        return {
            "spline": spl,
            "zgrid": zgrid,
            "ygrid": ygrid,
            "n_local_min": n_local_min,
        }, "multiple_minima", np.nan

    try:
        res = minimize_scalar(
            lambda zz: float(spl(zz)),
            bounds=(zmin, zmax),
            method="bounded"
        )
    except Exception:
        return None, "minimize_failed", np.nan

    if not res.success:
        return None, "minimize_failed", np.nan

    z0 = float(res.x)

    if (z0 <= zmin + edge_buffer) or (z0 >= zmax - edge_buffer):
        return {
            "spline": spl,
            "zgrid": zgrid,
            "ygrid": ygrid,
            "n_local_min": n_local_min,
        }, "minimum_at_edge", np.nan

    return {
        "spline": spl,
        "zgrid": zgrid,
        "ygrid": ygrid,
        "n_local_min": n_local_min,
    }, "ok", z0

def binned_mean_and_counts(Zs, Ys, z_edges):
    """
    Compute mean(Y) and count in each z-bin.
    """
    nb = len(z_edges) - 1
    mean = np.full(nb, np.nan, float)
    cnt  = np.zeros(nb, int)

    inds = np.searchsorted(z_edges, Zs, side="right") - 1
    ok = (inds >= 0) & (inds < nb) & np.isfinite(Zs) & np.isfinite(Ys)
    inds = inds[ok]
    Ys   = Ys[ok]

    for i in range(nb):
        m = (inds == i)
        cnt[i] = int(np.sum(m))
        if cnt[i] > 0:
            mean[i] = np.nanmean(Ys[m])

    return mean, cnt


def quad_fit_on_trend(z_centers, trend, cnt,
                      zmin=-1.5, zmax=1.5,
                      min_cnt=5, min_bins=8):
    """
    Fit y = a z^2 + b z + c to the trend within [zmin, zmax],
    requiring cnt >= min_cnt and finite trend values.

    Returns
    -------
    coeffs : tuple
        (a, b, c)
    status : str
        "ok", "too_few_bins", "bad_curvature", or "min_outside_window"
    z0 : float
        Location of the quadratic minimum if valid, else np.nan
    """
    zc  = np.asarray(z_centers, float)
    tr  = np.asarray(trend, float)
    cnt = np.asarray(cnt, int)

    m = (zc >= zmin) & (zc <= zmax) & np.isfinite(tr) & (cnt >= min_cnt)
    if np.sum(m) < min_bins:
        return (np.nan, np.nan, np.nan), "too_few_bins", np.nan

    x = zc[m]
    y = tr[m]

    a, b, c = np.polyfit(x, y, deg=2)

    if (not np.isfinite(a)) or a <= 0:
        return (a, b, c), "bad_curvature", np.nan

    z0 = -b / (2 * a)
    if not (zmin <= z0 <= zmax):
        return (a, b, c), "min_outside_window", np.nan

    return (a, b, c), "ok", float(z0)

import numpy as np
import pandas as pd


def compute_df_quad(R_star, Z_star, mgfe_star, segm, seg_plot, z_edges,
                    R_edges=None,
                    minN_seg_total=15,
                    min_bin_count=5,
                    z_fit_min=-1.5,
                    z_fit_max=1.5,
                    min_bins_for_fit=8):
    """
    Compute quadratic-fit z_min for all (annulus, segment).

    Returns
    -------
    df_quad : pandas.DataFrame
        Columns: Rmin, Rmax, seg, z_min_quad, status_quad
    """

    if R_edges is None:
        R_edges = np.arange(0.5, 15.5 + 0.5, 0.5)

    R_edges_kept = np.asarray(R_edges, float)
    if len(R_edges_kept) < 2:
        raise RuntimeError("Not enough annuli edges to compute z_min map.")

    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    rows = []
    for ir in range(len(R_edges_kept) - 1):
        Rmin_focus = float(R_edges_kept[ir])
        Rmax_focus = float(R_edges_kept[ir + 1])

        mR = (R_star >= Rmin_focus) & (R_star < Rmax_focus)
        Ntot = int(np.sum(mR))

        if Ntot == 0:
            for seg_id in seg_plot:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_quad=np.nan, status_quad="no_stars_annulus"
                ))
            continue

        for seg_id in seg_plot:
            mseg = mR & segm[seg_id]
            Nseg = int(np.sum(mseg))

            if Nseg < minN_seg_total:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_quad=np.nan, status_quad="too_few_stars"
                ))
                continue

            Zs = Z_star[mseg]
            Ys = mgfe_star[mseg]
            good = np.isfinite(Zs) & np.isfinite(Ys)
            Zs, Ys = Zs[good], Ys[good]

            mean_line, cnt_line = binned_mean_and_counts(Zs, Ys, z_edges)
            (a, b, c), st, z0 = quad_fit_on_trend(
                z_centers, mean_line, cnt_line,
                zmin=z_fit_min, zmax=z_fit_max,
                min_cnt=min_bin_count, min_bins=min_bins_for_fit
            )

            rows.append(dict(
                Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                z_min_quad=z0, status_quad=st
            ))

    df_quad = pd.DataFrame(rows)
    return df_quad

def build_zmin_polar_map(df_quad, crowd_bounds, seg_plot, seg_rest_id=None, zlim=0.6):
    """
    Build a polar heatmap array from df_quad.

    Returns
    -------
    Zmap : ndarray
    theta_edges : ndarray
    R_edges_kept : ndarray
    seg_plot_heat : list
    """
    # Decide which segments are plottable with crowd_bounds
    if seg_rest_id is not None:
        seg_plot_heat = [s for s in seg_plot if s != seg_rest_id]
    else:
        seg_plot_heat = list(seg_plot)

    n_th_from_bounds = len(crowd_bounds) - 1
    if len(seg_plot_heat) != n_th_from_bounds:
        seg_plot_heat = list(range(1, n_th_from_bounds + 1))

    n_th = len(seg_plot_heat)

    theta_edges = np.unwrap(np.array(crowd_bounds, float))
    for i in range(1, len(theta_edges)):
        while theta_edges[i] <= theta_edges[i - 1]:
            theta_edges[i] += 2 * np.pi

    if len(theta_edges) != n_th + 1:
        raise RuntimeError(
            f"theta_edges has len={len(theta_edges)} but needs n_th+1={n_th+1}. "
            "This means SEG_PLOT_heat doesn't match crowd_bounds."
        )

    R_edges_kept = np.array(
        sorted(set(df_quad["Rmin"].tolist() + df_quad["Rmax"].tolist())),
        float
    )
    if len(R_edges_kept) < 2:
        raise RuntimeError("Not enough annuli to build heatmap.")

    n_r = len(R_edges_kept) - 1
    Zmap_raw = np.full((n_r, n_th), np.nan, float)
    seg_to_col = {seg: j for j, seg in enumerate(seg_plot_heat)}

    for _, row in df_quad.iterrows():
        if row["status_quad"] != "ok":
            continue

        seg = int(row["seg"])
        if seg not in seg_to_col:
            continue

        zmin = float(row["z_min_quad"])
        if not np.isfinite(zmin):
            continue

        Rmin = float(row["Rmin"])
        ir = np.searchsorted(R_edges_kept, Rmin, side="right") - 1
        if not (0 <= ir < n_r):
            continue

        it = seg_to_col[seg]
        Zmap_raw[ir, it] = zmin

    Zmap = np.where(
        np.isfinite(Zmap_raw) & (np.abs(Zmap_raw) <= zlim),
        Zmap_raw,
        np.nan
    )

    return Zmap, theta_edges, R_edges_kept, seg_plot_heat

import numpy as np
import pandas as pd

from .models import poggio_m1_global_Zw


def build_model_map(R_all, phi_u, segm, seg_plot_heat, R_edges_kept):
    """
    Build polar map of mean Poggio warp model Zw in each (R, seg) bin.

    Returns
    -------
    Zmap : ndarray
    df_warp : DataFrame
    R_edges_plot : ndarray
    """
    Zw_star = np.full_like(R_all, np.nan, dtype=float)
    ok = np.isfinite(R_all) & np.isfinite(phi_u)
    Zw_star[ok] = poggio_m1_global_Zw(R_all[ok], phi_u[ok])

    rows = []
    for ir in range(len(R_edges_kept) - 1):
        Rmin = float(R_edges_kept[ir])
        Rmax = float(R_edges_kept[ir + 1])

        mR = (R_all >= Rmin) & (R_all < Rmax) & np.isfinite(R_all)

        if int(np.sum(mR)) == 0:
            for seg in seg_plot_heat:
                rows.append(dict(
                    Rmin=Rmin, Rmax=Rmax, seg=int(seg),
                    Zw_mean=np.nan, status="no_stars_annulus"
                ))
            continue

        for seg in seg_plot_heat:
            good = mR & segm[seg] & np.isfinite(Zw_star)
            if int(np.sum(good)) == 0:
                rows.append(dict(
                    Rmin=Rmin, Rmax=Rmax, seg=int(seg),
                    Zw_mean=np.nan, status="empty"
                ))
                continue

            rows.append(dict(
                Rmin=Rmin, Rmax=Rmax, seg=int(seg),
                Zw_mean=float(np.nanmean(Zw_star[good])),
                status="ok"
            ))

    df_warp = pd.DataFrame(rows)

    R_edges_plot = np.array(
        sorted(set(df_warp["Rmin"].tolist() + df_warp["Rmax"].tolist())),
        float
    )
    n_r = len(R_edges_plot) - 1
    n_th = len(seg_plot_heat)

    Zmap = np.full((n_r, n_th), np.nan, float)
    seg_to_col = {seg: j for j, seg in enumerate(seg_plot_heat)}

    for _, row in df_warp.iterrows():
        if row["status"] != "ok":
            continue

        Rmin = float(row["Rmin"])
        seg = int(row["seg"])
        Zw = float(row["Zw_mean"])

        ir = np.searchsorted(R_edges_plot, Rmin, side="right") - 1
        if (0 <= ir < n_r) and (seg in seg_to_col) and np.isfinite(Zw):
            Zmap[ir, seg_to_col[seg]] = Zw

    return Zmap, df_warp, R_edges_plot

def binned_mean_count_sem(Zs, Ys, z_edges):
    """
    Return mean(Y), count, and SEM(Y) in each z-bin.
    SEM = sample std / sqrt(N), requiring N >= 2.
    """
    nb = len(z_edges) - 1
    mean = np.full(nb, np.nan, float)
    cnt  = np.zeros(nb, int)
    sem  = np.full(nb, np.nan, float)

    inds = np.searchsorted(z_edges, Zs, side="right") - 1
    ok = (inds >= 0) & (inds < nb) & np.isfinite(Zs) & np.isfinite(Ys)
    inds = inds[ok]
    Ys   = Ys[ok]

    for i in range(nb):
        m = (inds == i)
        n = int(np.sum(m))
        cnt[i] = n
        if n > 0:
            ybin = Ys[m]
            mean[i] = np.nanmean(ybin)
            if n >= 2:
                s = np.nanstd(ybin, ddof=1)
                sem[i] = s / np.sqrt(n)

    return mean, cnt, sem

def quad_fit_on_trend_with_err(z_centers, trend, cnt, sem,
                               zmin=-1.5, zmax=1.5,
                               min_cnt=5, min_bins=8):
    """
    Weighted quadratic fit y = a z^2 + b z + c.
    Uses SEM on mean(Mg/Fe) per z-bin as fit uncertainty.

    Returns
    -------
    coeffs : tuple
        (a, b, c)
    status : str
    z0 : float
        Fitted minimum location
    sigma_z0 : float
        Propagated uncertainty in z0
    """
    zc  = np.asarray(z_centers, float)
    tr  = np.asarray(trend, float)
    cnt = np.asarray(cnt, int)
    sem = np.asarray(sem, float)

    m = (
        (zc >= zmin) & (zc <= zmax) &
        np.isfinite(tr) &
        (cnt >= min_cnt) &
        np.isfinite(sem) & (sem > 0)
    )

    if np.sum(m) < min_bins:
        return (np.nan, np.nan, np.nan), "too_few_bins", np.nan, np.nan

    x = zc[m]
    y = tr[m]
    s = sem[m]

    pos = s[np.isfinite(s) & (s > 0)]
    if len(pos) == 0:
        return (np.nan, np.nan, np.nan), "bad_sem", np.nan, np.nan

    s_floor = max(np.nanpercentile(pos, 10) * 0.25, 1e-4)
    s = np.maximum(s, s_floor)

    try:
        coeffs, cov = np.polyfit(x, y, deg=2, w=1.0 / s, cov=True)
    except Exception:
        return (np.nan, np.nan, np.nan), "fit_failed", np.nan, np.nan

    a, b, c = coeffs

    if (not np.isfinite(a)) or a <= 0:
        return (a, b, c), "bad_curvature", np.nan, np.nan

    z0 = -b / (2.0 * a)
    if not (zmin <= z0 <= zmax):
        return (a, b, c), "min_outside_window", np.nan, np.nan

    var_a  = cov[0, 0]
    var_b  = cov[1, 1]
    cov_ab = cov[0, 1]

    dz_da = b / (2.0 * a**2)
    dz_db = -1.0 / (2.0 * a)

    var_z0 = dz_da**2 * var_a + dz_db**2 * var_b + 2.0 * dz_da * dz_db * cov_ab

    if (not np.isfinite(var_z0)) or (var_z0 < 0):
        sigma_z0 = np.nan
    else:
        sigma_z0 = np.sqrt(var_z0)

    return (a, b, c), "ok", float(z0), float(sigma_z0)

def build_zmin_and_err_maps(R_use, Z_use, mgfe_use, segm, seg_plot_heat, R_edges_kept, z_edges,
                            minN_seg_total=15, min_bin_count=5,
                            z_fit_min=-1.5, z_fit_max=1.5, min_bins_for_fit=8,
                            zlim=0.6):
    """
    Build polar maps of:
      - quadratic-fit z_min
      - propagated uncertainty sigma(z_min)

    Returns
    -------
    Zmap : ndarray
    Emap : ndarray
    df_quad : DataFrame
    R_edges_plot : ndarray
    """
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    rows = []
    for ir in range(len(R_edges_kept) - 1):
        Rmin_focus = float(R_edges_kept[ir])
        Rmax_focus = float(R_edges_kept[ir + 1])

        mR = (R_use >= Rmin_focus) & (R_use < Rmax_focus)
        Ntot = int(np.sum(mR))

        if Ntot == 0:
            for seg_id in seg_plot_heat:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_quad=np.nan, z_min_err=np.nan,
                    status_quad="no_stars_annulus"
                ))
            continue

        for seg_id in seg_plot_heat:
            mseg = mR & segm[seg_id]
            good = mseg & np.isfinite(Z_use) & np.isfinite(mgfe_use)
            Nseg = int(np.sum(good))

            if Nseg < minN_seg_total:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_quad=np.nan, z_min_err=np.nan,
                    status_quad="too_few_stars"
                ))
                continue

            Zs = Z_use[good]
            Ys = mgfe_use[good]

            mean_line, cnt_line, sem_line = binned_mean_count_sem(Zs, Ys, z_edges)
            (_, _, _), st, z0, z0err = quad_fit_on_trend_with_err(
                z_centers, mean_line, cnt_line, sem_line,
                zmin=z_fit_min, zmax=z_fit_max,
                min_cnt=min_bin_count, min_bins=min_bins_for_fit
            )

            rows.append(dict(
                Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                z_min_quad=z0, z_min_err=z0err,
                status_quad=st
            ))

    df_quad = pd.DataFrame(rows)

    R_edges_plot = np.array(
        sorted(set(df_quad["Rmin"].tolist() + df_quad["Rmax"].tolist())),
        float
    )

    n_r = len(R_edges_plot) - 1
    n_th = len(seg_plot_heat)
    seg_to_col = {seg: j for j, seg in enumerate(seg_plot_heat)}

    Zmap_raw = np.full((n_r, n_th), np.nan, float)
    Emap_raw = np.full((n_r, n_th), np.nan, float)

    for _, row in df_quad.iterrows():
        if row["status_quad"] != "ok":
            continue

        seg = int(row["seg"])
        if seg not in seg_to_col:
            continue

        zmin = float(row["z_min_quad"])
        zerr = float(row["z_min_err"]) if np.isfinite(row["z_min_err"]) else np.nan

        if not np.isfinite(zmin):
            continue

        Rmin = float(row["Rmin"])
        ir2 = np.searchsorted(R_edges_plot, Rmin, side="right") - 1
        if not (0 <= ir2 < n_r):
            continue

        it = seg_to_col[seg]
        Zmap_raw[ir2, it] = zmin
        Emap_raw[ir2, it] = zerr

    Zmap = np.where(np.isfinite(Zmap_raw) & (np.abs(Zmap_raw) <= zlim), Zmap_raw, np.nan)
    Emap = np.where(np.isfinite(Zmap), Emap_raw, np.nan)

    return Zmap, Emap, df_quad, R_edges_plot

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar


def spline_fit_on_trend(
    z_centers,
    trend,
    cnt,
    sem=None,
    zmin=-1.5,
    zmax=1.5,
    min_cnt=5,
    min_bins=8,
    spline_k=3,
    spline_s=None,
    n_eval=400,
    max_allowed_minima=1,
    edge_buffer=0.05,
):
    """
    Fit a smoothing spline to trend(z), then find the minimum inside [zmin, zmax].

    Parameters
    ----------
    z_centers : array
    trend : array
        Mean Mg/Fe per z-bin
    cnt : array
        Count per z-bin
    sem : array or None
        SEM per z-bin; if provided, use weights = 1/sem
    zmin, zmax : float
        Fit / search window
    min_cnt : int
        Minimum stars per z-bin to include
    min_bins : int
        Minimum number of valid bins required
    spline_k : int
        Spline degree (3 = cubic)
    spline_s : float or None
        Smoothing factor passed to UnivariateSpline.
        If None, SciPy chooses its default smoothing behavior.
    n_eval : int
        Number of grid points for post-fit checks
    max_allowed_minima : int
        Reject if spline has more than this many local minima in the window
    edge_buffer : float
        Reject minima too close to zmin/zmax by this amount

    Returns
    -------
    fit_obj : dict
        Contains spline object and dense-grid info
    status : str
        "ok", "too_few_bins", "fit_failed", "multiple_minima",
        "minimum_at_edge", or "minimize_failed"
    z0 : float
        Inferred minimum location
    """
    zc = np.asarray(z_centers, float)
    tr = np.asarray(trend, float)
    cnt = np.asarray(cnt, int)

    if sem is None:
        use = (
            (zc >= zmin) & (zc <= zmax) &
            np.isfinite(tr) &
            (cnt >= min_cnt)
        )
        w = None
    else:
        sem = np.asarray(sem, float)
        use = (
            (zc >= zmin) & (zc <= zmax) &
            np.isfinite(tr) &
            (cnt >= min_cnt) &
            np.isfinite(sem) & (sem > 0)
        )

    if np.sum(use) < min_bins:
        return None, "too_few_bins", np.nan

    x = zc[use]
    y = tr[use]

    # x must be strictly increasing for spline fitting
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if sem is not None:
        s = sem[use][order]
        # small floor for numerical stability
        pos = s[np.isfinite(s) & (s > 0)]
        if len(pos) == 0:
            return None, "fit_failed", np.nan
        s_floor = max(np.nanpercentile(pos, 10) * 0.25, 1e-4)
        s = np.maximum(s, s_floor)
        w = 1.0 / s
    else:
        w = None

    try:
        spl = UnivariateSpline(x, y, w=w, k=spline_k, s=spline_s)
    except Exception:
        return None, "fit_failed", np.nan

    # Dense-grid post-check for multiple minima / edge behavior
    zgrid = np.linspace(zmin, zmax, n_eval)
    ygrid = spl(zgrid)

    # local minima on dense grid
    is_local_min = (
        (ygrid[1:-1] < ygrid[:-2]) &
        (ygrid[1:-1] < ygrid[2:])
    )
    idx_local_min = np.where(is_local_min)[0] + 1
    n_local_min = len(idx_local_min)

    if n_local_min == 0:
        # still try bounded minimization; could be very broad/flat
        pass
    elif n_local_min > max_allowed_minima:
        fit_obj = {"spline": spl, "zgrid": zgrid, "ygrid": ygrid, "n_local_min": n_local_min}
        return fit_obj, "multiple_minima", np.nan

    try:
        res = minimize_scalar(
            lambda zz: float(spl(zz)),
            bounds=(zmin, zmax),
            method="bounded"
        )
    except Exception:
        return {"spline": spl, "zgrid": zgrid, "ygrid": ygrid, "n_local_min": n_local_min}, "minimize_failed", np.nan

    if not res.success:
        return {"spline": spl, "zgrid": zgrid, "ygrid": ygrid, "n_local_min": n_local_min}, "minimize_failed", np.nan

    z0 = float(res.x)

    if (z0 <= zmin + edge_buffer) or (z0 >= zmax - edge_buffer):
        fit_obj = {"spline": spl, "zgrid": zgrid, "ygrid": ygrid, "n_local_min": n_local_min}
        return fit_obj, "minimum_at_edge", np.nan

    fit_obj = {
        "spline": spl,
        "zgrid": zgrid,
        "ygrid": ygrid,
        "n_local_min": n_local_min,
    }
    return fit_obj, "ok", z0


def compute_df_spline(
    R_star,
    Z_star,
    mgfe_star,
    segm,
    seg_plot,
    z_edges,
    R_edges=None,
    minN_seg_total=15,
    min_bin_count=5,
    z_fit_min=-1.5,
    z_fit_max=1.5,
    min_bins_for_fit=8,
    use_sem_weights=True,
    spline_k=3,
    spline_s=None,
    n_eval=400,
    max_allowed_minima=1,
    edge_buffer=0.05,
):
    """
    Compute spline-fit z_min for all (annulus, segment).

    Returns
    -------
    df_spline : pandas.DataFrame
        Columns: Rmin, Rmax, seg, z_min_spline, status_spline
    """
    if R_edges is None:
        R_edges = np.arange(0.5, 15.5 + 0.5, 0.5)

    R_edges_kept = np.asarray(R_edges, float)
    if len(R_edges_kept) < 2:
        raise RuntimeError("Not enough annuli edges to compute z_min map.")

    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    rows = []
    for ir in range(len(R_edges_kept) - 1):
        Rmin_focus = float(R_edges_kept[ir])
        Rmax_focus = float(R_edges_kept[ir + 1])

        mR = (R_star >= Rmin_focus) & (R_star < Rmax_focus)
        Ntot = int(np.sum(mR))

        if Ntot == 0:
            for seg_id in seg_plot:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_spline=np.nan, status_spline="no_stars_annulus"
                ))
            continue

        for seg_id in seg_plot:
            mseg = mR & segm[seg_id]
            good = mseg & np.isfinite(Z_star) & np.isfinite(mgfe_star)
            Nseg = int(np.sum(good))

            if Nseg < minN_seg_total:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_spline=np.nan, status_spline="too_few_stars"
                ))
                continue

            Zs = Z_star[good]
            Ys = mgfe_star[good]

            if use_sem_weights:
                mean_line, cnt_line, sem_line = binned_mean_count_sem(Zs, Ys, z_edges)
                fit_obj, st, z0 = spline_fit_on_trend(
                    z_centers, mean_line, cnt_line, sem=sem_line,
                    zmin=z_fit_min, zmax=z_fit_max,
                    min_cnt=min_bin_count, min_bins=min_bins_for_fit,
                    spline_k=spline_k, spline_s=spline_s,
                    n_eval=n_eval,
                    max_allowed_minima=max_allowed_minima,
                    edge_buffer=edge_buffer,
                )
            else:
                mean_line, cnt_line = binned_mean_and_counts(Zs, Ys, z_edges)
                fit_obj, st, z0 = spline_fit_on_trend(
                    z_centers, mean_line, cnt_line, sem=None,
                    zmin=z_fit_min, zmax=z_fit_max,
                    min_cnt=min_bin_count, min_bins=min_bins_for_fit,
                    spline_k=spline_k, spline_s=spline_s,
                    n_eval=n_eval,
                    max_allowed_minima=max_allowed_minima,
                    edge_buffer=edge_buffer,
                )

            rows.append(dict(
                Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                z_min_spline=z0, status_spline=st
            ))

    return pd.DataFrame(rows)

def compute_df_spline_splrep(
    R_star,
    Z_star,
    mgfe_star,
    segm,
    seg_plot,
    z_edges,
    R_edges=None,
    minN_seg_total=15,
    min_bin_count=5,
    z_fit_min=-1.5,
    z_fit_max=1.5,
    min_bins_for_fit=8,
    use_sem_weights=True,
    spline_k=2,
    spline_s=0.05,
    n_eval=400,
    max_allowed_minima=1,
    edge_buffer=0.15,
):
    """
    Compute spline-fit z_min for all (annulus, segment).

    Returns
    -------
    df_spline : pandas.DataFrame
        Columns: Rmin, Rmax, seg, z_min_spline, status_spline
    """
    if R_edges is None:
        R_edges = np.arange(0.5, 15.5 + 0.5, 0.5)

    R_edges_kept = np.asarray(R_edges, float)
    if len(R_edges_kept) < 2:
        raise RuntimeError("Not enough annuli edges to compute z_min map.")

    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    rows = []
    for ir in range(len(R_edges_kept) - 1):
        Rmin_focus = float(R_edges_kept[ir])
        Rmax_focus = float(R_edges_kept[ir + 1])

        mR = (R_star >= Rmin_focus) & (R_star < Rmax_focus)
        Ntot = int(np.sum(mR))

        if Ntot == 0:
            for seg_id in seg_plot:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_spline=np.nan, status_spline="no_stars_annulus"
                ))
            continue

        for seg_id in seg_plot:
            mseg = mR & segm[seg_id]
            good = mseg & np.isfinite(Z_star) & np.isfinite(mgfe_star)
            Nseg = int(np.sum(good))

            if Nseg < minN_seg_total:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_spline=np.nan, status_spline="too_few_stars"
                ))
                continue

            Zs = Z_star[good]
            Ys = mgfe_star[good]

            if use_sem_weights:
                mean_line, cnt_line, sem_line = binned_mean_count_sem(Zs, Ys, z_edges)
                fit_obj, st, z0 = spline_fit_on_trend_splrep(
                    z_centers, mean_line, cnt_line, sem=sem_line,
                    zmin=z_fit_min, zmax=z_fit_max,
                    min_cnt=min_bin_count, min_bins=min_bins_for_fit,
                    spline_k=spline_k, spline_s=spline_s,
                    n_eval=n_eval,
                    max_allowed_minima=max_allowed_minima,
                    edge_buffer=edge_buffer,
                )
            else:
                mean_line, cnt_line = binned_mean_and_counts(Zs, Ys, z_edges)
                fit_obj, st, z0 = spline_fit_on_trend_splrep(
                    z_centers, mean_line, cnt_line, sem=None,
                    zmin=z_fit_min, zmax=z_fit_max,
                    min_cnt=min_bin_count, min_bins=min_bins_for_fit,
                    spline_k=spline_k, spline_s=spline_s,
                    n_eval=n_eval,
                    max_allowed_minima=max_allowed_minima,
                    edge_buffer=edge_buffer,
                )

            rows.append(dict(
                Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                z_min_spline=z0, status_spline=st
            ))

    return pd.DataFrame(rows)


def build_spline_polar_map(df_spline, crowd_bounds, seg_plot, seg_rest_id=None, zlim=0.6):
    """
    Build a polar heatmap array from df_spline.
    """
    if seg_rest_id is not None:
        seg_plot_heat = [s for s in seg_plot if s != seg_rest_id]
    else:
        seg_plot_heat = list(seg_plot)

    n_th_from_bounds = len(crowd_bounds) - 1
    if len(seg_plot_heat) != n_th_from_bounds:
        seg_plot_heat = list(range(1, n_th_from_bounds + 1))

    n_th = len(seg_plot_heat)

    theta_edges = np.unwrap(np.array(crowd_bounds, float))
    for i in range(1, len(theta_edges)):
        while theta_edges[i] <= theta_edges[i - 1]:
            theta_edges[i] += 2 * np.pi

    if len(theta_edges) != n_th + 1:
        raise RuntimeError(
            f"theta_edges has len={len(theta_edges)} but needs n_th+1={n_th+1}."
        )

    R_edges_kept = np.array(
        sorted(set(df_spline["Rmin"].tolist() + df_spline["Rmax"].tolist())),
        float
    )
    if len(R_edges_kept) < 2:
        raise RuntimeError("Not enough annuli to build heatmap.")

    n_r = len(R_edges_kept) - 1
    Zmap_raw = np.full((n_r, n_th), np.nan, float)
    seg_to_col = {seg: j for j, seg in enumerate(seg_plot_heat)}

    for _, row in df_spline.iterrows():
        if row["status_spline"] != "ok":
            continue

        seg = int(row["seg"])
        if seg not in seg_to_col:
            continue

        zmin = float(row["z_min_spline"])
        if not np.isfinite(zmin):
            continue

        Rmin = float(row["Rmin"])
        ir = np.searchsorted(R_edges_kept, Rmin, side="right") - 1
        if not (0 <= ir < n_r):
            continue

        it = seg_to_col[seg]
        Zmap_raw[ir, it] = zmin

    Zmap = np.where(np.isfinite(Zmap_raw) & (np.abs(Zmap_raw) <= zlim), Zmap_raw, np.nan)

    return Zmap, theta_edges, R_edges_kept, seg_plot_heat

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize_scalar
import numpy as np
import pandas as pd


def interp_spline_fit_on_trend(
    z_centers,
    trend,
    cnt,
    zmin=-1.5,
    zmax=1.5,
    min_cnt=5,
    min_bins=8,
    spline_k=3,
    n_eval=400,
    max_allowed_minima=1,
    edge_buffer=0.05,
):
    """
    Fit an interpolating spline to trend(z), then find the minimum inside [zmin, zmax].

    Returns
    -------
    fit_obj : dict
    status : str
    z0 : float
    """
    zc = np.asarray(z_centers, float)
    tr = np.asarray(trend, float)
    cnt = np.asarray(cnt, int)

    use = (
        (zc >= zmin) & (zc <= zmax) &
        np.isfinite(tr) &
        (cnt >= min_cnt)
    )

    if np.sum(use) < min_bins:
        return None, "too_few_bins", np.nan

    x = zc[use]
    y = tr[use]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # remove duplicate x values just in case
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]

    if len(x_unique) < max(min_bins, spline_k + 1):
        return None, "too_few_bins", np.nan

    try:
        spl = InterpolatedUnivariateSpline(x_unique, y_unique, k=spline_k)
    except Exception:
        return None, "fit_failed", np.nan

    zgrid = np.linspace(zmin, zmax, n_eval)
    ygrid = spl(zgrid)

    is_local_min = (
        (ygrid[1:-1] < ygrid[:-2]) &
        (ygrid[1:-1] < ygrid[2:])
    )
    idx_local_min = np.where(is_local_min)[0] + 1
    n_local_min = len(idx_local_min)

    if n_local_min > max_allowed_minima:
        fit_obj = {"spline": spl, "zgrid": zgrid, "ygrid": ygrid, "n_local_min": n_local_min}
        return fit_obj, "multiple_minima", np.nan

    try:
        res = minimize_scalar(
            lambda zz: float(spl(zz)),
            bounds=(zmin, zmax),
            method="bounded"
        )
    except Exception:
        return {"spline": spl, "zgrid": zgrid, "ygrid": ygrid, "n_local_min": n_local_min}, "minimize_failed", np.nan

    if not res.success:
        return {"spline": spl, "zgrid": zgrid, "ygrid": ygrid, "n_local_min": n_local_min}, "minimize_failed", np.nan

    z0 = float(res.x)

    if (z0 <= zmin + edge_buffer) or (z0 >= zmax - edge_buffer):
        fit_obj = {"spline": spl, "zgrid": zgrid, "ygrid": ygrid, "n_local_min": n_local_min}
        return fit_obj, "minimum_at_edge", np.nan

    fit_obj = {"spline": spl, "zgrid": zgrid, "ygrid": ygrid, "n_local_min": n_local_min}
    return fit_obj, "ok", z0


def compute_df_interp_spline(
    R_star,
    Z_star,
    mgfe_star,
    segm,
    seg_plot,
    z_edges,
    R_edges=None,
    minN_seg_total=15,
    min_bin_count=5,
    z_fit_min=-1.5,
    z_fit_max=1.5,
    min_bins_for_fit=8,
    spline_k=3,
    n_eval=400,
    max_allowed_minima=1,
    edge_buffer=0.05,
):
    """
    Compute interpolating-spline z_min for all (annulus, segment).
    """
    if R_edges is None:
        R_edges = np.arange(0.5, 15.5 + 0.5, 0.5)

    R_edges_kept = np.asarray(R_edges, float)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    rows = []
    for ir in range(len(R_edges_kept) - 1):
        Rmin_focus = float(R_edges_kept[ir])
        Rmax_focus = float(R_edges_kept[ir + 1])

        mR = (R_star >= Rmin_focus) & (R_star < Rmax_focus)
        Ntot = int(np.sum(mR))

        if Ntot == 0:
            for seg_id in seg_plot:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_interp=np.nan, status_interp="no_stars_annulus"
                ))
            continue

        for seg_id in seg_plot:
            mseg = mR & segm[seg_id]
            good = mseg & np.isfinite(Z_star) & np.isfinite(mgfe_star)
            Nseg = int(np.sum(good))

            if Nseg < minN_seg_total:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_interp=np.nan, status_interp="too_few_stars"
                ))
                continue

            Zs = Z_star[good]
            Ys = mgfe_star[good]

            mean_line, cnt_line = binned_mean_and_counts(Zs, Ys, z_edges)
            fit_obj, st, z0 = interp_spline_fit_on_trend(
                z_centers, mean_line, cnt_line,
                zmin=z_fit_min, zmax=z_fit_max,
                min_cnt=min_bin_count, min_bins=min_bins_for_fit,
                spline_k=spline_k,
                n_eval=n_eval,
                max_allowed_minima=max_allowed_minima,
                edge_buffer=edge_buffer,
            )

            rows.append(dict(
                Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                z_min_interp=z0, status_interp=st
            ))

    return pd.DataFrame(rows)


def build_interp_polar_map(df_interp, crowd_bounds, seg_plot, seg_rest_id=None, zlim=0.6):
    """
    Build polar heatmap from interpolating-spline results.
    """
    if seg_rest_id is not None:
        seg_plot_heat = [s for s in seg_plot if s != seg_rest_id]
    else:
        seg_plot_heat = list(seg_plot)

    n_th_from_bounds = len(crowd_bounds) - 1
    if len(seg_plot_heat) != n_th_from_bounds:
        seg_plot_heat = list(range(1, n_th_from_bounds + 1))

    theta_edges = np.unwrap(np.array(crowd_bounds, float))
    for i in range(1, len(theta_edges)):
        while theta_edges[i] <= theta_edges[i - 1]:
            theta_edges[i] += 2 * np.pi

    R_edges_kept = np.array(
        sorted(set(df_interp["Rmin"].tolist() + df_interp["Rmax"].tolist())),
        float
    )

    n_r = len(R_edges_kept) - 1
    n_th = len(seg_plot_heat)
    Zmap_raw = np.full((n_r, n_th), np.nan, float)
    seg_to_col = {seg: j for j, seg in enumerate(seg_plot_heat)}

    for _, row in df_interp.iterrows():
        if row["status_interp"] != "ok":
            continue

        seg = int(row["seg"])
        if seg not in seg_to_col:
            continue

        zmin = float(row["z_min_interp"])
        if not np.isfinite(zmin):
            continue

        Rmin = float(row["Rmin"])
        ir = np.searchsorted(R_edges_kept, Rmin, side="right") - 1
        if not (0 <= ir < n_r):
            continue

        Zmap_raw[ir, seg_to_col[seg]] = zmin

    Zmap = np.where(np.isfinite(Zmap_raw) & (np.abs(Zmap_raw) <= zlim), Zmap_raw, np.nan)
    return Zmap, theta_edges, R_edges_kept, seg_plot_heat

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar


def binned_weighted_mean_and_sem(z, y, sigma_y, z_edges):
    """
    Compute inverse-variance weighted mean and uncertainty of the mean
    in each z-bin.

    Parameters
    ----------
    z : array
        z coordinate for each star
    y : array
        quantity to average (here [Mg/Fe])
    sigma_y : array
        per-star uncertainty on y
    z_edges : array
        bin edges in z

    Returns
    -------
    mean : array
        weighted mean in each bin
    cnt : array
        number of stars in each bin
    sem : array
        propagated uncertainty on the weighted mean in each bin
    """
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    sigma_y = np.asarray(sigma_y, float)

    nb = len(z_edges) - 1
    mean = np.full(nb, np.nan, float)
    cnt = np.zeros(nb, int)
    sem = np.full(nb, np.nan, float)

    inds = np.searchsorted(z_edges, z, side="right") - 1
    ok = (
        (inds >= 0) & (inds < nb) &
        np.isfinite(z) &
        np.isfinite(y) &
        np.isfinite(sigma_y) &
        (sigma_y > 0)
    )

    inds = inds[ok]
    y = y[ok]
    sigma_y = sigma_y[ok]

    for i in range(nb):
        m = (inds == i)
        n = int(np.sum(m))
        cnt[i] = n

        if n == 0:
            continue

        y_i = y[m]
        s_i = sigma_y[m]

        w = 1.0 / s_i**2
        wsum = np.sum(w)

        if not np.isfinite(wsum) or wsum <= 0:
            continue

        mean[i] = np.sum(w * y_i) / wsum
        sem[i] = np.sqrt(1.0 / wsum)

    return mean, cnt, sem


def spline_fit_on_trend(
    z_centers,
    trend,
    cnt,
    sem=None,
    zmin=-1.5,
    zmax=1.5,
    min_cnt=5,
    min_bins=8,
    spline_k=3,
    spline_s=None,
    n_eval=400,
    max_allowed_minima=1,
    edge_buffer=0.05,
):
    """
    Fit a smoothing spline to trend(z), then find the minimum inside [zmin, zmax].

    If sem is provided, use weights = 1/sem.
    """
    zc = np.asarray(z_centers, float)
    tr = np.asarray(trend, float)
    cnt = np.asarray(cnt, int)

    if sem is None:
        use = (
            (zc >= zmin) & (zc <= zmax) &
            np.isfinite(tr) &
            (cnt >= min_cnt)
        )
        w = None
    else:
        sem = np.asarray(sem, float)
        use = (
            (zc >= zmin) & (zc <= zmax) &
            np.isfinite(tr) &
            (cnt >= min_cnt) &
            np.isfinite(sem) & (sem > 0)
        )

    if np.sum(use) < min_bins:
        return None, "too_few_bins", np.nan

    x = zc[use]
    y = tr[use]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if sem is not None:
        s = sem[use][order]
        pos = s[np.isfinite(s) & (s > 0)]
        if len(pos) == 0:
            return None, "bad_sem", np.nan

        # numerical stability floor
        s_floor = max(np.nanpercentile(pos, 10) * 0.25, 1e-4)
        s = np.maximum(s, s_floor)
        w = 1.0 / s
    else:
        w = None

    try:
        spl = UnivariateSpline(x, y, w=w, k=spline_k, s=spline_s)
    except Exception:
        return None, "fit_failed", np.nan

    zgrid = np.linspace(zmin, zmax, n_eval)
    ygrid = spl(zgrid)

    # count local minima on a dense grid
    is_local_min = (
        (ygrid[1:-1] < ygrid[:-2]) &
        (ygrid[1:-1] < ygrid[2:])
    )
    idx_local_min = np.where(is_local_min)[0] + 1
    n_local_min = len(idx_local_min)

    if n_local_min > max_allowed_minima:
        fit_obj = {
            "spline": spl,
            "zgrid": zgrid,
            "ygrid": ygrid,
            "n_local_min": n_local_min,
        }
        return fit_obj, "multiple_minima", np.nan

    try:
        res = minimize_scalar(
            lambda zz: float(spl(zz)),
            bounds=(zmin, zmax),
            method="bounded"
        )
    except Exception:
        fit_obj = {
            "spline": spl,
            "zgrid": zgrid,
            "ygrid": ygrid,
            "n_local_min": n_local_min,
        }
        return fit_obj, "minimize_failed", np.nan

    if not res.success:
        fit_obj = {
            "spline": spl,
            "zgrid": zgrid,
            "ygrid": ygrid,
            "n_local_min": n_local_min,
        }
        return fit_obj, "minimize_failed", np.nan

    z0 = float(res.x)

    if (z0 <= zmin + edge_buffer) or (z0 >= zmax - edge_buffer):
        fit_obj = {
            "spline": spl,
            "zgrid": zgrid,
            "ygrid": ygrid,
            "n_local_min": n_local_min,
        }
        return fit_obj, "minimum_at_edge", np.nan

    fit_obj = {
        "spline": spl,
        "zgrid": zgrid,
        "ygrid": ygrid,
        "n_local_min": n_local_min,
    }
    return fit_obj, "ok", z0


def compute_df_spline_measerr(
    R_star,
    Z_star,
    mgfe_star,
    e_mg_h_star,
    e_fe_h_star,
    segm,
    seg_plot,
    z_edges,
    R_edges=None,
    minN_seg_total=15,
    min_bin_count=5,
    z_fit_min=-1.5,
    z_fit_max=1.5,
    min_bins_for_fit=8,
    spline_k=3,
    spline_s=None,
    n_eval=400,
    max_allowed_minima=1,
    edge_buffer=0.05,
):
    """
    Compute smoothing-spline z_min for all (annulus, segment),
    using propagated measurement uncertainties from e_mg_h and e_fe_h.

    Returns
    -------
    df_spline : pandas.DataFrame
        Columns: Rmin, Rmax, seg, z_min_spline, status_spline
    """
    if R_edges is None:
        R_edges = np.arange(0.5, 15.5 + 0.5, 0.5)

    R_edges_kept = np.asarray(R_edges, float)
    if len(R_edges_kept) < 2:
        raise RuntimeError("Not enough annuli edges to compute z_min map.")

    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    # propagate per-star Mg/Fe uncertainty
    sigma_mgfe_star = np.sqrt(
        np.asarray(e_mg_h_star, float)**2 +
        np.asarray(e_fe_h_star, float)**2
    )

    rows = []
    for ir in range(len(R_edges_kept) - 1):
        Rmin_focus = float(R_edges_kept[ir])
        Rmax_focus = float(R_edges_kept[ir + 1])

        mR = (R_star >= Rmin_focus) & (R_star < Rmax_focus)
        Ntot = int(np.sum(mR))

        if Ntot == 0:
            for seg_id in seg_plot:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_spline=np.nan, status_spline="no_stars_annulus"
                ))
            continue

        for seg_id in seg_plot:
            mseg = mR & segm[seg_id]
            good = (
                mseg &
                np.isfinite(Z_star) &
                np.isfinite(mgfe_star) &
                np.isfinite(sigma_mgfe_star) &
                (sigma_mgfe_star > 0)
            )

            Nseg = int(np.sum(good))

            if Nseg < minN_seg_total:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_spline=np.nan, status_spline="too_few_stars"
                ))
                continue

            Zs = Z_star[good]
            Ys = mgfe_star[good]
            sigs = sigma_mgfe_star[good]

            mean_line, cnt_line, sem_line = binned_weighted_mean_and_sem(
                Zs, Ys, sigs, z_edges
            )

            fit_obj, st, z0 = spline_fit_on_trend(
                z_centers=z_centers,
                trend=mean_line,
                cnt=cnt_line,
                sem=sem_line,
                zmin=z_fit_min,
                zmax=z_fit_max,
                min_cnt=min_bin_count,
                min_bins=min_bins_for_fit,
                spline_k=spline_k,
                spline_s=spline_s,
                n_eval=n_eval,
                max_allowed_minima=max_allowed_minima,
                edge_buffer=edge_buffer,
            )

            rows.append(dict(
                Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                z_min_spline=z0, status_spline=st
            ))

    return pd.DataFrame(rows)


def build_spline_polar_map(df_spline, crowd_bounds, seg_plot, seg_rest_id=None, zlim=0.6):
    """
    Build a polar heatmap array from df_spline.
    """
    if seg_rest_id is not None:
        seg_plot_heat = [s for s in seg_plot if s != seg_rest_id]
    else:
        seg_plot_heat = list(seg_plot)

    n_th_from_bounds = len(crowd_bounds) - 1
    if len(seg_plot_heat) != n_th_from_bounds:
        seg_plot_heat = list(range(1, n_th_from_bounds + 1))

    n_th = len(seg_plot_heat)

    theta_edges = np.unwrap(np.array(crowd_bounds, float))
    for i in range(1, len(theta_edges)):
        while theta_edges[i] <= theta_edges[i - 1]:
            theta_edges[i] += 2 * np.pi

    if len(theta_edges) != n_th + 1:
        raise RuntimeError(
            f"theta_edges has len={len(theta_edges)} but needs n_th+1={n_th+1}."
        )

    R_edges_kept = np.array(
        sorted(set(df_spline["Rmin"].tolist() + df_spline["Rmax"].tolist())),
        float
    )

    if len(R_edges_kept) < 2:
        raise RuntimeError("Not enough annuli to build heatmap.")

    n_r = len(R_edges_kept) - 1
    Zmap_raw = np.full((n_r, n_th), np.nan, float)
    seg_to_col = {seg: j for j, seg in enumerate(seg_plot_heat)}

    for _, row in df_spline.iterrows():
        if row["status_spline"] != "ok":
            continue

        seg = int(row["seg"])
        if seg not in seg_to_col:
            continue

        zmin = float(row["z_min_spline"])
        if not np.isfinite(zmin):
            continue

        Rmin = float(row["Rmin"])
        ir = np.searchsorted(R_edges_kept, Rmin, side="right") - 1
        if not (0 <= ir < n_r):
            continue

        it = seg_to_col[seg]
        Zmap_raw[ir, it] = zmin

    Zmap = np.where(np.isfinite(Zmap_raw) & (np.abs(Zmap_raw) <= zlim), Zmap_raw, np.nan)

    return Zmap, theta_edges, R_edges_kept, seg_plot_heat

import numpy as np
import pandas as pd


def estimate_spline_zmin_uncertainty_mc(
    z_centers,
    mean_line,
    cnt_line,
    sem_line,
    zmin=-1.5,
    zmax=1.5,
    min_cnt=5,
    min_bins=8,
    spline_k=3,
    spline_s=None,
    n_eval=400,
    max_allowed_minima=1,
    edge_buffer=0.05,
    n_mc=200,
    random_seed=42,
):
    """
    Estimate uncertainty in spline-fit z_min by Monte Carlo resampling
    of the binned mean trend.

    Parameters
    ----------
    mean_line : array
        Weighted mean Mg/Fe in each z-bin
    sem_line : array
        Uncertainty on the weighted mean in each z-bin

    Returns
    -------
    sigma_z0 : float
        MC uncertainty on inferred spline minimum
    n_ok_mc : int
        Number of successful MC realizations
    """
    rng = np.random.default_rng(random_seed)

    z0_samples = []

    for _ in range(n_mc):
        y_draw = np.array(mean_line, copy=True)

        ok = np.isfinite(mean_line) & np.isfinite(sem_line) & (sem_line > 0)
        y_draw[ok] = rng.normal(loc=mean_line[ok], scale=sem_line[ok])

        fit_obj_mc, st_mc, z0_mc = spline_fit_on_trend(
            z_centers=z_centers,
            trend=y_draw,
            cnt=cnt_line,
            sem=sem_line,
            zmin=zmin,
            zmax=zmax,
            min_cnt=min_cnt,
            min_bins=min_bins,
            spline_k=spline_k,
            spline_s=spline_s,
            n_eval=n_eval,
            max_allowed_minima=max_allowed_minima,
            edge_buffer=edge_buffer,
        )

        if st_mc == "ok" and np.isfinite(z0_mc):
            z0_samples.append(z0_mc)

    if len(z0_samples) < 5:
        return np.nan, len(z0_samples)

    return float(np.nanstd(z0_samples, ddof=1)), len(z0_samples)


def build_spline_measerr_zmin_and_err_maps(
    R_star,
    Z_star,
    mgfe_star,
    e_mg_h_star,
    e_fe_h_star,
    segm,
    seg_plot_heat,
    R_edges_kept,
    z_edges,
    minN_seg_total=15,
    min_bin_count=5,
    z_fit_min=-1.5,
    z_fit_max=1.5,
    min_bins_for_fit=8,
    spline_k=3,
    spline_s=None,
    n_eval=400,
    max_allowed_minima=1,
    edge_buffer=0.05,
    zlim=0.6,
    n_mc=200,
    random_seed=42,
):
    """
    Build polar maps of:
      - spline-fit z_min
      - MC uncertainty in spline-fit z_min

    using propagated measurement errors on Mg/Fe.

    Returns
    -------
    Zmap : ndarray
    Emap : ndarray
    df_spline : DataFrame
    R_edges_plot : ndarray
    """
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    sigma_mgfe_star = np.sqrt(
        np.asarray(e_mg_h_star, float)**2 +
        np.asarray(e_fe_h_star, float)**2
    )

    rows = []
    for ir in range(len(R_edges_kept) - 1):
        Rmin_focus = float(R_edges_kept[ir])
        Rmax_focus = float(R_edges_kept[ir + 1])

        mR = (R_star >= Rmin_focus) & (R_star < Rmax_focus)
        Ntot = int(np.sum(mR))

        if Ntot == 0:
            for seg_id in seg_plot_heat:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_spline=np.nan, z_min_err=np.nan,
                    status_spline="no_stars_annulus"
                ))
            continue

        for seg_id in seg_plot_heat:
            mseg = mR & segm[seg_id]
            good = (
                mseg &
                np.isfinite(Z_star) &
                np.isfinite(mgfe_star) &
                np.isfinite(sigma_mgfe_star) &
                (sigma_mgfe_star > 0)
            )

            Nseg = int(np.sum(good))
            if Nseg < minN_seg_total:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_spline=np.nan, z_min_err=np.nan,
                    status_spline="too_few_stars"
                ))
                continue

            Zs = Z_star[good]
            Ys = mgfe_star[good]
            sigs = sigma_mgfe_star[good]

            mean_line, cnt_line, sem_line = binned_weighted_mean_and_sem(
                Zs, Ys, sigs, z_edges
            )

            fit_obj, st, z0 = spline_fit_on_trend(
                z_centers=z_centers,
                trend=mean_line,
                cnt=cnt_line,
                sem=sem_line,
                zmin=z_fit_min,
                zmax=z_fit_max,
                min_cnt=min_bin_count,
                min_bins=min_bins_for_fit,
                spline_k=spline_k,
                spline_s=spline_s,
                n_eval=n_eval,
                max_allowed_minima=max_allowed_minima,
                edge_buffer=edge_buffer,
            )

            if st == "ok" and np.isfinite(z0):
                z0err, n_ok_mc = estimate_spline_zmin_uncertainty_mc(
                    z_centers=z_centers,
                    mean_line=mean_line,
                    cnt_line=cnt_line,
                    sem_line=sem_line,
                    zmin=z_fit_min,
                    zmax=z_fit_max,
                    min_cnt=min_bin_count,
                    min_bins=min_bins_for_fit,
                    spline_k=spline_k,
                    spline_s=spline_s,
                    n_eval=n_eval,
                    max_allowed_minima=max_allowed_minima,
                    edge_buffer=edge_buffer,
                    n_mc=n_mc,
                    random_seed=random_seed + ir * 100 + seg_id,
                )
            else:
                z0err = np.nan

            rows.append(dict(
                Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                z_min_spline=z0, z_min_err=z0err,
                status_spline=st
            ))

    df_spline = pd.DataFrame(rows)

    R_edges_plot = np.array(
        sorted(set(df_spline["Rmin"].tolist() + df_spline["Rmax"].tolist())),
        float
    )

    n_r = len(R_edges_plot) - 1
    n_th = len(seg_plot_heat)
    seg_to_col = {seg: j for j, seg in enumerate(seg_plot_heat)}

    Zmap_raw = np.full((n_r, n_th), np.nan, float)
    Emap_raw = np.full((n_r, n_th), np.nan, float)

    for _, row in df_spline.iterrows():
        if row["status_spline"] != "ok":
            continue

        seg = int(row["seg"])
        if seg not in seg_to_col:
            continue

        zmin = float(row["z_min_spline"])
        zerr = float(row["z_min_err"]) if np.isfinite(row["z_min_err"]) else np.nan

        if not np.isfinite(zmin):
            continue

        Rmin = float(row["Rmin"])
        ir2 = np.searchsorted(R_edges_plot, Rmin, side="right") - 1
        if not (0 <= ir2 < n_r):
            continue

        it = seg_to_col[seg]
        Zmap_raw[ir2, it] = zmin
        Emap_raw[ir2, it] = zerr

    Zmap = np.where(np.isfinite(Zmap_raw) & (np.abs(Zmap_raw) <= zlim), Zmap_raw, np.nan)
    Emap = np.where(np.isfinite(Zmap), Emap_raw, np.nan)

    return Zmap, Emap, df_spline, R_edges_plot

import numpy as np
import pandas as pd


def binned_weighted_mean_and_sem(z, y, sigma_y, z_edges):
    """
    Inverse-variance weighted mean and uncertainty of the mean in each z-bin.

    Parameters
    ----------
    z : array
    y : array
        quantity to average, here [Mg/Fe]
    sigma_y : array
        per-star uncertainty on y
    z_edges : array

    Returns
    -------
    mean : array
    cnt : array
    sem : array
    """
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    sigma_y = np.asarray(sigma_y, float)

    nb = len(z_edges) - 1
    mean = np.full(nb, np.nan, float)
    cnt = np.zeros(nb, int)
    sem = np.full(nb, np.nan, float)

    inds = np.searchsorted(z_edges, z, side="right") - 1
    ok = (
        (inds >= 0) & (inds < nb) &
        np.isfinite(z) &
        np.isfinite(y) &
        np.isfinite(sigma_y) &
        (sigma_y > 0)
    )

    inds = inds[ok]
    y = y[ok]
    sigma_y = sigma_y[ok]

    for i in range(nb):
        m = (inds == i)
        n = int(np.sum(m))
        cnt[i] = n

        if n == 0:
            continue

        y_i = y[m]
        s_i = sigma_y[m]

        w = 1.0 / s_i**2
        wsum = np.sum(w)

        if not np.isfinite(wsum) or wsum <= 0:
            continue

        mean[i] = np.sum(w * y_i) / wsum
        sem[i] = np.sqrt(1.0 / wsum)

    return mean, cnt, sem


def quad_fit_on_trend_with_err(z_centers, trend, cnt, sem,
                               zmin=-1.5, zmax=1.5,
                               min_cnt=5, min_bins=8):
    """
    Weighted quadratic fit y = a z^2 + b z + c using bin uncertainties.

    Returns
    -------
    coeffs : tuple
    status : str
    z0 : float
    sigma_z0 : float
    """
    zc = np.asarray(z_centers, float)
    tr = np.asarray(trend, float)
    cnt = np.asarray(cnt, int)
    sem = np.asarray(sem, float)

    m = (
        (zc >= zmin) & (zc <= zmax) &
        np.isfinite(tr) &
        (cnt >= min_cnt) &
        np.isfinite(sem) & (sem > 0)
    )

    if np.sum(m) < min_bins:
        return (np.nan, np.nan, np.nan), "too_few_bins", np.nan, np.nan

    x = zc[m]
    y = tr[m]
    s = sem[m]

    pos = s[np.isfinite(s) & (s > 0)]
    if len(pos) == 0:
        return (np.nan, np.nan, np.nan), "bad_sem", np.nan, np.nan

    s_floor = max(np.nanpercentile(pos, 10) * 0.25, 1e-4)
    s = np.maximum(s, s_floor)

    try:
        coeffs, cov = np.polyfit(x, y, deg=2, w=1.0 / s, cov=True)
    except Exception:
        return (np.nan, np.nan, np.nan), "fit_failed", np.nan, np.nan

    a, b, c = coeffs

    if (not np.isfinite(a)) or a <= 0:
        return (a, b, c), "bad_curvature", np.nan, np.nan

    z0 = -b / (2.0 * a)
    if not (zmin <= z0 <= zmax):
        return (a, b, c), "min_outside_window", np.nan, np.nan

    var_a = cov[0, 0]
    var_b = cov[1, 1]
    cov_ab = cov[0, 1]

    dz_da = b / (2.0 * a**2)
    dz_db = -1.0 / (2.0 * a)

    var_z0 = dz_da**2 * var_a + dz_db**2 * var_b + 2.0 * dz_da * dz_db * cov_ab

    if (not np.isfinite(var_z0)) or (var_z0 < 0):
        sigma_z0 = np.nan
    else:
        sigma_z0 = np.sqrt(var_z0)

    return (a, b, c), "ok", float(z0), float(sigma_z0)


def build_quad_measerr_zmin_and_err_maps(
    R_star,
    Z_star,
    mgfe_star,
    e_mg_h_star,
    e_fe_h_star,
    segm,
    seg_plot_heat,
    R_edges_kept,
    z_edges,
    minN_seg_total=15,
    min_bin_count=5,
    z_fit_min=-1.5,
    z_fit_max=1.5,
    min_bins_for_fit=8,
    zlim=0.6,
):
    """
    Build polar maps of:
      - quadratic-fit z_min
      - propagated uncertainty in quadratic-fit z_min

    using measurement-error weighting instead of scatter-based bin uncertainties.
    """
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    sigma_mgfe_star = np.sqrt(
        np.asarray(e_mg_h_star, float)**2 +
        np.asarray(e_fe_h_star, float)**2
    )

    rows = []
    for ir in range(len(R_edges_kept) - 1):
        Rmin_focus = float(R_edges_kept[ir])
        Rmax_focus = float(R_edges_kept[ir + 1])

        mR = (R_star >= Rmin_focus) & (R_star < Rmax_focus)
        Ntot = int(np.sum(mR))

        if Ntot == 0:
            for seg_id in seg_plot_heat:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_quad=np.nan, z_min_err=np.nan,
                    status_quad="no_stars_annulus"
                ))
            continue

        for seg_id in seg_plot_heat:
            mseg = mR & segm[seg_id]
            good = (
                mseg &
                np.isfinite(Z_star) &
                np.isfinite(mgfe_star) &
                np.isfinite(sigma_mgfe_star) &
                (sigma_mgfe_star > 0)
            )

            Nseg = int(np.sum(good))
            if Nseg < minN_seg_total:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_quad=np.nan, z_min_err=np.nan,
                    status_quad="too_few_stars"
                ))
                continue

            Zs = Z_star[good]
            Ys = mgfe_star[good]
            sigs = sigma_mgfe_star[good]

            mean_line, cnt_line, sem_line = binned_weighted_mean_and_sem(
                Zs, Ys, sigs, z_edges
            )

            (_, _, _), st, z0, z0err = quad_fit_on_trend_with_err(
                z_centers=z_centers,
                trend=mean_line,
                cnt=cnt_line,
                sem=sem_line,
                zmin=z_fit_min,
                zmax=z_fit_max,
                min_cnt=min_bin_count,
                min_bins=min_bins_for_fit
            )

            rows.append(dict(
                Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                z_min_quad=z0, z_min_err=z0err,
                status_quad=st
            ))

    df_quad = pd.DataFrame(rows)

    R_edges_plot = np.array(
        sorted(set(df_quad["Rmin"].tolist() + df_quad["Rmax"].tolist())),
        float
    )

    n_r = len(R_edges_plot) - 1
    n_th = len(seg_plot_heat)
    seg_to_col = {seg: j for j, seg in enumerate(seg_plot_heat)}

    Zmap_raw = np.full((n_r, n_th), np.nan, float)
    Emap_raw = np.full((n_r, n_th), np.nan, float)

    for _, row in df_quad.iterrows():
        if row["status_quad"] != "ok":
            continue

        seg = int(row["seg"])
        if seg not in seg_to_col:
            continue

        zmin = float(row["z_min_quad"])
        zerr = float(row["z_min_err"]) if np.isfinite(row["z_min_err"]) else np.nan

        if not np.isfinite(zmin):
            continue

        Rmin = float(row["Rmin"])
        ir2 = np.searchsorted(R_edges_plot, Rmin, side="right") - 1
        if not (0 <= ir2 < n_r):
            continue

        it = seg_to_col[seg]
        Zmap_raw[ir2, it] = zmin
        Emap_raw[ir2, it] = zerr

    Zmap = np.where(np.isfinite(Zmap_raw) & (np.abs(Zmap_raw) <= zlim), Zmap_raw, np.nan)
    Emap = np.where(np.isfinite(Zmap), Emap_raw, np.nan)

    return Zmap, Emap, df_quad, R_edges_plot

import numpy as np
import pandas as pd


def binned_mean_and_measerr_sem(z, y, sigma_y, z_edges):
    """
    Plain mean in each z-bin, but uncertainty of the mean from
    inverse-variance combination of per-star measurement errors.
    """
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    sigma_y = np.asarray(sigma_y, float)

    nb = len(z_edges) - 1
    mean = np.full(nb, np.nan, float)
    cnt = np.zeros(nb, int)
    sem = np.full(nb, np.nan, float)

    inds = np.searchsorted(z_edges, z, side="right") - 1
    ok = (
        (inds >= 0) & (inds < nb) &
        np.isfinite(z) &
        np.isfinite(y) &
        np.isfinite(sigma_y) &
        (sigma_y > 0)
    )

    inds = inds[ok]
    y = y[ok]
    sigma_y = sigma_y[ok]

    for i in range(nb):
        m = (inds == i)
        n = int(np.sum(m))
        cnt[i] = n

        if n == 0:
            continue

        y_i = y[m]
        s_i = sigma_y[m]

        # keep the SAME plain mean as before
        mean[i] = np.nanmean(y_i)

        # but compute uncertainty from measurement errors
        w = 1.0 / s_i**2
        wsum = np.sum(w)

        if np.isfinite(wsum) and wsum > 0:
            sem[i] = np.sqrt(1.0 / wsum)

    return mean, cnt, sem


def build_quad_measerr_only_emap(
    R_use,
    Z_use,
    mgfe_use,
    e_mg_h_use,
    e_fe_h_use,
    segm,
    seg_plot_heat,
    R_edges_kept,
    z_edges,
    minN_seg_total=15,
    min_bin_count=5,
    z_fit_min=-1.5,
    z_fit_max=1.5,
    min_bins_for_fit=8,
    zlim=0.6,
):
    """
    Build ONLY the uncertainty map for quadratic-fit z_min,
    using propagated measurement errors, while leaving the top-row
    z_min map to come from the original quadratic pipeline.

    Returns
    -------
    Emap : ndarray
    df_err : DataFrame
    R_edges_plot : ndarray
    """
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    sigma_mgfe_star = np.sqrt(
        np.asarray(e_mg_h_use, float)**2 +
        np.asarray(e_fe_h_use, float)**2
    )

    rows = []
    for ir in range(len(R_edges_kept) - 1):
        Rmin_focus = float(R_edges_kept[ir])
        Rmax_focus = float(R_edges_kept[ir + 1])

        mR = (R_use >= Rmin_focus) & (R_use < Rmax_focus)
        Ntot = int(np.sum(mR))

        if Ntot == 0:
            for seg_id in seg_plot_heat:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_err=np.nan, status_err="no_stars_annulus"
                ))
            continue

        for seg_id in seg_plot_heat:
            mseg = mR & segm[seg_id]
            good = (
                mseg &
                np.isfinite(Z_use) &
                np.isfinite(mgfe_use) &
                np.isfinite(sigma_mgfe_star) &
                (sigma_mgfe_star > 0)
            )

            Nseg = int(np.sum(good))
            if Nseg < minN_seg_total:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_err=np.nan, status_err="too_few_stars"
                ))
                continue

            Zs = Z_use[good]
            Ys = mgfe_use[good]
            sigs = sigma_mgfe_star[good]

            # SAME mean trend as original top-row method
            mean_line, cnt_line, sem_line = binned_mean_and_measerr_sem(
                Zs, Ys, sigs, z_edges
            )

            (_, _, _), st, z0, z0err = quad_fit_on_trend_with_err(
                z_centers=z_centers,
                trend=mean_line,
                cnt=cnt_line,
                sem=sem_line,
                zmin=z_fit_min,
                zmax=z_fit_max,
                min_cnt=min_bin_count,
                min_bins=min_bins_for_fit
            )

            rows.append(dict(
                Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                z_min_err=z0err,
                status_err=st
            ))

    df_err = pd.DataFrame(rows)

    R_edges_plot = np.array(
        sorted(set(df_err["Rmin"].tolist() + df_err["Rmax"].tolist())),
        float
    )

    n_r = len(R_edges_plot) - 1
    n_th = len(seg_plot_heat)
    seg_to_col = {seg: j for j, seg in enumerate(seg_plot_heat)}

    Emap_raw = np.full((n_r, n_th), np.nan, float)

    for _, row in df_err.iterrows():
        if row["status_err"] != "ok":
            continue

        seg = int(row["seg"])
        if seg not in seg_to_col:
            continue

        zerr = float(row["z_min_err"]) if np.isfinite(row["z_min_err"]) else np.nan
        if not np.isfinite(zerr):
            continue

        Rmin = float(row["Rmin"])
        ir2 = np.searchsorted(R_edges_plot, Rmin, side="right") - 1
        if not (0 <= ir2 < n_r):
            continue

        it = seg_to_col[seg]
        Emap_raw[ir2, it] = zerr

    return Emap_raw, df_err, R_edges_plot


import numpy as np
import pandas as pd


def binned_mean_and_measerr_sem(z, y, sigma_y, z_edges):
    """
    Plain mean in each z-bin, but uncertainty of the mean from
    inverse-variance combination of per-star measurement errors.
    """
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    sigma_y = np.asarray(sigma_y, float)

    nb = len(z_edges) - 1
    mean = np.full(nb, np.nan, float)
    cnt = np.zeros(nb, int)
    sem = np.full(nb, np.nan, float)

    inds = np.searchsorted(z_edges, z, side="right") - 1
    ok = (
        (inds >= 0) & (inds < nb) &
        np.isfinite(z) &
        np.isfinite(y) &
        np.isfinite(sigma_y) &
        (sigma_y > 0)
    )

    inds = inds[ok]
    y = y[ok]
    sigma_y = sigma_y[ok]

    for i in range(nb):
        m = (inds == i)
        n = int(np.sum(m))
        cnt[i] = n

        if n == 0:
            continue

        y_i = y[m]
        s_i = sigma_y[m]

        # keep SAME mean as your old spline top-row method
        mean[i] = np.nanmean(y_i)

        w = 1.0 / s_i**2
        wsum = np.sum(w)

        if np.isfinite(wsum) and wsum > 0:
            sem[i] = np.sqrt(1.0 / wsum)

    return mean, cnt, sem


def estimate_spline_zmin_uncertainty_mc_from_measerr(
    z_centers,
    mean_line,
    cnt_line,
    sem_line,
    zmin=-1.5,
    zmax=1.5,
    min_cnt=5,
    min_bins=8,
    spline_k=3,
    spline_s=None,
    n_eval=400,
    max_allowed_minima=1,
    edge_buffer=0.05,
    n_mc=200,
    random_seed=42,
):
    """
    Estimate uncertainty in smoothing-spline z_min by Monte Carlo resampling
    of the binned mean trend using sem_line from propagated measurement errors.
    """
    rng = np.random.default_rng(random_seed)

    z0_samples = []

    for _ in range(n_mc):
        y_draw = np.array(mean_line, copy=True)

        ok = np.isfinite(mean_line) & np.isfinite(sem_line) & (sem_line > 0)
        y_draw[ok] = rng.normal(loc=mean_line[ok], scale=sem_line[ok])

        fit_obj_mc, st_mc, z0_mc = spline_fit_on_trend(
            z_centers=z_centers,
            trend=y_draw,
            cnt=cnt_line,
            sem=None,   # keep top-row spline behavior; only perturb means
            zmin=zmin,
            zmax=zmax,
            min_cnt=min_cnt,
            min_bins=min_bins,
            spline_k=spline_k,
            spline_s=spline_s,
            n_eval=n_eval,
            max_allowed_minima=max_allowed_minima,
            edge_buffer=edge_buffer,
        )

        if st_mc == "ok" and np.isfinite(z0_mc):
            z0_samples.append(z0_mc)

    if len(z0_samples) < 5:
        return np.nan, len(z0_samples)

    return float(np.nanstd(z0_samples, ddof=1)), len(z0_samples)


def build_spline_measerr_only_emap(
    R_use,
    Z_use,
    mgfe_use,
    e_mg_h_use,
    e_fe_h_use,
    segm,
    seg_plot_heat,
    R_edges_kept,
    z_edges,
    minN_seg_total=15,
    min_bin_count=5,
    z_fit_min=-1.5,
    z_fit_max=1.5,
    min_bins_for_fit=8,
    spline_k=3,
    spline_s=None,
    n_eval=400,
    max_allowed_minima=1,
    edge_buffer=0.05,
    n_mc=200,
    random_seed=42,
):
    """
    Build ONLY the uncertainty map for smoothing-spline z_min,
    using propagated measurement errors, while leaving the top-row
    spline z_min maps unchanged.
    """
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    sigma_mgfe_star = np.sqrt(
        np.asarray(e_mg_h_use, float)**2 +
        np.asarray(e_fe_h_use, float)**2
    )

    rows = []
    for ir in range(len(R_edges_kept) - 1):
        Rmin_focus = float(R_edges_kept[ir])
        Rmax_focus = float(R_edges_kept[ir + 1])

        mR = (R_use >= Rmin_focus) & (R_use < Rmax_focus)
        Ntot = int(np.sum(mR))

        if Ntot == 0:
            for seg_id in seg_plot_heat:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_err=np.nan, status_err="no_stars_annulus"
                ))
            continue

        for seg_id in seg_plot_heat:
            mseg = mR & segm[seg_id]
            good = (
                mseg &
                np.isfinite(Z_use) &
                np.isfinite(mgfe_use) &
                np.isfinite(sigma_mgfe_star) &
                (sigma_mgfe_star > 0)
            )

            Nseg = int(np.sum(good))
            if Nseg < minN_seg_total:
                rows.append(dict(
                    Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                    z_min_err=np.nan, status_err="too_few_stars"
                ))
                continue

            Zs = Z_use[good]
            Ys = mgfe_use[good]
            sigs = sigma_mgfe_star[good]

            # same mean trend as old spline top-row method
            mean_line, cnt_line, sem_line = binned_mean_and_measerr_sem(
                Zs, Ys, sigs, z_edges
            )

            # first check whether the nominal spline fit is OK
            fit_obj, st, z0 = spline_fit_on_trend(
                z_centers=z_centers,
                trend=mean_line,
                cnt=cnt_line,
                sem=None,   # keep top-row spline behavior unchanged
                zmin=z_fit_min,
                zmax=z_fit_max,
                min_cnt=min_bin_count,
                min_bins=min_bins_for_fit,
                spline_k=spline_k,
                spline_s=spline_s,
                n_eval=n_eval,
                max_allowed_minima=max_allowed_minima,
                edge_buffer=edge_buffer,
            )

            if st == "ok" and np.isfinite(z0):
                z0err, n_ok_mc = estimate_spline_zmin_uncertainty_mc_from_measerr(
                    z_centers=z_centers,
                    mean_line=mean_line,
                    cnt_line=cnt_line,
                    sem_line=sem_line,
                    zmin=z_fit_min,
                    zmax=z_fit_max,
                    min_cnt=min_bin_count,
                    min_bins=min_bins_for_fit,
                    spline_k=spline_k,
                    spline_s=spline_s,
                    n_eval=n_eval,
                    max_allowed_minima=max_allowed_minima,
                    edge_buffer=edge_buffer,
                    n_mc=n_mc,
                    random_seed=random_seed + ir * 100 + seg_id,
                )
            else:
                z0err = np.nan

            rows.append(dict(
                Rmin=Rmin_focus, Rmax=Rmax_focus, seg=int(seg_id),
                z_min_err=z0err,
                status_err=st
            ))

    df_err = pd.DataFrame(rows)

    R_edges_plot = np.array(
        sorted(set(df_err["Rmin"].tolist() + df_err["Rmax"].tolist())),
        float
    )

    n_r = len(R_edges_plot) - 1
    n_th = len(seg_plot_heat)
    seg_to_col = {seg: j for j, seg in enumerate(seg_plot_heat)}

    Emap_raw = np.full((n_r, n_th), np.nan, float)

    for _, row in df_err.iterrows():
        if row["status_err"] != "ok":
            continue

        seg = int(row["seg"])
        if seg not in seg_to_col:
            continue

        zerr = float(row["z_min_err"]) if np.isfinite(row["z_min_err"]) else np.nan
        if not np.isfinite(zerr):
            continue

        Rmin = float(row["Rmin"])
        ir2 = np.searchsorted(R_edges_plot, Rmin, side="right") - 1
        if not (0 <= ir2 < n_r):
            continue

        it = seg_to_col[seg]
        Emap_raw[ir2, it] = zerr

    return Emap_raw, df_err, R_edges_plot

