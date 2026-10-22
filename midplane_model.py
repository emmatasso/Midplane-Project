"""
midplane_model.py

Reusable sech^2 mixture-model pipeline for measuring the chemical/vertical
midplane height (z0) and effective scale height (h_eff) of a stellar disk,
either as a function of (R, phi) on a grid, or as a function of R alone
(the "flare" analysis).

This module packages the fitting machinery built and tuned against the
observational sample (SDSS-V APOGEE) so it can be applied unchanged to a
simulation snapshot for direct comparison. It intentionally does NOT include
any plotting/heatmap code -- only the model, the fits, and the h_eff
post-processing. Bring your own R/z/[Mg/Fe]-analog arrays and, optionally,
azimuthal segment masks.

Expected input arrays (names are illustrative; any per-star arrays work):
    R        : Galactocentric radius, kpc
    z        : height above/below the plane, kpc
    tracer   : the abundance-ratio-like quantity being fit as a function of z
               (e.g. [Mg/Fe] for real data; the simulation's analogous tracer)
    tracer_err : per-star uncertainty on `tracer`

Two models, not one
--------------------
The pipeline this was built from settled on running TWO separate fits per
cell/bin, not one:
  - `model_z0`   : used to measure the midplane height z0 (the canonical
    model -- tighter d_h2 ceiling, a sampled outlier fraction Q).
  - `model_h_eff`: used specifically for the h_eff (effective scale height)
    calculation -- a looser d_h2 ceiling (needed to reach observed trough
    depths without the two components collapsing into a degenerate,
    overly-wide combination) and Q fixed at 0 (no outlier mixture).
z0 and h_eff are therefore NOT two numbers pulled from the same posterior;
they come from two independently-run MCMC fits over the same binned data.
Every fitting function below takes `model_fn` as a parameter so you pick
which one to use per call.

Usage sketch
------------
    import midplane_model as mm

    # z0 grid, mirroring Section 7 of the observational notebook
    per_cell_z0 = mm.run_grid_fit(
        R, z, tracer, tracer_err, seg_masks, R_edges, model_fn=mm.model_z0,
        label="sim_z0", results_path="sim_grid_fit_z0.pkl",
    )
    z0_med, z0_std, _, _ = mm.summarize_grid_fit(
        per_cell_z0, n_R=len(R_edges) - 1, n_seg=len(seg_masks)
    )

    # separate grid fit with model_h_eff, used only to get h_eff
    per_cell_heff = mm.run_grid_fit(
        R, z, tracer, tracer_err, seg_masks, R_edges, model_fn=mm.model_h_eff,
        label="sim_h_eff", results_path="sim_grid_fit_h_eff.pkl",
    )
    _, _, h_eff_med, h_eff_std = mm.summarize_grid_fit(
        per_cell_heff, n_R=len(R_edges) - 1, n_seg=len(seg_masks)
    )

    # radial-only "flare" fit, mirroring Section 9 (uses model_h_eff, since
    # h_eff is the quantity of interest there)
    per_R_flare = mm.fit_flare_bin(
        R, z, tracer, tracer_err, R_centers, R_width=1.0, model_fn=mm.model_h_eff,
        label="sim_flare", results_path="sim_flare_fit.pkl",
    )
    h_eff_med_flare, h_eff_std_flare = mm.compute_h_eff_flare(per_R_flare)
"""

import pickle

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
from scipy.stats import binned_statistic
from scipy.optimize import brentq
from tqdm import tqdm


# =============================================================================
# 1. Model
# =============================================================================

def mixture_of_sech2(z, A1, A2, h1, h2, z0):
    """Two-component sech^2 dip, both components sharing the same center z0."""
    dz = z - z0
    return A1 * (1 / jnp.cosh(dz / h1)) ** 2 + A2 * (1 / jnp.cosh(dz / h2)) ** 2


def model_z0(data):
    """
    Numpyro model used to measure the midplane height z0. Two-component
    sech^2 mixture with a sampled outlier-mixture likelihood, fit to binned
    tracer-vs-z data. This is the canonical/first-settled version of the
    model (narrower d_h2 ceiling, Q sampled rather than fixed).

      - h1 in [0.04, 1.0] kpc
      - d_h2 in [0.04, 0.5] kpc -- the separation between h1 and h2
      - Q (outlier fraction) ~ TruncatedNormal(0, 0.05, [0, 1.0]), sampled
    Adjust freely; these are the values used for the observational z0 fit,
    not universal constants.
    """
    _A1 = numpyro.sample("_A1", dist.HalfNormal(0.2))
    _A2 = numpyro.sample("_A2", dist.HalfNormal(0.2))
    A1 = numpyro.deterministic("A1", -_A1)
    A2 = numpyro.deterministic("A2", -_A2)

    h1 = numpyro.sample("h1", dist.Uniform(0.04, 1.0))
    d_h2 = numpyro.sample("d_h2", dist.Uniform(0.04, 0.5))
    h2 = numpyro.deterministic("h2", h1 + d_h2)

    z0 = numpyro.sample("z0", dist.Normal(0.0, 0.2))
    offset = numpyro.sample("offset", dist.Normal(0.05, 0.10))

    model_y = mixture_of_sech2(data["z"], A1, A2, h1, h2, z0) + offset

    Q = numpyro.sample("Q", dist.TruncatedNormal(0.0, 0.05, low=0.0, high=1.0))
    mix = dist.Categorical(probs=jnp.array([Q, 1.0 - Q]))
    components = [
        dist.Normal(0.0, jnp.sqrt(1.0 ** 2 + data["tracer_err"] ** 2)),
        dist.Normal(model_y, data["tracer_err"]),
    ]

    numpyro.sample("obs", dist.Mixture(mix, components), obs=data["tracer"])


def model_h_eff(data):
    """
    Numpyro model used specifically for the h_eff (effective scale height)
    calculation -- a separate fit from model_z0, not a shared posterior.

    Differences from model_z0, arrived at during troubleshooting of h_eff
    fits that were failing to reach the depth/width of the observed
    [Mg/Fe]-vs-z trough:
      - d_h2 ceiling loosened to 0.75 kpc (from 0.5). A narrower ceiling
        forced h1 and h2 to sit too close together to jointly capture a
        sharp central dip and broad wings; how far this needs to go is
        data-dependent -- when tuning this against the observational
        sample, ceilings from 0.5 up through 1.5 were tried, and the
        posterior's behavior at the ceiling (does the 84th percentile of
        d_h2 pile up against it?) was the diagnostic used to judge whether
        it was still too tight. Re-run that same check on simulation data
        before trusting 0.75 here.
      - offset prior loosened to Normal(0.0, 0.2) (from Normal(0.05, 0.10)),
        removing a mildly informative prior that could compete with the
        amplitude parameters for how deep the trough is allowed to be.
      - Q fixed at 0 (numpyro.deterministic, not sampled) -- the outlier
        mixture is disabled entirely for this fit, rather than just
        tightened.
    """
    _A1 = numpyro.sample("_A1", dist.HalfNormal(0.2))
    _A2 = numpyro.sample("_A2", dist.HalfNormal(0.2))
    A1 = numpyro.deterministic("A1", -_A1)
    A2 = numpyro.deterministic("A2", -_A2)

    h1 = numpyro.sample("h1", dist.Uniform(0.04, 1.0))
    d_h2 = numpyro.sample("d_h2", dist.Uniform(0.04, 0.75))
    h2 = numpyro.deterministic("h2", h1 + d_h2)

    z0 = numpyro.sample("z0", dist.Normal(0.0, 0.2))
    offset = numpyro.sample("offset", dist.Normal(0.0, 0.2))

    model_y = mixture_of_sech2(data["z"], A1, A2, h1, h2, z0) + offset

    Q = numpyro.deterministic("Q", 0.0)
    mix = dist.Categorical(probs=jnp.array([Q, 1.0 - Q]))
    components = [
        dist.Normal(0.0, jnp.sqrt(1.0 ** 2 + data["tracer_err"] ** 2)),
        dist.Normal(model_y, data["tracer_err"]),
    ]

    numpyro.sample("obs", dist.Mixture(mix, components), obs=data["tracer"])


# =============================================================================
# 2. Binning + single-cell fit
# =============================================================================

Z_BINS_FIT = np.arange(-5, 5 + 1e-3, 0.1)


def bin_one(z, tracer, tracer_err, z_bins=Z_BINS_FIT):
    """Inverse-variance-weighted mean of `tracer` vs z, with intrinsic-scatter
    inflation of the per-bin uncertainty."""
    stat1 = binned_statistic(z, tracer / tracer_err ** 2, bins=z_bins, statistic="sum")
    stat2 = binned_statistic(z, 1 / tracer_err ** 2, bins=z_bins, statistic="sum")
    stat_std = binned_statistic(z, tracer, bins=z_bins, statistic="std")
    stat_count = binned_statistic(z, tracer, bins=z_bins, statistic="count")

    intrinsic_scatter = stat_std.statistic / np.sqrt(stat_count.statistic)

    mean_tracer = stat1.statistic / stat2.statistic
    mean_tracer_err = np.sqrt(1 / stat2.statistic + intrinsic_scatter ** 2)
    z_binc = 0.5 * (z_bins[1:] + z_bins[:-1])

    return z_binc, mean_tracer, mean_tracer_err, stat_count.statistic


def fit_sech2(z_binc, mean_tracer, mean_tracer_err, key, model_fn,
              num_warmup=1000, num_samples=1000, num_chains=2,
              progress_bar=False):
    """Run NUTS on one binned (z, tracer, tracer_err) cell, using whichever
    model function is passed in (model_z0 or model_h_eff)."""
    sampler = infer.MCMC(
        infer.NUTS(
            model_fn,
            dense_mass=True,
            target_accept_prob=0.95,
            init_strategy=infer.init_to_median(num_samples=20),
        ),
        num_warmup=num_warmup, num_samples=num_samples,
        num_chains=num_chains, progress_bar=progress_bar,
    )
    data = {
        "z": jnp.asarray(z_binc),
        "tracer": jnp.asarray(mean_tracer),
        "tracer_err": jnp.asarray(mean_tracer_err),
    }
    sampler.run(key, data=data)
    return sampler


def select_annulus_segment(R_use, Z_use, tracer_use, tracer_err_use, seg_mask,
                            Rmin, Rmax,
                            minN_total=15, min_bin_count=5, min_bins_for_fit=8,
                            z_fit_min=-1.5, z_fit_max=1.5):
    """Select stars in one (R, segment) cell, bin them, and apply
    minimum-count cuts. Returns (z_binc, mean, err, n_stars, ok)."""
    m = (
        seg_mask
        & (R_use >= Rmin) & (R_use < Rmax)
        & np.isfinite(Z_use) & np.isfinite(tracer_use)
        & np.isfinite(tracer_err_use) & (tracer_err_use > 0)
    )
    n_stars = int(m.sum())
    if n_stars < minN_total:
        return None, None, None, n_stars, False

    z_binc, mean_tracer, mean_tracer_err, counts = bin_one(
        Z_use[m], tracer_use[m], tracer_err_use[m]
    )

    good = (
        np.isfinite(mean_tracer) & np.isfinite(mean_tracer_err)
        & (counts >= min_bin_count)
        & (z_binc >= z_fit_min) & (z_binc <= z_fit_max)
    )
    if good.sum() < min_bins_for_fit:
        return None, None, None, n_stars, False

    return (
        np.asarray(z_binc[good], float),
        np.asarray(mean_tracer[good], float),
        np.asarray(mean_tracer_err[good], float),
        n_stars,
        True,
    )


# =============================================================================
# 3. (R, phi) grid fit
# =============================================================================

def run_grid_fit(R_use, Z_use, tracer_use, tracer_err_use, seg_masks, R_edges,
                  model_fn, label="fit", results_path=None,
                  minN_total=15, min_bin_count=5, min_bins_for_fit=8,
                  z_fit_min=-1.5, z_fit_max=1.5, seed=0):
    """
    Fit the sech^2 mixture model in every (R, segment) cell.

    Parameters
    ----------
    R_use, Z_use, tracer_use, tracer_err_use : per-star arrays, same length
    seg_masks : dict {seg_id: boolean array} same length as R_use, one mask
        per azimuthal segment
    R_edges : radial bin edges (n_R + 1 values)
    model_fn : which model to fit -- model_z0 (for midplane height) or
        model_h_eff (for the effective-scale-height calculation). These
        are two independent fits over the same binned data, not shared.
    results_path : if given, cache the result dict there (pickle)

    Returns
    -------
    per_cell : dict {(i_R, j_seg): {"z_binc", "mean", "err", "samples"}}
    """
    seg_ids = list(seg_masks.keys())
    n_R = len(R_edges) - 1
    n_seg = len(seg_ids)

    z0_med = np.full((n_R, n_seg), np.nan)
    z0_std = np.full((n_R, n_seg), np.nan)
    n_stars_grid = np.zeros((n_R, n_seg), dtype=int)
    n_bins_grid = np.zeros((n_R, n_seg), dtype=int)
    ok_grid = np.zeros((n_R, n_seg), dtype=bool)
    per_cell = {}

    key = jr.PRNGKey(seed)
    pairs = [(i_R, j_seg) for i_R in range(n_R) for j_seg in range(n_seg)]

    for k, (i_R, j_seg) in enumerate(tqdm(pairs, desc=f"[{label}] grid fit"), 1):
        Rmin, Rmax = float(R_edges[i_R]), float(R_edges[i_R + 1])
        seg_mask = seg_masks[seg_ids[j_seg]]

        z_binc, mean_t, err_t, n_stars, ok = select_annulus_segment(
            R_use, Z_use, tracer_use, tracer_err_use, seg_mask, Rmin, Rmax,
            minN_total=minN_total, min_bin_count=min_bin_count,
            min_bins_for_fit=min_bins_for_fit,
            z_fit_min=z_fit_min, z_fit_max=z_fit_max,
        )
        n_stars_grid[i_R, j_seg] = n_stars
        if not ok:
            continue
        n_bins_grid[i_R, j_seg] = len(z_binc)

        try:
            key, sub = jr.split(key)
            sampler = fit_sech2(z_binc, mean_t, err_t, sub, model_fn, progress_bar=False)
            samples = sampler.get_samples()
            z0_med[i_R, j_seg] = float(np.median(samples["z0"]))
            z0_std[i_R, j_seg] = float(np.std(samples["z0"]))
            ok_grid[i_R, j_seg] = True
            per_cell[(i_R, j_seg)] = {
                "z_binc": np.asarray(z_binc), "mean": np.asarray(mean_t),
                "err": np.asarray(err_t),
                "samples": {kk: np.asarray(vv[:1000]) for kk, vv in samples.items()},
            }
        except Exception as e:
            print(f"  [{label}] fit failed: seg={seg_ids[j_seg]} R=[{Rmin:.1f},{Rmax:.1f}] -> {e}")

    print(f"[{label}] grid fits complete: {int(ok_grid.sum())}/{ok_grid.size}")

    if results_path is not None:
        with open(results_path, "wb") as f:
            pickle.dump({
                "z0_med": z0_med, "z0_std": z0_std,
                "n_stars": n_stars_grid, "n_bins": n_bins_grid,
                "ok_grid": ok_grid, "per_cell": per_cell,
                "R_edges": R_edges, "seg_ids": seg_ids,
            }, f)
        print(f"[{label}] saved to {results_path}")

    return per_cell


# =============================================================================
# 4. h_eff from posterior samples
# =============================================================================

def compute_h_eff_fwhm(pars, z_max=2.0, half_width=False):
    """
    Two-sided effective scale height: find the z-crossings on either side of
    z0 where the model amplitude drops to half its central value.

    half_width=False (default) returns the full width (right crossing minus
    left crossing). half_width=True returns that divided by 2, i.e. the
    average distance from z0 to either edge -- useful if you want this on
    the same rough scale as the one-sided flare definition below.
    """
    z0 = pars["z0"]
    half_max = 0.5 * (pars["A1"] + pars["A2"])

    f_z0 = mixture_of_sech2(z0, pars["A1"], pars["A2"], pars["h1"], pars["h2"], z0)
    f_right = mixture_of_sech2(z0 + z_max, pars["A1"], pars["A2"], pars["h1"], pars["h2"], z0)
    f_left = mixture_of_sech2(z0 - z_max, pars["A1"], pars["A2"], pars["h1"], pars["h2"], z0)

    try:
        if (f_z0 - half_max) * (f_right - half_max) < 0:
            zstar_right = brentq(
                lambda z: mixture_of_sech2(z, pars["A1"], pars["A2"], pars["h1"], pars["h2"], z0) - half_max,
                z0 + 1e-3, z0 + z_max,
            )
        else:
            return np.nan

        if (f_z0 - half_max) * (f_left - half_max) < 0:
            zstar_left = brentq(
                lambda z: mixture_of_sech2(z, pars["A1"], pars["A2"], pars["h1"], pars["h2"], z0) - half_max,
                z0 - z_max, z0 - 1e-3,
            )
        else:
            return np.nan

        width = zstar_right - zstar_left
        return width / 2 if half_width else width
    except ValueError:
        return np.nan


def summarize_grid_fit(per_cell, n_R, n_seg, z_max=2.0, half_width=True,
                        n_draws=256):
    """
    Post-process a grid fit's `per_cell` into (z0_med, z0_std, h_eff_med,
    h_eff_std) arrays. No refitting -- this only touches posterior samples
    already sitting in `per_cell`.
    """
    z0_med = np.full((n_R, n_seg), np.nan)
    z0_std = np.full((n_R, n_seg), np.nan)
    h_eff_med = np.full((n_R, n_seg), np.nan)
    h_eff_std = np.full((n_R, n_seg), np.nan)

    for (i_R, j_seg), cell in tqdm(per_cell.items(), desc="summarizing grid fit"):
        s = cell["samples"]
        z0_med[i_R, j_seg] = float(np.median(s["z0"]))
        z0_std[i_R, j_seg] = float(np.std(s["z0"]))

        h_effs = []
        for i in range(min(n_draws, len(s["A1"]))):
            cell_sample = {k: v[i] for k, v in s.items()}
            cell_sample["h2"] = cell_sample["h1"] + cell_sample["d_h2"]
            h_eff = compute_h_eff_fwhm(cell_sample, z_max=z_max, half_width=half_width)
            if np.isfinite(h_eff):
                h_effs.append(h_eff)

        if len(h_effs) > 0:
            h_effs = np.array(h_effs)
            h_eff_med[i_R, j_seg] = np.median(h_effs)
            pcl = np.percentile(h_effs, [16, 84])
            h_eff_std[i_R, j_seg] = 0.5 * (pcl[1] - pcl[0])

    return z0_med, z0_std, h_eff_med, h_eff_std


# =============================================================================
# 5. Radial-only ("flare") fit
# =============================================================================

def fit_flare_bin(R_arr, Z_arr, tracer_arr, err_arr, R_centers, model_fn,
                   R_width=1.0, label="flare", results_path=None,
                   minN_total=15, min_bin_count=5, min_bins_for_fit=8,
                   z_fit_min=-1.5, z_fit_max=1.5, seed=0):
    """
    Fit the sech^2 mixture model in overlapping radial-only bins (azimuth
    collapsed). `R_centers` is an array of bin-center radii; each bin spans
    [center - R_width/2, center + R_width/2], so with R_width > the spacing
    between centers, neighboring bins share stars.

    model_fn : model_z0 or model_h_eff (see module docstring) -- the flare
        analysis is normally run with model_h_eff, since h_eff vs R is the
        quantity of interest, but the parameter is left open in case you
        want z0 vs R too.

    Returns per_R_flare : dict {i_R: {"Rlo", "Rhi", "n_stars", "z_binc",
    "mean", "err", "samples"}}, where i_R is the index into R_centers.
    """
    R_arr = np.asarray(R_arr)
    Z_arr = np.asarray(Z_arr)
    tracer_arr = np.asarray(tracer_arr)
    err_arr = np.asarray(err_arr)

    m_base = (
        np.isfinite(R_arr) & np.isfinite(Z_arr)
        & np.isfinite(tracer_arr) & np.isfinite(err_arr)
        & (err_arr > 0)
    )

    per_R_flare = {}
    key = jr.PRNGKey(seed)

    for i_R in tqdm(range(len(R_centers)), desc=f"[{label}] fitting radial bins"):
        Rlo = R_centers[i_R] - R_width / 2
        Rhi = R_centers[i_R] + R_width / 2

        m = m_base & (R_arr >= Rlo) & (R_arr < Rhi)
        n_stars = int(m.sum())

        if n_stars < minN_total:
            print(f"[{label}] R=[{Rlo:.1f},{Rhi:.1f}]: skipping, only {n_stars} stars")
            continue

        z_binc, mean_t, err_t, counts = bin_one(Z_arr[m], tracer_arr[m], err_arr[m])

        good = (
            np.isfinite(mean_t) & np.isfinite(err_t)
            & (counts >= min_bin_count)
            & (z_binc >= z_fit_min) & (z_binc <= z_fit_max)
        )
        if good.sum() < min_bins_for_fit:
            print(f"[{label}] R=[{Rlo:.1f},{Rhi:.1f}]: skipping, only {good.sum()} good bins")
            continue

        try:
            key, sub = jr.split(key)
            sampler = fit_sech2(
                np.asarray(z_binc[good], float),
                np.asarray(mean_t[good], float),
                np.asarray(err_t[good], float),
                sub, model_fn, progress_bar=False,
            )
            samples = sampler.get_samples()
            per_R_flare[i_R] = {
                "Rlo": Rlo, "Rhi": Rhi, "n_stars": n_stars,
                "z_binc": np.asarray(z_binc[good]),
                "mean": np.asarray(mean_t[good]),
                "err": np.asarray(err_t[good]),
                "samples": {k: np.asarray(v) for k, v in samples.items()},
            }
            print(f"[{label}] R=[{Rlo:.1f},{Rhi:.1f}]: fit ok, n_stars={n_stars}")
        except Exception as e:
            print(f"[{label}] R=[{Rlo:.1f},{Rhi:.1f}]: fit failed -> {e}")

    print(f"[{label}] fits complete: {len(per_R_flare)}/{len(R_centers)}")

    if results_path is not None:
        with open(results_path, "wb") as f:
            pickle.dump(per_R_flare, f)
        print(f"[{label}] saved to {results_path}")

    return per_R_flare


def _root_func_one_sided(z, pars):
    Ystar = mixture_of_sech2(z, pars["A1"], pars["A2"], pars["h1"], pars["h2"], pars["z0"])
    return Ystar - 0.5 * (pars["A1"] + pars["A2"])


def compute_h_eff_flare(per_R_flare, n_draws=256):
    """
    One-sided h_eff: distance from z0 to the right-hand half-max crossing
    only (not a full width). This is the definition used for the radial-only
    flare analysis; note it is NOT numerically the same quantity as
    `compute_h_eff_fwhm` above (see module docstring / project notes on
    reconciling the two).

    Returns (h_eff_med, h_eff_std) arrays indexed the same way as
    `per_R_flare`'s keys (0..n_R_flare-1; entries with no fit stay NaN).
    """
    n_R_flare = max(per_R_flare.keys()) + 1 if per_R_flare else 0
    h_eff_med = np.full(n_R_flare, np.nan)
    h_eff_std = np.full(n_R_flare, np.nan)

    for i_R, cell in tqdm(per_R_flare.items(), desc="computing flare h_eff"):
        s = cell["samples"]
        h_effs = []
        for i in range(min(n_draws, len(s["A1"]))):
            cell_sample = {k: v[i] for k, v in s.items()}
            cell_sample["h2"] = cell_sample["h1"] + cell_sample["d_h2"]
            hi = 8 * max(cell_sample["h1"], cell_sample["h2"])
            try:
                zstar = brentq(_root_func_one_sided, cell_sample["z0"] + 1e-9, hi,
                                args=(cell_sample,))
                h_effs.append(zstar - cell_sample["z0"])
            except ValueError:
                continue

        if len(h_effs) > 0:
            h_effs = np.array(h_effs)
            h_eff_med[i_R] = np.median(h_effs)
            pcl = np.percentile(h_effs, [16, 84])
            h_eff_std[i_R] = 0.5 * (pcl[1] - pcl[0])

    return h_eff_med, h_eff_std
