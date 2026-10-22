"""
warp_fit_nuts.py

Everything needed for the NUTS (NumPyro) global warp fit, copied from the
Sep2026 notebook (cells 20, 21, 22, 24, 29) so it can be run on its own.

WHAT IT FITS
    The measured midplane heights z0(R, phi) from the per-cell chemistry fits
    (one z0 and one uncertainty per R bin x azimuthal segment).

MODEL (warp_model, 6 parameters)
    h_w(R)    = h_w0 * (R - R_w)**alpha_w          for R > R_w, else 0
    phi_LON(R)= phi_LON0                           for R <= R_t
              = phi_LON0 + beta_w * (R - R_t)      for R > R_t
    z_pred    = h_w(R) * sin(phi - phi_LON(R))
    likelihood: z0_obs ~ Normal(z_pred, z0_err)

PRIORS
    R_w      Uniform(2, 9)        kpc, radius where the warp starts
    h_w0     HalfNormal(0.05)     amplitude scale
    alpha_w  Uniform(0.5, 3)      how fast the amplitude grows with R
    phi_LON0 Normal(0, 10)        deg, line-of-nodes angle (inner disk)
    R_t      Uniform(9, 15)       kpc, radius where the twist starts
    beta_w   Normal(0, 15)        deg/kpc, twist rate
    (R_w tops out at 9 and R_t starts at 9, so R_t >= R_w automatically.)

CROSS-CHECK MODEL (warp_model_straight_LON)
    No twist, R_w fixed at the Poggio+25 young-giant value (5.5 kpc).
    Fits h_w0, alpha_w (Uniform(0.5, 4)), phi_LON0.

ANGLE CONVENTION
    phi is in degrees, in Poggio's convention (phi = 0 toward the anti-centre,
    increasing with rotation). `to_poggio_phi_deg` converts from this
    project's azimuth.

USAGE
    from warp_fit_nuts import *
    R, phi, z0, z0err = extract_fit_arrays(z0_med, z0_std, ok_grid,
                                           R_edges, crowd_bounds)
    mcmc = run_warp_fit(warp_model, R, phi, z0, z0err)
    divergence_report("unified (z_sun=0)", mcmc, len(z0))
    samples = mcmc.get_samples()
    summarize_posterior(samples)
    corner_plot(samples)
"""

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


# =============================================================================
# Reference values: Poggio+25 Table 1, young giants
# best value, statistical (+, -), systematic (+, -)
# =============================================================================

POGGIO_YOUNG_GIANT = {
    "R_w": 5.5, "h_w0": 0.012, "alpha_w": 1.9,
    "phi_LON0": 0.1, "R_t": 12.1, "beta_w": 9.9,
}

POGGIO_TABLE = {
    "R_w":      (5.5,   (0.8, 0.7),     (1.3, 0.8)),
    "h_w0":     (0.012, (0.002, 0.001), (0.006, 0.001)),
    "alpha_w":  (1.9,   (0.3, 0.3),     (0.5, 0.6)),
    "phi_LON0": (0.1,   (0.6, 0.6),     (1.1, 1.2)),
    "R_t":      (12.1,  (0.1, 0.1),     (0.1, 0.1)),
    "beta_w":   (9.9,   (0.5, 0.4),     (0.4, 0.2)),
}

PARAMS = ["R_w", "h_w0", "alpha_w", "phi_LON0", "R_t", "beta_w"]

LABELS = {
    "R_w":      r"$R_w$ [kpc]",
    "h_w0":     r"$h_{w,0}$ [kpc$^{1-\alpha_w}$]",
    "alpha_w":  r"$\alpha_w$",
    "phi_LON0": r"$\phi_{\rm LON,0}$ [deg]",
    "R_t":      r"$R_t$ [kpc]",
    "beta_w":   r"$\beta_w$ [deg/kpc]",
}


# =============================================================================
# 1. Models
# =============================================================================

def warp_model(R, phi_deg, z0_obs=None, z0_err=None):
    """Full 6-parameter twisted warp model."""
    R_w = numpyro.sample("R_w", dist.Uniform(2.0, 9.0))
    h_w0 = numpyro.sample("h_w0", dist.HalfNormal(0.05))
    alpha_w = numpyro.sample("alpha_w", dist.Uniform(0.5, 3.0))
    phi_LON0 = numpyro.sample("phi_LON0", dist.Normal(0.0, 10.0))
    R_t = numpyro.sample("R_t", dist.Uniform(9.0, 15.0))
    beta_w = numpyro.sample("beta_w", dist.Normal(0.0, 15.0))

    dR = jnp.maximum(R - R_w, 0.0)
    h_w = jnp.where(R > R_w, h_w0 * jnp.power(dR, alpha_w), 0.0)
    phi_LON = jnp.where(R <= R_t, phi_LON0, phi_LON0 + beta_w * (R - R_t))
    z_pred = h_w * jnp.sin(jnp.deg2rad(phi_deg) - jnp.deg2rad(phi_LON))

    numpyro.deterministic("z_pred", z_pred)
    numpyro.sample("obs", dist.Normal(z_pred, z0_err), obs=z0_obs)


def warp_model_straight_LON(R, phi_deg, z0_obs=None, z0_err=None):
    """No twist term. R_w fixed at the Poggio+25 value. Cross-check model."""
    R_w = POGGIO_YOUNG_GIANT["R_w"]
    h_w0 = numpyro.sample("h_w0", dist.HalfNormal(0.05))
    alpha_w = numpyro.sample("alpha_w", dist.Uniform(0.5, 4.0))
    phi_LON0 = numpyro.sample("phi_LON0", dist.Normal(0.0, 10.0))

    dR = jnp.maximum(R - R_w, 0.0)
    h_w = jnp.where(R > R_w, h_w0 * jnp.power(dR, alpha_w), 0.0)
    z_pred = h_w * jnp.sin(jnp.deg2rad(phi_deg) - jnp.deg2rad(phi_LON0))

    numpyro.deterministic("z_pred", z_pred)
    numpyro.sample("obs", dist.Normal(z_pred, z0_err), obs=z0_obs)


# =============================================================================
# 2. Building the data arrays from the z0 grid
# =============================================================================

def to_poggio_phi_deg(phi_deg_project):
    """Project azimuth (deg) -> Poggio convention, wrapped to [-180, 180)."""
    phi = 180.0 - np.asarray(phi_deg_project, float)
    return (phi + 180.0) % 360.0 - 180.0


def segment_centers_poggio_deg(crowd_bounds):
    """Azimuthal segment centers (deg) in Poggio's convention."""
    theta_edges = np.unwrap(np.array(crowd_bounds, float))
    for i in range(1, len(theta_edges)):
        while theta_edges[i] <= theta_edges[i - 1]:
            theta_edges[i] += 2 * np.pi
    centers_deg = np.degrees(0.5 * (theta_edges[:-1] + theta_edges[1:]))
    return to_poggio_phi_deg(centers_deg)


def extract_fit_arrays(z0_med, z0_std, ok_grid, R_edges, crowd_bounds,
                       R_min=8.277, R_max=14.5):
    """
    Flatten the (n_R, n_seg) z0 grid into 1D arrays for the fit.

    z0_med, z0_std : (n_R, n_seg) median z0 and uncertainty per cell
    ok_grid        : (n_R, n_seg) bool, cells that passed the fit requirements
    R_edges        : radial bin edges (kpc)
    crowd_bounds   : azimuthal segment boundaries (radians, project convention)
    R_min, R_max   : only cells with bin center in this range are used.
                     NOTE: R_min = 8.277 is just outside the solar radius, so
                     there are no data inside R_w for R_w to be constrained by.
                     A posterior R_w near the lower data edge is a data
                     coverage effect, not a measurement.

    Returns R, phi_deg (Poggio convention), z0, z0_err (all 1D).
    """
    R_edges = np.asarray(R_edges, float)
    R_centers = 0.5 * (R_edges[:-1] + R_edges[1:])
    phi_centers = segment_centers_poggio_deg(crowd_bounds)
    R_grid, Phi_grid = np.meshgrid(R_centers, phi_centers, indexing="ij")

    mask = (np.isfinite(z0_med) & np.isfinite(z0_std) & (z0_std > 0)
            & (R_grid >= R_min) & (R_grid <= R_max))
    if ok_grid is not None:
        mask &= ok_grid
    return R_grid[mask], Phi_grid[mask], z0_med[mask], z0_std[mask]


# =============================================================================
# 3. Running the fit
# =============================================================================

def run_warp_fit(model_fn, R, phi_deg, z0, z0_err, seed=0,
                 num_warmup=1500, num_samples=2000, num_chains=4):
    """NUTS fit. Returns the MCMC object. Prints the NumPyro summary."""
    rng_key = jax.random.PRNGKey(seed)
    kernel = NUTS(model_fn, target_accept_prob=0.95, max_tree_depth=12)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains)
    mcmc.run(rng_key, R=jnp.asarray(R), phi_deg=jnp.asarray(phi_deg),
             z0_obs=jnp.asarray(z0), z0_err=jnp.asarray(z0_err),
             extra_fields=("diverging",))
    mcmc.print_summary()
    return mcmc


def divergence_report(label, mcmc_obj, n_points):
    div = mcmc_obj.get_extra_fields()["diverging"]
    n_div, n_total = int(div.sum()), len(div)
    print(f"{label}: {n_points} data points, {n_div}/{n_total} divergences "
          f"({100 * n_div / n_total:.1f}%)")


# =============================================================================
# 4. Printing and plotting results
# =============================================================================

def print_priors():
    """Print the NUTS priors (plain text, same as the notebook printout)."""
    print("=== NUTS (NumPyro) priors ===")
    print("Twisted LON model (warp_model):")
    rows = [("R_w", "Uniform(low=2, high=9)"),
            ("h_w0", "HalfNormal(scale=0.05)"),
            ("alpha_w", "Uniform(low=0.5, high=3)"),
            ("phi_LON0", "Normal(loc=0, scale=10)"),
            ("R_t", "Uniform(low=9, high=15)"),
            ("beta_w", "Normal(loc=0, scale=15)")]
    for name, text in rows:
        print(f"  {name:<9} {text}")


def summarize_posterior(samples, params=PARAMS):
    """Median and 16th/84th percentile for each parameter."""
    print(f"{'parameter':<10} {'median':>9} {'-1sig':>9} {'+1sig':>9}")
    for p in params:
        lo, med, hi = np.percentile(np.asarray(samples[p]), [16, 50, 84])
        print(f"{p:<10} {med:>9.3f} {med - lo:>9.3f} {hi - med:>9.3f}")


def corner_plot(samples, params=PARAMS, labels=LABELS):
    """Corner plot of the posterior with 1 and 2 sigma contours."""
    import corner
    import matplotlib.pyplot as plt

    chain = np.column_stack([np.asarray(samples[p]) for p in params])
    fig = corner.corner(
        chain,
        labels=[labels[p] for p in params],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2.0)),
        smooth=1.0,
        bins=40,
    )
    fig.set_size_inches(14, 14)
    plt.show()
    return fig
