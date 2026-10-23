"""
Galactocentric coordinate transform, z_sun = 0 pc version.

Converts observed (ra, dec, parallax, proper motion, radial velocity) into
Galactocentric Cartesian (x, y, z) and cylindrical (R, phi) coordinates,
with the Sun placed exactly on the Galactic midplane (z_sun = 0 pc) rather
than astropy's default offset (~20.8 pc).

Two entry points:
    build_galactocentric_xyz_zsun0  -- bare x, y, z from a table
    build_star_pack_zsun0           -- full star pack (adds R, phi, [Mg/Fe],
                                        parallax/finite cuts, rmax_keep cut)

Both take an astropy Table with columns: ra, dec, zgr_plx, pmra, pmde,
v_rad (plus mg_h, fe_h for build_star_pack_zsun0).
"""

import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, Distance

from src.utils import col_to_float_array


# --- Assumed Galactocentric frame constants ---------------------------------
GALCEN_DISTANCE_KPC = 8.275          # Sun-to-Galactic-center distance
GALCEN_V_SUN = (8, 254, 8)           # Sun's velocity, km/s (U, V, W)
Z_SUN = 0 * u.pc                     # height of the Sun above the midplane


def build_galactocentric_xyz_zsun0(
    table,
    galcen_distance_kpc=GALCEN_DISTANCE_KPC,
    galcen_v_sun=GALCEN_V_SUN,
):
    """
    Build Galactocentric x, y, z arrays from a table with
    ra, dec, zgr_plx, pmra, pmde, v_rad columns, with z_sun = 0 pc.
    """
    plx = col_to_float_array(table, "zgr_plx")  # mas
    ra = col_to_float_array(table, "ra")
    dec = col_to_float_array(table, "dec")
    pmra = col_to_float_array(table, "pmra")
    pmde = col_to_float_array(table, "pmde")
    vrad = col_to_float_array(table, "v_rad")

    dist_pc = 1000.0 / plx
    dist = Distance(dist_pc * u.pc)

    c = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=dist,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmde * u.mas / u.yr,
        radial_velocity=vrad * u.km / u.s,
    )

    galcen_frame = coord.Galactocentric(
        galcen_distance=galcen_distance_kpc * u.kpc,
        galcen_v_sun=list(galcen_v_sun) * u.km / u.s,
        z_sun=Z_SUN,
    )

    galcen = c.transform_to(galcen_frame)

    x = galcen.x.to_value(u.kpc)
    y = galcen.y.to_value(u.kpc)
    z = galcen.z.to_value(u.kpc)

    return x, y, z


def build_star_pack_zsun0(
    table,
    galcen_distance_kpc=GALCEN_DISTANCE_KPC,
    galcen_v_sun=GALCEN_V_SUN,
    rmax_keep=16.0,
    plx_min=None,
):
    """
    Build one canonical aligned star pack from a chosen table, with
    z_sun = 0 pc.

    Returns
    -------
    star_pack : dict
        X_star, Y_star, Z_star, R_star, phi_star, mgfe_star, t_star
    diagnostics : dict
        Counts before/after cuts
    """
    t_use = table

    plx = col_to_float_array(t_use, "zgr_plx")
    plx_ok = np.isfinite(plx) & (plx > 0)

    if plx_min is not None:
        plx_ok &= (plx >= plx_min)

    t_use = t_use[plx_ok]

    dist = Distance((1000.0 / col_to_float_array(t_use, "zgr_plx")) * u.pc)

    galcen_frame = coord.Galactocentric(
        galcen_distance=galcen_distance_kpc * u.kpc,
        galcen_v_sun=list(galcen_v_sun) * u.km / u.s,
        z_sun=Z_SUN,
    )

    c = SkyCoord(
        ra=col_to_float_array(t_use, "ra") * u.deg,
        dec=col_to_float_array(t_use, "dec") * u.deg,
        distance=dist,
        pm_ra_cosdec=col_to_float_array(t_use, "pmra") * u.mas / u.yr,
        pm_dec=col_to_float_array(t_use, "pmde") * u.mas / u.yr,
        radial_velocity=col_to_float_array(t_use, "v_rad") * u.km / u.s,
    )

    gc = c.transform_to(galcen_frame)

    X_all = gc.x.to_value(u.kpc)
    Y_all = gc.y.to_value(u.kpc)
    Z_all = gc.z.to_value(u.kpc)
    R_all = np.sqrt(X_all**2 + Y_all**2)
    phi_all = np.arctan2(Y_all, X_all)

    mgfe_all = (
        col_to_float_array(t_use, "mg_h") -
        col_to_float_array(t_use, "fe_h")
    )

    finite = (
        np.isfinite(X_all) & np.isfinite(Y_all) & np.isfinite(Z_all) &
        np.isfinite(R_all) & np.isfinite(phi_all) & np.isfinite(mgfe_all)
    )

    X_star = X_all[finite]
    Y_star = Y_all[finite]
    Z_star = Z_all[finite]
    R_star = R_all[finite]
    phi_star = phi_all[finite]
    mgfe_star = mgfe_all[finite]
    t_star = t_use[finite]

    if rmax_keep is not None:
        mRmax = (R_star < rmax_keep)
        X_star = X_star[mRmax]
        Y_star = Y_star[mRmax]
        Z_star = Z_star[mRmax]
        R_star = R_star[mRmax]
        phi_star = phi_star[mRmax]
        mgfe_star = mgfe_star[mRmax]
        t_star = t_star[mRmax]

    star_pack = {
        "X_star": X_star,
        "Y_star": Y_star,
        "Z_star": Z_star,
        "R_star": R_star,
        "phi_star": phi_star,
        "mgfe_star": mgfe_star,
        "t_star": t_star,
    }

    diagnostics = {
        "input_size": len(table),
        "after_plx_cut": len(t_use),
        "after_finite_cut": int(np.sum(finite)),
        "final_size": len(R_star),
        "rmax_keep": rmax_keep,
    }

    return star_pack, diagnostics
