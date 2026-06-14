import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, Distance

from .utils import col_to_float_array


def build_galactocentric_xyz(table,
                             galcen_distance_kpc=8.275,
                             galcen_v_sun=(8, 254, 8)):
    """
    Build Galactocentric x, y, z arrays from a table with
    ra, dec, zgr_plx, pmra, pmde, v_rad columns.
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
        pm_ra_cosdec=pmra * u.mas/u.yr,
        pm_dec=pmde * u.mas/u.yr,
        radial_velocity=vrad * u.km/u.s,
    )

    galcen_frame = coord.Galactocentric(
        galcen_distance=galcen_distance_kpc * u.kpc,
        galcen_v_sun=list(galcen_v_sun) * u.km/u.s
    )

    galcen = c.transform_to(galcen_frame)

    x = galcen.x.to_value(u.kpc)
    y = galcen.y.to_value(u.kpc)
    z = galcen.z.to_value(u.kpc)

    return x, y, z

import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, Distance

from .utils import col_to_float_array, masked_fallback_array


def build_galactocentric_phase_space(table,
                                     galcen_distance_kpc=8.275,
                                     galcen_v_sun=(8, 254, 8)):
    """
    Build a 6D Galactocentric phase-space sample from a table.
    Uses v_rad if available, otherwise falls back to VHELIO_AVG.

    Returns
    -------
    phase : dict
        Arrays for X,Y,Z,VX,VY,VZ,R,phi,vR,vphi,vz
    diagnostics : dict
        Counts for RV availability and final kept sample
    """
    plx  = col_to_float_array(table, "zgr_plx")
    ra   = col_to_float_array(table, "ra")
    dec  = col_to_float_array(table, "dec")
    pmra = col_to_float_array(table, "pmra")
    pmde = col_to_float_array(table, "pmde")

    vrad = (
        col_to_float_array(table, "v_rad")
        if "v_rad" in table.colnames
        else np.full(len(table), np.nan)
    )
    vhel = (
        col_to_float_array(table, "VHELIO_AVG")
        if "VHELIO_AVG" in table.colnames
        else np.full(len(table), np.nan)
    )

    v = np.where(np.isfinite(vrad), vrad, vhel)

    has_vrad = np.isfinite(vrad)
    has_vhel = np.isfinite(vhel)
    has_v = np.isfinite(v)

    ok6d = (
        np.isfinite(plx) & (plx > 0) &
        np.isfinite(ra) & np.isfinite(dec) &
        np.isfinite(pmra) & np.isfinite(pmde) &
        has_v
    )

    dist = Distance((1000.0 / plx[ok6d]) * u.pc)

    c = SkyCoord(
        ra=ra[ok6d] * u.deg,
        dec=dec[ok6d] * u.deg,
        distance=dist,
        pm_ra_cosdec=pmra[ok6d] * u.mas/u.yr,
        pm_dec=pmde[ok6d] * u.mas/u.yr,
        radial_velocity=v[ok6d] * u.km/u.s,
        frame="icrs",
    )

    galcen_frame = coord.Galactocentric(
        galcen_distance=galcen_distance_kpc * u.kpc,
        galcen_v_sun=list(galcen_v_sun) * u.km/u.s
    )

    gc = c.transform_to(galcen_frame)

    X  = gc.x.to_value(u.kpc)
    Y  = gc.y.to_value(u.kpc)
    Z  = gc.z.to_value(u.kpc)
    VX = gc.v_x.to_value(u.km/u.s)
    VY = gc.v_y.to_value(u.km/u.s)
    VZ = gc.v_z.to_value(u.km/u.s)

    R   = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)

    good = (
        np.isfinite(R) & (R > 1e-6) &
        np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) &
        np.isfinite(VX) & np.isfinite(VY) & np.isfinite(VZ)
    )

    X, Y, Z, VX, VY, VZ, R, phi = (
        X[good], Y[good], Z[good],
        VX[good], VY[good], VZ[good],
        R[good], phi[good]
    )

    vR   = (X * VX + Y * VY) / R
    vphi = (-X * VY + Y * VX) / R
    vz   = VZ

    phase = {
        "X": X, "Y": Y, "Z": Z,
        "VX": VX, "VY": VY, "VZ": VZ,
        "R": R, "phi": phi,
        "vR": vR, "vphi": vphi, "vz": vz,
    }

    diagnostics = {
        "input_size": len(table),
        "have_vrad": int(np.sum(has_vrad)),
        "have_vhel": int(np.sum(has_vhel)),
        "have_either_rv": int(np.sum(has_v)),
        "using_vrad": int(np.sum(ok6d & has_vrad)),
        "using_vhel_fallback": int(np.sum(ok6d & (~has_vrad) & has_vhel)),
        "dropped_before_transform": int(np.sum(~ok6d)),
        "phase_space_size": len(X),
    }

    return phase, diagnostics

import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, Distance

from .utils import col_to_float_array


def build_annuli_sample_from_table(table,
                                   plx_unit="mas",
                                   plx_snr_min=5.0,
                                   galcen_distance_kpc=8.275,
                                   galcen_v_sun=(8, 254, 8)):
    """
    Build R, Z, and Mg/Fe arrays for annuli plots from a final table.

    Parameters
    ----------
    table : astropy Table
    plx_unit : str
        'mas' or 'arcsec'
    plx_snr_min : float
        Minimum parallax S/N. Set to 0 to disable.

    Returns
    -------
    sample : dict
        Contains R_all, Z_all, mgfe_all, and the masked table used.
    diagnostics : dict
        Counts for parallax-valid and final usable sample size.
    """
    plx = col_to_float_array(table, "zgr_plx")

    has_eplx = ("zgr_e_plx" in table.colnames)
    if has_eplx:
        eplx = col_to_float_array(table, "zgr_e_plx")
    else:
        eplx = None

    if plx_unit == "mas":
        plx_arcsec = plx / 1000.0
        eplx_arcsec = eplx / 1000.0 if eplx is not None else None
    elif plx_unit == "arcsec":
        plx_arcsec = plx
        eplx_arcsec = eplx
    else:
        raise ValueError("plx_unit must be 'mas' or 'arcsec'")

    parallax_mask = np.isfinite(plx_arcsec) & (plx_arcsec > 0)

    if eplx_arcsec is not None and plx_snr_min > 0:
        parallax_mask &= (
            np.isfinite(eplx_arcsec) &
            (eplx_arcsec > 0) &
            ((plx_arcsec / eplx_arcsec) >= plx_snr_min)
        )

    t_use = table[parallax_mask]

    dist = Distance(parallax=plx_arcsec[parallax_mask] * u.arcsec)

    c = SkyCoord(
        ra=col_to_float_array(t_use, "ra") * u.deg,
        dec=col_to_float_array(t_use, "dec") * u.deg,
        distance=dist,
        pm_ra_cosdec=col_to_float_array(t_use, "pmra") * u.mas / u.yr,
        pm_dec=col_to_float_array(t_use, "pmde") * u.mas / u.yr,
        radial_velocity=col_to_float_array(t_use, "v_rad") * u.km / u.s,
    )

    galcen_frame = coord.Galactocentric(
        galcen_distance=galcen_distance_kpc * u.kpc,
        galcen_v_sun=list(galcen_v_sun) * u.km / u.s,
    )

    gc = c.transform_to(galcen_frame)

    X_all = gc.x.to_value(u.kpc)
    Y_all = gc.y.to_value(u.kpc)
    Z_all = gc.z.to_value(u.kpc)
    R_all = np.sqrt(X_all**2 + Y_all**2)

    mgfe_all = (
        col_to_float_array(t_use, "mg_h") -
        col_to_float_array(t_use, "fe_h")
    )

    finite_all = np.isfinite(R_all) & np.isfinite(Z_all) & np.isfinite(mgfe_all)

    sample = {
        "t_use": t_use[finite_all],
        "R_all": R_all[finite_all],
        "Z_all": Z_all[finite_all],
        "mgfe_all": mgfe_all[finite_all],
    }

    diagnostics = {
        "parallax_valid": int(np.sum(parallax_mask)),
        "input_size": len(table),
        "final_size": int(np.sum(finite_all)),
        "has_eplx": has_eplx,
    }

    return sample, diagnostics

import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, Distance

from .utils import col_to_float_array


def build_star_pack(table,
                    galcen_distance_kpc=8.275,
                    galcen_v_sun=(8, 254, 8),
                    rmax_keep=16.0,
                    plx_min=None):
    """
    Build one canonical aligned star pack from a chosen table.

    Returns
    -------
    star_pack : dict
        X_star, Y_star, Z_star, R_star, phi_star, mgfe_star, t_use
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
        galcen_v_sun=list(galcen_v_sun) * u.km/u.s
    )

    c = SkyCoord(
        ra=col_to_float_array(t_use, "ra") * u.deg,
        dec=col_to_float_array(t_use, "dec") * u.deg,
        distance=dist,
        pm_ra_cosdec=col_to_float_array(t_use, "pmra") * u.mas/u.yr,
        pm_dec=col_to_float_array(t_use, "pmde") * u.mas/u.yr,
        radial_velocity=col_to_float_array(t_use, "v_rad") * u.km/u.s,
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

import numpy as np


def wrap_to_2pi(a):
    return np.mod(a, 2 * np.pi)


def between_ccw(x, a, b):
    da = (b - a) % (2 * np.pi)
    dx = (x - a) % (2 * np.pi)
    return dx < da


def segment_masks_ncrowd(phi, phi0=np.pi, crowd_halfwidth_deg=45, n_crowd=11):
    """
    Build crowded-window segment masks plus one 'rest' segment.

    Returns
    -------
    segm : dict
        {1..n_crowd, n_crowd+1(rest)} boolean masks
    crowd_bounds : ndarray
        Boundary angles (len = n_crowd+1) spanning crowded window CCW
    window_edges : tuple
        (a, b) crowded-window edges
    """
    ph = wrap_to_2pi(phi)
    phi0 = wrap_to_2pi(phi0)

    hw = np.deg2rad(crowd_halfwidth_deg)
    a = wrap_to_2pi(phi0 - hw)
    b = wrap_to_2pi(phi0 + hw)

    span = (b - a) % (2 * np.pi)
    crowd_bounds = wrap_to_2pi(a + np.linspace(0, span, n_crowd + 1))

    segm = {}
    for i in range(n_crowd):
        segm[i + 1] = between_ccw(ph, crowd_bounds[i], crowd_bounds[i + 1])

    in_crowd = between_ccw(ph, a, b)
    segm[n_crowd + 1] = ~in_crowd

    return segm, crowd_bounds, (a, b)


def build_segment_setup(phi_star, phi0=np.pi, crowd_halfwidth_deg=45, n_crowd=11):
    """
    Convenience wrapper for segment setup.
    """
    segm, crowd_bounds, window_edges = segment_masks_ncrowd(
        phi_star,
        phi0=phi0,
        crowd_halfwidth_deg=crowd_halfwidth_deg,
        n_crowd=n_crowd
    )

    seg_rest_id = n_crowd + 1
    seg_plot = list(range(1, n_crowd + 1))

    diagnostics = {
        "n_crowd": n_crowd,
        "seg_rest_id": seg_rest_id,
        "seg_plot": seg_plot,
    }

    return segm, crowd_bounds, window_edges, diagnostics

