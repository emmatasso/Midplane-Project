import numpy as np
from matplotlib.path import Path

from .utils import col_to_float_array, masked_fallback_array
from .config import GB_VERTICES, SNR_MIN, TEFF_RANGE, LOGG_RANGE


def close_poly(vertices):
    v = np.asarray(vertices, float)
    if not np.allclose(v[0], v[-1]):
        v = np.vstack([v, v[0]])
    return v


def signed_area(v_closed):
    x = v_closed[:, 0]
    y = v_closed[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def build_gb_polygon(vertices=GB_VERTICES):
    gb_closed = close_poly(vertices)
    if signed_area(gb_closed) < 0:
        gb_closed = gb_closed[::-1]
    return gb_closed


def giant_branch_mask(teff, logg, gb_closed=None):
    if gb_closed is None:
        gb_closed = build_gb_polygon()

    gb_path = Path(gb_closed)
    mask = gb_path.contains_points(np.vstack([teff, logg]).T)

    mask &= (
        (teff >= TEFF_RANGE[0]) & (teff <= TEFF_RANGE[1]) &
        (logg > LOGG_RANGE[0]) & (logg < LOGG_RANGE[1])
    )
    return mask


def build_prechem_selection(table, snr_min=SNR_MIN):
    teff = col_to_float_array(table, "teff")
    logg = col_to_float_array(table, "logg")
    feh  = col_to_float_array(table, "fe_h")
    mgh  = col_to_float_array(table, "mg_h")
    snr  = masked_fallback_array(table, "snr", "SNR")

    ra   = col_to_float_array(table, "ra")
    dec  = col_to_float_array(table, "dec")
    pmra = col_to_float_array(table, "pmra")
    pmde = col_to_float_array(table, "pmde")
    vrad = masked_fallback_array(table, "v_rad", "VHELIO_AVG")

    if "zgr_plx" not in table.colnames:
        raise KeyError("Column 'zgr_plx' not found.")
    plx = col_to_float_array(table, "zgr_plx")

    starflag_col = "flag_bad"
    if starflag_col not in table.colnames:
        raise KeyError("No STARFLAG column found.")
    starflag_mask = ~table[starflag_col]

    finite_basic = np.isfinite(teff) & np.isfinite(logg)
    finite_chem = finite_basic & np.isfinite(feh) & np.isfinite(mgh) & np.isfinite(snr)

    member_flags = np.asarray(table["sdss4_apogee_member_flags"], dtype=np.int64)
    no_cluster = (member_flags == 0)

    snr_mask = (snr > snr_min)
    plx_mask = np.isfinite(plx) & (plx > 0)

    kin_mask = (
        np.isfinite(ra) & np.isfinite(dec) &
        np.isfinite(pmra) & np.isfinite(pmde) &
        np.isfinite(vrad)
    )

    zgr_q = np.asarray(table["zgr_quality_flags"], dtype=np.int64)
    spectrum_q = np.asarray(table["spectrum_flags"], dtype=np.int64)

    zgr_q_mask = (zgr_q < 8)

    qual_mask = (
        ~table["flag_bad"] &
        (table["fe_h_flags"] == 0) &
        (table["mg_h_flags"] == 0)
    )

    gb_closed = build_gb_polygon()
    gb_mask = giant_branch_mask(teff, logg, gb_closed)

    master_mask = (
        finite_chem &
        gb_mask &
        snr_mask &
        starflag_mask &
        no_cluster &
        plx_mask &
        kin_mask &
        zgr_q_mask &
        qual_mask
    )

    diagnostics = {
        "starflag_pass": int(np.sum(starflag_mask)),
        "zgr_q_pass": int(np.sum(zgr_q_mask)),
        "qual_pass": int(np.sum(qual_mask)),
        "gb_pass": int(np.sum(gb_mask)),
        "finite_basic_pass": int(np.sum(finite_basic)),
        "finite_chem_pass": int(np.sum(finite_chem)),
        "snr_pass": int(np.sum(snr_mask)),
        "plx_pass": int(np.sum(plx_mask)),
        "kin_pass": int(np.sum(kin_mask)),
        "prechem_selected": int(np.sum(master_mask)),
        "gb_closed": gb_closed,
    }

    arrays = {
        "teff": teff,
        "logg": logg,
        "feh": feh,
        "mgh": mgh,
        "snr": snr,
        "ra": ra,
        "dec": dec,
        "pmra": pmra,
        "pmde": pmde,
        "vrad": vrad,
        "plx": plx,
    }

    return master_mask, diagnostics, arrays