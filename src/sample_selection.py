"""
sample_selection.py

Every cut used to build the sample, in the order they are applied.
Read it top to bottom. Each block has:
    WHAT   what the cut does
    SOURCE where the numbers come from   (TODO = fill in)
    WHY    why we do it                  (TODO = fill in)
followed by the code.

select_sample(t) runs all of it on the joined catalogue and prints how many
stars are left after each step.

This is a readable copy of the logic in src/selection.py, src/chemistry.py and
src/coords.py plus a few cuts that live in the notebook. If you change a cut in
src/, change it here too (or the other way around).
"""

import numpy as np
from matplotlib.path import Path

from .utils import col_to_float_array, masked_fallback_array


# =============================================================================
# VALUES (all the numbers in one place)
# =============================================================================

# --- step 2: giant branch -----------------------------------------------------
# polygon in (Teff [K], logg)
GB_VERTICES = np.array([
    [5100, 2.20], [4700, 1.85], [4300, 1.25], [3900, 0.70], [3400, 0.20],
    [3050, -0.10], [3050, 0.80], [3500, 1.30], [4000, 2.00], [4400, 2.75],
    [4700, 3.45], [4850, 3.90], [5000, 4.00], [5100, 4.00],
], float)
TEFF_RANGE = (3000, 7500)
LOGG_RANGE = (-2, 6.5)

# --- step 3: signal to noise -------------------------------------------------
SNR_MIN = 20

# --- step 8: ZGR quality ------------------------------------------------------
ZGR_QUALITY_MAX = 8        # keep zgr_quality_flags < 8

# --- step 10: low-alpha dividing line in ([Fe/H], [Mg/Fe]) -------------------
FE_BREAK = -0.1
M1, B1 = -0.05, 0.10                      # line for [Fe/H] >= FE_BREAK
M2 = -0.20                                # slope for [Fe/H] <  FE_BREAK
B2 = (M1 * FE_BREAK + B1) - M2 * FE_BREAK # so the two lines meet at FE_BREAK

# --- steps 11 and 12: metallicity floors ------------------------------------
FEH_CUT = -1.2             # in src/chemistry.py
FEH_MIN_EXTRA = -0.9       # extra cut in the notebook, applied after. This is the effective floor.

# --- step 13: [Al/Fe] vs [Mg/Mn] polygon -------------------------------------
ALFE_MGMN_POLY_VERTICES = np.array([
    [-0.4, -0.25],   # lower-left
    [0.15, -0.40],   # lower-right
    [0.6, 0.7],      # upper-right
    [-0.2, 0.70],    # upper-left
], float)

# --- steps 14 and 15: geometry -----------------------------------------------
GALCEN_DISTANCE_KPC = 8.275
GALCEN_V_SUN = (8, 254, 8)   # km/s
Z_SUN_PC = 0                 # Sun placed exactly on the midplane (the *_zsun0 functions)
R_MAX_KPC = 16.0

# --- steps 16 to 20: binning and fit requirements (applied in the notebook) --
PHI0 = np.pi                 # Sun-Galactic centre line
HALFWIDTH_DEG = 45           # wedge is +/- this about PHI0
N_AZ_SEGMENTS = 11
R_BIN_WIDTH_KPC = (0.5, 1.0) # depends on the analysis
Z_BIN_WIDTH_KPC = 0.1
Z_FIT_MIN, Z_FIT_MAX = -1.5, 1.5
MIN_STARS_PER_CELL = 15
MIN_STARS_PER_ZBIN = 5
MIN_ZBINS_PER_CELL = 8


# =============================================================================
# THE CUTS, IN ORDER
# =============================================================================
def select_sample(t):
    """
    t: astropy Table = astraAllStarASPCAP joined to DR17 allStarLite (adds SNR, VHELIO_AVG)
       on sdss4_apogee_id (left join). See the notebook for the join.
    Returns the final table and prints the number of stars left after each step.
    """
    log = []

    def keep(label, table, mask):
        """apply a boolean mask, record the count"""
        table = table[mask]
        log.append((label, len(table)))
        return table

    log.append(("0. parent catalogue", len(t)))

    # ---- columns used by several cuts ---------------------------------------
    teff = col_to_float_array(t, "teff")
    logg = col_to_float_array(t, "logg")
    feh = col_to_float_array(t, "fe_h")
    mgh = col_to_float_array(t, "mg_h")
    snr = masked_fallback_array(t, "snr", "SNR")
    plx = col_to_float_array(t, "zgr_plx")
    ra = col_to_float_array(t, "ra")
    dec = col_to_float_array(t, "dec")
    pmra = col_to_float_array(t, "pmra")
    pmde = col_to_float_array(t, "pmde")

    # build all pre-chemistry masks on the full table, then combine in one go
    # (the counts below are for the cumulative selection)
    masks = {}

    # 1. finite values ---------------------------------------------------------
    # WHAT   Teff, logg, [Fe/H], [Mg/H] and S/N must all be finite.
    # SOURCE n/a      WHY   cannot use a star without them.
    masks["1. finite Teff, logg, [Fe/H], [Mg/H], S/N"] = (
        np.isfinite(teff) & np.isfinite(logg) & np.isfinite(feh)
        & np.isfinite(mgh) & np.isfinite(snr))

    # 2. red giant branch -------------------------------------------------------
    # WHAT   Teff-logg polygon (GB_VERTICES) plus the Teff and logg ranges above.
    # SOURCE TODO        WHY TODO
    gb = GB_VERTICES
    if not np.allclose(gb[0], gb[-1]):
        gb = np.vstack([gb, gb[0]])
    in_polygon = Path(gb).contains_points(np.vstack([teff, logg]).T)
    masks["2. giant branch polygon, Teff and logg ranges"] = (
        in_polygon
        & (teff >= TEFF_RANGE[0]) & (teff <= TEFF_RANGE[1])
        & (logg > LOGG_RANGE[0]) & (logg < LOGG_RANGE[1]))

    # 3. signal to noise --------------------------------------------------------
    # WHAT   APOGEE spectral S/N > SNR_MIN (S/N comes from the DR17 join).
    # SOURCE TODO        WHY TODO
    masks[f"3. S/N > {SNR_MIN}"] = snr > SNR_MIN

    # 4. bad-star flag ----------------------------------------------------------
    # WHAT   flag_bad must be False.
    # SOURCE TODO (what does flag_bad contain?)        WHY TODO
    masks["4. flag_bad is False"] = ~np.asarray(t["flag_bad"], bool)

    # 5. abundance flags --------------------------------------------------------
    # WHAT   fe_h_flags == 0 and mg_h_flags == 0.
    # SOURCE TODO        WHY TODO
    masks["5. fe_h_flags == 0 and mg_h_flags == 0"] = (
        (np.asarray(t["fe_h_flags"]) == 0) & (np.asarray(t["mg_h_flags"]) == 0))

    # 6. cluster members --------------------------------------------------------
    # WHAT   sdss4_apogee_member_flags == 0, i.e. no "likely member of <cluster>" bit is set.
    #        The bits flag likely members of clusters such as M92, M15, M53, NGC 5466,
    #        NGC 4147, M2, M13, M3, M5, M12, M107, M71, NGC 2243 (bits 0-12 checked;
    #        the list continues, see the DR19 page).
    # SOURCE SDSS DR19 Astra "Source and Spectrum Flags" page
    #        (www.sdss.org/dr19/software/pipelines/astra/flags/). The flags are inherited
    #        from the SDSS-IV APOGEE data reduction, not computed by us.
    # NOTE   Stars with no SDSS-IV counterpart (sdss4_apogee_id == -1 after the join)
    #        may have no flag set, so they would pass this cut. Not checked.
    # WHY    TODO
    masks["6. not a flagged cluster member"] = (
        np.asarray(t["sdss4_apogee_member_flags"], dtype=np.int64) == 0)

    # 7. parallax ---------------------------------------------------------------
    # WHAT   zgr_plx finite and positive. Distance = 1 / parallax.
    #        There is NO parallax S/N or fractional-error cut.
    # SOURCE TODO        WHY TODO
    masks["7. zgr_plx finite and > 0"] = np.isfinite(plx) & (plx > 0)

    # 8. ZGR quality ------------------------------------------------------------
    # WHAT   zgr_quality_flags < ZGR_QUALITY_MAX (= 8).
    #        Bits 1-3 (values 1, 2, 4) say the confidence in Teff, [Fe/H] or logg is
    #        < 0.5; these are allowed. The cut removes bit 4 (chi2_opt/61 > 2),
    #        bit 5 (ln_prior < -7.43, unlikely under the model prior) and bit 6
    #        (ZGR parallax more than 10 sigma from the Gaia DR3 parallax).
    # SOURCE Zhang, Green & Rix 2023 (MNRAS 524, 1855), Table 3: the recommended
    #        "basic reliability cut". quality_flags == 0 would be stricter.
    # WHY    removes poor SED fits, stars unlikely under the model, and stars whose
    #        ZGR parallax disagrees with Gaia DR3.
    masks[f"8. zgr_quality_flags < {ZGR_QUALITY_MAX}"] = (
        np.asarray(t["zgr_quality_flags"], dtype=np.int64) < ZGR_QUALITY_MAX)

    # 9. astrometry -------------------------------------------------------------
    # WHAT   ra, dec, pmra and pmde finite.
    #        Radial velocity is NOT required. Checked on the final sample: 150,194 of
    #        391,384 stars have no v_rad, and none are missing a proper motion.
    #        (src/selection.py builds a kin_mask that includes v_rad, but it does not
    #        remove stars without one. Not resolved; this file follows what the
    #        notebook output shows.)
    #        Positions (R, phi, z) do not depend on proper motions or radial velocity
    #        (checked: identical x, y, z with all-NaN velocities). The pm/rv requirement
    #        in src/selection.py removes only 22 of 630,101 stars at this stage.
    # NOTE   spectrum_flags is NOT used anywhere, so there is no spectrum-flag cut.
    # SOURCE n/a         WHY TODO
    masks["9. finite ra, dec, proper motions"] = (
        np.isfinite(ra) & np.isfinite(dec) & np.isfinite(pmra) & np.isfinite(pmde))

    total = np.ones(len(t), bool)
    for label, m in masks.items():
        total &= m
        log.append((label + " (cumulative)", int(total.sum())))
    t_sel = t[total]

    # ---- Galactocentric coordinates (needed by the chemistry cuts) ----------
    # Distance = 1/parallax, Galactocentric frame with
    # galcen_distance = GALCEN_DISTANCE_KPC, galcen_v_sun = GALCEN_V_SUN, z_sun = 0.
    # Code: src/coords.py  build_galactocentric_xyz_zsun0 / build_star_pack_zsun0
    from .coords import build_galactocentric_xyz_zsun0, build_star_pack_zsun0
    x, y, z = build_galactocentric_xyz_zsun0(t_sel)

    # 10. low-alpha disk --------------------------------------------------------
    # WHAT   [Mg/Fe] below a two-piece straight line in [Fe/H]:
    #            [Fe/H] >= FE_BREAK:  [Mg/Fe] < M1*[Fe/H] + B1
    #            [Fe/H] <  FE_BREAK:  [Mg/Fe] < M2*[Fe/H] + B2
    # SOURCE TODO        WHY TODO
    feh_s = col_to_float_array(t_sel, "fe_h")
    mgfe_s = col_to_float_array(t_sel, "mg_h") - feh_s
    divide = np.where(feh_s >= FE_BREAK, M1 * feh_s + B1, M2 * feh_s + B2)
    finite = np.isfinite(feh_s) & np.isfinite(mgfe_s)
    low_alpha = finite & (mgfe_s < divide)
    t_sel = keep("10. low-alpha ([Mg/Fe] below the dividing line)", t_sel, low_alpha)
    feh_s = feh_s[low_alpha]

    # 11. metallicity floor (src) ----------------------------------------------
    # WHAT   [Fe/H] >= FEH_CUT.      SOURCE TODO      WHY TODO
    ok = feh_s >= FEH_CUT
    t_sel = keep(f"11. [Fe/H] >= {FEH_CUT}", t_sel, ok)
    feh_s = feh_s[ok]

    # 12. metallicity floor (notebook) -----------------------------------------
    # WHAT   [Fe/H] > FEH_MIN_EXTRA. Tighter than step 11, so this is the effective floor.
    # SOURCE TODO        WHY TODO
    t_sel = keep(f"12. [Fe/H] > {FEH_MIN_EXTRA}", t_sel, feh_s > FEH_MIN_EXTRA)

    # 13. [Al/Fe] vs [Mg/Mn] polygon -------------------------------------------
    # WHAT   [Al/Fe] = al_h - fe_h and [Mg/Mn] = mg_h - mn_h must lie inside the
    #        polygon ALFE_MGMN_POLY_VERTICES. Non-finite values are dropped.
    # SOURCE TODO        WHY TODO
    alfe = col_to_float_array(t_sel, "al_h") - col_to_float_array(t_sel, "fe_h")
    mgmn = col_to_float_array(t_sel, "mg_h") - col_to_float_array(t_sel, "mn_h")
    finite = np.isfinite(alfe) & np.isfinite(mgmn)
    inside = np.zeros(len(t_sel), bool)
    inside[finite] = Path(ALFE_MGMN_POLY_VERTICES, closed=True).contains_points(
        np.vstack([alfe[finite], mgmn[finite]]).T)
    t_final = keep("13. inside the [Al/Fe]-[Mg/Mn] polygon", t_sel, inside)

    # 14 and 15. coordinates and maximum radius --------------------------------
    # WHAT   finite Galactocentric coordinates and R < R_MAX_KPC.
    # Code: build_star_pack_zsun0(t_final, rmax_keep=R_MAX_KPC)
    star_pack, diag = build_star_pack_zsun0(t_final, rmax_keep=R_MAX_KPC, plx_min=None)
    log.append((f"14-15. finite coordinates and R < {R_MAX_KPC:g} kpc", diag["final_size"]))

    # ---- print the log ------------------------------------------------------
    print(f"{'step':62s} {'N stars':>10s}")
    for label, n in log:
        print(f"{label:62s} {n:>10,d}")

    return t_final, star_pack, log

