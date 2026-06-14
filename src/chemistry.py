import numpy as np

from matplotlib.path import Path
from .utils import col_to_float_array
from .config import ALFE_MGMN_POLY_VERTICES

FE_BREAK = -0.1
M1, B1 = -0.05, 0.10
M2 = -0.20
B2 = (M1 * FE_BREAK + B1) - M2 * FE_BREAK

FEH_CUT = -1.2


def compute_fe_mgfe(table):
    fe = col_to_float_array(table, "fe_h")
    mgfe = col_to_float_array(table, "mg_h") - fe
    return fe, mgfe


def mgfe_dividing_line(feh_vals,
                       fe_break=FE_BREAK,
                       m1=M1, b1=B1,
                       m2=M2, b2=B2):
    feh_vals = np.asarray(feh_vals, float)
    return np.where(feh_vals >= fe_break, m1 * feh_vals + b1, m2 * feh_vals + b2)


def apply_low_alpha_cut(table, x=None, y=None, z=None, feh_cut=FEH_CUT):
    """
    Apply the low-alpha + metallicity cut to a selected table.
    Optionally also apply the same mask to x, y, z arrays.
    """
    fe, mgfe = compute_fe_mgfe(table)

    finite_alpha = np.isfinite(fe) & np.isfinite(mgfe)
    fe0 = fe[finite_alpha]
    mgfe0 = mgfe[finite_alpha]

    mgfe_div0 = mgfe_dividing_line(fe0)

    low_alpha_mask = (mgfe0 < mgfe_div0)
    metal_mask = (fe0 >= feh_cut)
    chem_cut_mask = low_alpha_mask & metal_mask

    idx0 = np.where(finite_alpha)[0]
    table_cut = table[idx0[chem_cut_mask]]

    result = {
        "table": table_cut,
        "fe": fe0[chem_cut_mask],
        "mgfe": mgfe0[chem_cut_mask],
        "finite_alpha_mask": finite_alpha,
        "chem_cut_mask": chem_cut_mask,
    }

    if x is not None and y is not None and z is not None:
        x0, y0, z0 = x[finite_alpha], y[finite_alpha], z[finite_alpha]
        result["x"] = x0[chem_cut_mask]
        result["y"] = y0[chem_cut_mask]
        result["z"] = z0[chem_cut_mask]

    return result

def compute_alfe_mgmn(table):
    alfe = col_to_float_array(table, "al_h") - col_to_float_array(table, "fe_h")
    mgmn = col_to_float_array(table, "mg_h") - col_to_float_array(table, "mn_h")
    return alfe, mgmn


def apply_alfe_mgmn_polygon_cut(table, poly_vertices=ALFE_MGMN_POLY_VERTICES):
    """
    Apply the [Al/Fe] vs [Mg/Mn] polygon cut.
    Returns the cut table plus useful arrays and diagnostics.
    """
    needed = ["al_h", "fe_h", "mg_h", "mn_h"]
    missing = [c for c in needed if c not in table.colnames]
    if missing:
        raise KeyError(f"Missing columns in table: {missing}. Check table.colnames.")

    alfe_all, mgmn_all = compute_alfe_mgmn(table)

    finite = np.isfinite(alfe_all) & np.isfinite(mgmn_all)

    x = alfe_all[finite]
    y = mgmn_all[finite]

    poly_path = Path(poly_vertices, closed=True)

    pts_finite = np.vstack([x, y]).T
    inside_finite = poly_path.contains_points(pts_finite)

    inside_full = np.zeros(len(table), dtype=bool)
    inside_full[np.where(finite)[0]] = inside_finite

    table_cut = table[inside_full]

    diagnostics = {
        "input_size": len(table),
        "finite_points": int(np.sum(finite)),
        "inside_polygon_finite": int(np.sum(inside_finite)),
        "inside_fraction_finite": (
            float(np.sum(inside_finite) / np.sum(finite)) if np.sum(finite) > 0 else np.nan
        ),
        "output_size": len(table_cut),
    }

    arrays = {
        "alfe_all": alfe_all,
        "mgmn_all": mgmn_all,
        "finite_mask": finite,
        "inside_full_mask": inside_full,
        "x_finite": x,
        "y_finite": y,
        "poly_vertices": poly_vertices,
    }

    return table_cut, diagnostics, arrays

