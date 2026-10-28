"""The (R, phi) cells used in the analysis.

The midplane is fit separately in each cell: 0.5 kpc annuli in Galactocentric
cylindrical radius R, crossed with 11 azimuthal segments spanning 180 +/- 45 deg
(the Sun is at phi = 180 deg, with phi = arctan2(y, x)). A cell is

    R_EDGES[i] <= R < R_EDGES[i + 1]   and   PHI_EDGES[j] <= phi < PHI_EDGES[j + 1]

with i = 0..N_R-1 and j = 0..N_SEG-1. Plots and tables call segment j "segment j + 1".
"""

import numpy as np

R_EDGES = np.arange(0.5, 18.5 + 1e-3, 0.5)  # kpc

PHI_EDGES = np.deg2rad(np.arange(180 - 60.0, 180 + 60 + 1e-3, 8.0))
N_R, N_SEG = len(R_EDGES) - 1, len(PHI_EDGES) - 1

# (i, j) of the two cells shown in the [Mg/Fe] vs z figure, on either side of the Sun
EXAMPLE_CELLS = [
    (16, 3),
    (13, 5),
]  # R 8.5-9.0 kpc in segment 4, R 7.0-7.5 kpc in segment 6


def cell_index(R, phi):
    """Annulus index i and segment index j of each star, or -1 where outside the grid.

    The two are independent: i depends only on R and j only on phi (radians, any branch).
    """
    phi = np.mod(phi, 2 * np.pi)
    i = np.searchsorted(R_EDGES, R, side="right") - 1
    j = np.searchsorted(PHI_EDGES, phi, side="right") - 1
    i[(i < 0) | (i >= N_R)] = -1  # NaN sorts past the end, so it lands here too
    j[(j < 0) | (j >= N_SEG)] = -1
    return i, j


def outline(R_lo, R_hi, phi_lo, phi_hi, n=50):
    """Galactocentric x, y [kpc] tracing the closed boundary of an annular sector."""
    p = np.linspace(phi_lo, phi_hi, n)
    R = np.r_[np.full(n, R_hi), np.full(n, R_lo), R_hi]
    p = np.r_[p, p[::-1], phi_lo]
    return R * np.cos(p), R * np.sin(p)


def cell_outline(i, j, n=50):
    """Galactocentric x, y [kpc] tracing the boundary of cell (i, j)."""
    return outline(R_EDGES[i], R_EDGES[i + 1], PHI_EDGES[j], PHI_EDGES[j + 1], n)
