"""The (R, phi) cells the midplane is fit in.

Every cell is an annular sector

    R_lo <= R < R_hi   and   phi_lo <= phi < phi_hi

in Galactocentric cylindrical R [kpc] and phi = arctan2(y, x) [rad, in 0..2 pi; the Sun is at
phi = pi]. Cells come from a quadtree: the region is cut into root sectors (ROOT_R_EDGES x
ROOT_PHI_EDGES), and a sector is split into 4 (halving R and phi) only if every child is

  - usable:   `usable(z)` is True for the z of its stars (enough stars, on both sides of the plane)
  - resolved: its smaller side (radial width or arc length) is at least n_sigma times the
              median distance error of its stars

So cells are small near the Sun, where stars are many and distances precise, and large far
away. Splitting looks only at positions, z and distance errors, never at [Mg/Fe].

SampleSelection.ipynb builds the cell table for each sample and saves it next to the sample
file; analysis code reads the table and fits the cells with ok == True.
"""

import numpy as np
from astropy.table import Table

ROOT_R_EDGES = np.arange(0.5, 18.5 + 1e-3, 2.0)  # kpc
ROOT_PHI_EDGES = np.deg2rad(np.arange(120.0, 240.0 + 1e-3, 30.0))  # rad
N_LEVELS = 2  # splits allowed below a root: 2 kpc x 30 deg -> 1 x 15 -> 0.5 x 7.5

# (R [kpc], phi [deg]) inside the cells shown in the [Mg/Fe] vs z figure, either side of the Sun
EXAMPLE_POINTS = [(8.75, 163.6), (7.25, 180.0)]
SUN_POINT = (8.25, 180.0)  # inside the Sun's cell (just inside R_sun = 8.275 kpc)


def build_cells(R, phi, z, sigma_d, usable, n_sigma=2.0):
    """Quadtree cells for these stars, as a Table with one row per cell, sorted by R then phi.

    Columns: cell_id, R_lo, R_hi, phi_lo, phi_hi, level (0 = root), n_stars, ok (usable),
    R_mean, phi_mean (mean position of the cell's stars).
    """
    R, phi, z, sigma_d = (np.asarray(a, float) for a in (R, phi, z, sigma_d))
    phi = np.mod(phi, 2 * np.pi)
    rows = []

    def stars_in(idx, R_lo, R_hi, phi_lo, phi_hi):
        m = (R[idx] >= R_lo) & (R[idx] < R_hi) & (phi[idx] >= phi_lo) & (phi[idx] < phi_hi)
        return idx[m]

    def splits_ok(R_lo, R_hi, phi_lo, phi_hi, idx):
        size = min(R_hi - R_lo, 0.5 * (R_lo + R_hi) * (phi_hi - phi_lo))
        # NaN median (no distance errors) compares False, so the cell is not split
        return len(idx) > 0 and usable(z[idx]) and size >= n_sigma * np.nanmedian(sigma_d[idx])

    def visit(R_lo, R_hi, phi_lo, phi_hi, idx, level):
        if level < N_LEVELS:
            R_mid, phi_mid = 0.5 * (R_lo + R_hi), 0.5 * (phi_lo + phi_hi)
            kids = [
                (a, b, c, d, stars_in(idx, a, b, c, d))
                for a, b in ((R_lo, R_mid), (R_mid, R_hi))
                for c, d in ((phi_lo, phi_mid), (phi_mid, phi_hi))
            ]
            if all(splits_ok(*kid) for kid in kids):
                for kid in kids:
                    visit(*kid, level + 1)
                return
        n = len(idx)
        rows.append(
            (R_lo, R_hi, phi_lo, phi_hi, level, n, n > 0 and bool(usable(z[idx])),
             R[idx].mean() if n else np.nan, phi[idx].mean() if n else np.nan)
        )

    everything = np.flatnonzero(np.isfinite(R) & np.isfinite(phi))
    for R_lo, R_hi in zip(ROOT_R_EDGES[:-1], ROOT_R_EDGES[1:]):
        for phi_lo, phi_hi in zip(ROOT_PHI_EDGES[:-1], ROOT_PHI_EDGES[1:]):
            visit(R_lo, R_hi, phi_lo, phi_hi, stars_in(everything, R_lo, R_hi, phi_lo, phi_hi), 0)

    names = ["R_lo", "R_hi", "phi_lo", "phi_hi", "level", "n_stars", "ok", "R_mean", "phi_mean"]
    cells = Table(rows=rows, names=names)
    cells.sort(["R_lo", "phi_lo"])
    cells.add_column(np.arange(len(cells)), name="cell_id", index=0)
    for name in ["R_lo", "R_hi", "R_mean"]:
        cells[name].unit = "kpc"
    for name in ["phi_lo", "phi_hi", "phi_mean"]:
        cells[name].unit = "rad"
    return cells


def assign(cells, R, phi):
    """cell_id of each star, or -1 for stars outside every cell."""
    R = np.asarray(R, float)
    phi = np.mod(np.asarray(phi, float), 2 * np.pi)
    cell_id = np.full(len(R), -1)
    for c in cells:
        inside = (R >= c["R_lo"]) & (R < c["R_hi"]) & (phi >= c["phi_lo"]) & (phi < c["phi_hi"])
        cell_id[inside] = c["cell_id"]
    return cell_id


def outline(cell, n=50):
    """R [kpc] and phi [rad] tracing the closed boundary of a cell (a row of the cell table)."""
    p = np.linspace(cell["phi_lo"], cell["phi_hi"], n)
    R = np.r_[np.full(n, cell["R_hi"]), np.full(n, cell["R_lo"]), cell["R_hi"]]
    return R, np.r_[p, p[::-1], cell["phi_lo"]]


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 200_000
    R = rng.uniform(0, 20, n)
    phi = np.deg2rad(rng.uniform(100, 260, n))
    z = rng.normal(0, 0.5, n)
    d = np.hypot(R * np.cos(phi) + 8.275, R * np.sin(phi))  # distance from the Sun
    sigma_d = 0.05 * d

    # with everything usable and no distance errors, every root splits all the way down
    full = build_cells(R, phi, z, np.zeros(n), usable=lambda zz: True)
    assert len(full) == 9 * 4 * 4**N_LEVELS and (full["level"] == N_LEVELS).all()

    # realistic rules: cells tile the root region exactly once, counts and flags are consistent
    usable = lambda zz: len(zz) >= 200 and (zz < -0.1).sum() >= 20 and (zz > 0.1).sum() >= 20
    cells = build_cells(R, phi, z, sigma_d, usable)
    in_roots = (
        (R >= ROOT_R_EDGES[0]) & (R < ROOT_R_EDGES[-1])
        & (phi >= ROOT_PHI_EDGES[0]) & (phi < ROOT_PHI_EDGES[-1])
    )
    hits = np.zeros(n, int)
    for c in cells:
        hits += (R >= c["R_lo"]) & (R < c["R_hi"]) & (phi >= c["phi_lo"]) & (phi < c["phi_hi"])
    assert (hits == in_roots).all(), "cells overlap or leave gaps"
    cid = assign(cells, R, phi)
    assert (np.bincount(cid[cid >= 0], minlength=len(cells)) == cells["n_stars"]).all()
    for c in cells:
        assert c["ok"] == (c["n_stars"] > 0 and usable(z[cid == c["cell_id"]]))
    # distance errors grow away from the Sun, so cells there are split less
    d_cell = np.hypot(cells["R_mean"] * np.cos(cells["phi_mean"]) + 8.275,
                      cells["R_mean"] * np.sin(cells["phi_mean"]))
    assert cells["level"][d_cell < 3].mean() > cells["level"][d_cell > 8].mean()
    print(f"ok: {len(cells)} cells, levels {np.bincount(cells['level'])}, {int(cells['ok'].sum())} usable")
