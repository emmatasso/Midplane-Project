import numpy as np

GB_VERTICES = np.array([
    [5100, 2.20],
    [4700, 1.85],
    [4300, 1.25],
    [3900, 0.70],
    [3400, 0.20],
    [3050, -0.10],
    [3050, 0.80],
    [3500, 1.30],
    [4000, 2.00],
    [4400, 2.75],
    [4700, 3.45],
    [4850, 3.90],
    [5000, 4.00],
    [5100, 4.00],
], float)

SNR_MIN = 20
TEFF_RANGE = (3000, 7500)
LOGG_RANGE = (-2, 6.5)

ALFE_MGMN_POLY_VERTICES = np.array([
    [-0.4, -0.25],   # lower-left
    [ 0.15, -0.40],  # lower-right
    [ 0.6,  0.7],    # upper-right
    [-0.2,  0.70],   # upper-left
], dtype=float)