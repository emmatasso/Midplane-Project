import numpy as np


def poggio_m1_global_Zw(R, phi,
                        Rw=5.5,
                        A0=0.012,
                        alpha=1.9,
                        phiLON0_deg=0.1,
                        RT=12.1,
                        beta_deg_per_kpc=9.9):
    """
    Global Poggio m=1 warp model:
        Z_w(R, phi) = A(R) * sin(phi - phi_LON(R))
    """
    R = np.asarray(R, float)
    phi = np.asarray(phi, float)

    A = np.zeros_like(R)
    m = np.isfinite(R) & (R > Rw)
    A[m] = A0 * (R[m] - Rw) ** alpha

    phiLON0 = np.deg2rad(phiLON0_deg)
    beta = np.deg2rad(beta_deg_per_kpc)
    phiLON = phiLON0 + beta * np.maximum(R - RT, 0.0)

    return A * np.sin(phi - phiLON)

