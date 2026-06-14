import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def plot_xy_colored_by_z(x, y, z,
                         sun_x=-8.275, sun_y=0,
                         figsize=(10, 8),
                         xlim=(-25, 0),
                         ylim=(-50, 50),
                         title="Final sample - Galactocentric XY"):
    plt.figure(figsize=figsize)
    plt.scatter(x, y, c=z, s=1, alpha=0.7, cmap="viridis")
    plt.colorbar(label="Z [kpc]")
    plt.scatter(
        sun_x, sun_y,
        marker="x", s=60, c="red", linewidths=3,
        label="Sun", zorder=5
    )

    plt.xlabel("X [kpc]")
    plt.ylabel("Y [kpc]")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.tight_layout()
    plt.show()

def plot_mgfe_vs_feh(fe, mgfe, mgfe_dividing_line, feh_cut,
                     bins=200,
                     figsize=(8, 6),
                     cmap="viridis",
                     title="[Mg/Fe] vs [Fe/H] (final sample)"):
    plt.figure(figsize=figsize)
    plt.hist2d(fe, mgfe, bins=bins, norm=LogNorm(), cmap=cmap)
    plt.colorbar(label="Count")

    fe_grid = np.linspace(np.nanmin(fe), np.nanmax(fe), 500)
    plt.plot(fe_grid, mgfe_dividing_line(fe_grid),
             color="red", lw=2, label=r"$\alpha$-division")
    plt.axvline(feh_cut, color="white", lw=2, ls="--",
                label=f"[Fe/H] = {feh_cut}")

    plt.text(
        0.02, 0.02,
        f"N = {len(fe)} (below α line AND [Fe/H] ≥ {feh_cut})",
        transform=plt.gca().transAxes,
        fontsize=11,
        va="bottom",
        color="white",
        bbox=dict(facecolor="grey", alpha=0.9, edgecolor="none", pad=4),
    )

    plt.xlabel("[Fe/H]")
    plt.ylabel("[Mg/Fe]")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def plot_kiel_diagram(teff, logg, finite_basic, gb_vertices,
                      bins=200,
                      figsize=(6, 5),
                      teff_range=(3000, 7500),
                      logg_range=(-2, 6.5),
                      cmap="viridis",
                      title="Kiel diagram (2D histogram)"):
    x_all = teff[finite_basic]
    y_all = logg[finite_basic]

    plt.figure(figsize=figsize)
    plt.hist2d(
        x_all, y_all,
        bins=bins,
        range=[teff_range, logg_range],
        norm=LogNorm(),
        cmap=cmap
    )
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.colorbar(label="Count")

    plt.xlabel("Teff [K]")
    plt.ylabel("log g")
    plt.xlim(teff_range[1], teff_range[0])
    plt.title(title)

    gb_closed = np.vstack([gb_vertices, gb_vertices[0]])
    plt.plot(gb_closed[:, 0], gb_closed[:, 1], "r--", lw=2, label="Giant-branch cut")
    plt.fill(gb_closed[:, 0], gb_closed[:, 1], color="red", alpha=0.06)

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_alfe_mgmn_polygon(x, y, poly_vertices,
                           bins=250,
                           figsize=(7, 5),
                           cmap="viridis",
                           title="Mg/Mn vs Al/Fe (final sample) + polygon cut (full quad)"):
    closed = np.vstack([poly_vertices, poly_vertices[0]])

    plt.figure(figsize=figsize)
    plt.hist2d(x, y, bins=bins, norm=LogNorm(), cmap=cmap)
    plt.colorbar(label="Count")
    plt.plot(closed[:, 0], closed[:, 1], color="red", lw=3)
    plt.fill(closed[:, 0], closed[:, 1], color="red", alpha=0.08)
    plt.xlabel("[Al/Fe]")
    plt.ylabel("[Mg/Mn]")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_density_2d(x, y,
                    xlabel="x",
                    ylabel="y",
                    title="2D density",
                    bins=300,
                    figsize=(8, 6),
                    cmap="viridis"):
    plt.figure(figsize=figsize)
    plt.hist2d(x, y, bins=bins, norm=LogNorm(), cmap=cmap)
    plt.colorbar(label="Count")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def plot_annuli_mgfe_vs_z(R_all, Z_all, mgfe_all,
                          dR=0.5,
                          minN=100,
                          stop_after_k_consecutive_lowN=2,
                          R_max_plot=None,
                          z_edges=None,
                          mgfe_edges=None,
                          smooth_sigma=(1.0, 1.0),
                          vmin_pct=2.0,
                          vmax_pct=99.7,
                          cmap_name="Greys",
                          ncols=4):
    """
    Multi-panel annuli plot of P(Mg/Fe | Z) plus mean Mg/Fe(Z).
    """
    if z_edges is None:
        z_edges = np.linspace(-3.0, 3.0, 181)
    if mgfe_edges is None:
        mgfe_edges = np.linspace(-0.2, 0.6, 161)

    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    Rmin_data = np.nanmin(R_all)
    Rmax_data = np.nanmax(R_all)

    Rstart = np.floor(Rmin_data / dR) * dR
    Rstop = np.ceil(Rmax_data / dR) * dR
    if R_max_plot is not None:
        Rstop = min(Rstop, R_max_plot)

    annuli = []
    imgs = []
    mean_trends = []
    Ns = []

    lowN_streak = 0
    seen_good = False

    for Rmin in np.arange(Rstart, Rstop, dR):
        Rmax = Rmin + dR
        m = (R_all >= Rmin) & (R_all < Rmax)
        N = int(np.sum(m))

        if N < minN:
            if seen_good:
                lowN_streak += 1
                if lowN_streak >= stop_after_k_consecutive_lowN:
                    print(f"Stopping: reached {lowN_streak} consecutive annuli with N < {minN} at R~[{Rmin:.1f},{Rmax:.1f})")
                    break
            continue
        else:
            seen_good = True
            lowN_streak = 0

        Z = Z_all[m]
        mgfe = mgfe_all[m]

        finite = np.isfinite(Z) & np.isfinite(mgfe)
        Z = Z[finite]
        mgfe = mgfe[finite]
        N = int(len(Z))
        if N < minN:
            continue

        H, _, _ = np.histogram2d(Z, mgfe, bins=[z_edges, mgfe_edges])
        colsum = H.sum(axis=1, keepdims=True)
        Hnorm = np.divide(H, colsum, out=np.zeros_like(H), where=(colsum > 0))
        img = gaussian_filter(Hnorm, sigma=smooth_sigma).T

        trend = np.full_like(z_centers, np.nan, dtype=float)
        for i in range(len(z_edges) - 1):
            mm = (Z >= z_edges[i]) & (Z < z_edges[i + 1])
            if np.any(mm):
                trend[i] = np.nanmean(mgfe[mm])

        annuli.append((Rmin, Rmax))
        imgs.append(img)
        mean_trends.append(trend)
        Ns.append(N)

    print(f"Plotted {len(annuli)} annuli (dR={dR}) with minN={minN}.")

    if len(annuli) == 0:
        raise RuntimeError(
            "No annuli met minN after filtering. "
            "Try lowering minN, relaxing parallax SNR, or verifying input units."
        )

    vals = np.concatenate([img[(img > 0) & np.isfinite(img)] for img in imgs])
    if len(vals) > 0:
        vmin, vmax = np.percentile(vals, [vmin_pct, vmax_pct])
    else:
        vmin, vmax = 0.0, 1.0

    n_panels = len(annuli)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 3.6 * nrows),
        sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    pcm = None

    for i, ax in enumerate(axes.flat):
        if i >= n_panels:
            ax.axis("off")
            continue

        Rmin, Rmax = annuli[i]
        img = imgs[i]
        trend = mean_trends[i]
        N = Ns[i]

        pcm = ax.pcolormesh(
            z_edges, mgfe_edges, img,
            shading="auto",
            vmin=vmin, vmax=vmax,
            cmap=cmap_name
        )

        ax.plot(z_centers, trend, color="red", lw=1.5)
        ax.axvline(0, color="k", lw=0.8, alpha=0.7)

        ax.text(
            0.03, 0.95,
            rf"${Rmin:.1f}\leq R<{Rmax:.1f}$ kpc" + "\n" + rf"$N={N}$",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
        )

        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.2, 0.6)
        ax.tick_params(axis="x", which="both", labelbottom=True)

    fig.supxlabel(r"$Z$ [kpc]", fontsize=14)
    fig.supylabel(r"[Mg/Fe]", fontsize=14)

    fig.tight_layout(rect=[0.0, 0.0, 0.92, 0.97])
    cax = fig.add_axes([0.935, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label(r"Column-normalized density")

    plt.show()

    return {
        "annuli": annuli,
        "imgs": imgs,
        "mean_trends": mean_trends,
        "Ns": Ns,
        "z_edges": z_edges,
        "mgfe_edges": mgfe_edges,
    }

import numpy as np
import matplotlib.pyplot as plt


def plot_segment_geometry_xy(X_star, Y_star, crowd_bounds, window_edges, annuli,
                             n_crowd=11,
                             seg_rest_id=None,
                             crowd_halfwidth_deg=45,
                             figsize=(9, 9),
                             sun_R0=8.275):
    """
    Plot the XY distribution with annuli and crowded-window segment boundaries.
    """
    if seg_rest_id is None:
        seg_rest_id = n_crowd + 1

    a, b = window_edges

    plt.figure(figsize=figsize)
    plt.scatter(X_star, Y_star, s=1, alpha=0.05, zorder=1)

    theta = np.linspace(0, 2 * np.pi, 800)

    for (Rmin, Rmax) in annuli:
        plt.plot(Rmin * np.cos(theta), Rmin * np.sin(theta), lw=0.8, alpha=0.55, zorder=2)
        plt.plot(Rmax * np.cos(theta), Rmax * np.sin(theta), lw=0.8, alpha=0.55, zorder=2)

        for ang in crowd_bounds:
            plt.plot([Rmin * np.cos(ang), Rmax * np.cos(ang)],
                     [Rmin * np.sin(ang), Rmax * np.sin(ang)],
                     color="k", lw=2.0, alpha=0.90, zorder=3)

    R_label = np.median([r[0] for r in annuli]) * 1.1

    mid_angles = []
    for i in range(n_crowd):
        ang0 = crowd_bounds[i]
        ang1 = crowd_bounds[i + 1]
        d = (ang1 - ang0) % (2 * np.pi)
        mid_angles.append((ang0 + 0.5 * d) % (2 * np.pi))

    comp_span = (a - b) % (2 * np.pi)
    mid_rest = (b + 0.5 * comp_span) % (2 * np.pi)

    for i, mid in enumerate(mid_angles, start=1):
        plt.text(R_label * np.cos(mid), R_label * np.sin(mid),
                 f"Seg {i}", fontsize=12, weight="bold",
                 ha="center", va="center")

    plt.text(R_label * np.cos(mid_rest), R_label * np.sin(mid_rest),
             f"Seg {seg_rest_id} (rest)", fontsize=12, weight="bold",
             ha="center", va="center")

    plt.scatter(-sun_R0, 0, marker="x", s=80, c="purple",
                linewidths=3, zorder=5, label="Sun")

    plt.gca().set_aspect("equal", "box")
    plt.xlabel("X [kpc]")
    plt.ylabel("Y [kpc]")
    plt.title(f"Annuli split into {n_crowd+1} segments (crowded window ±{crowd_halfwidth_deg}°)")
    plt.grid(alpha=0.2)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from .fitting import binned_mean_and_counts, quad_fit_on_trend


def plot_single_annulus_segment_fits(
    R_star, Z_star, mgfe_star, segm,
    Rmin_focus, Rmax_focus,
    z_edges, mgfe_edges,
    seg_plot,
    minN_seg_total=10,
    min_bin_count=5,
    z_fit_min=-1.5,
    z_fit_max=1.5,
    min_bins_for_fit=8,
    smooth_sigma_img=(1.0, 1.0),
    xlim=(-3, 3),
    ylim=(-0.2, 0.6),
    ncols=4,
    cmap="Greys"
):
    """
    Plot one annulus with one panel per segment:
      - background P(Mg/Fe | Z) for all stars in annulus
      - mean Mg/Fe(Z) per segment
      - quadratic fit
      - fitted z_min if valid
    """
    mR = (R_star >= Rmin_focus) & (R_star < Rmax_focus)
    Ntot = int(np.sum(mR))
    print(f"Annulus {Rmin_focus:.1f}–{Rmax_focus:.1f} kpc: Ntot = {Ntot}")

    if Ntot == 0:
        raise RuntimeError("No stars in that annulus (check R_star units / cuts).")

    Zr_all = Z_star[mR]
    mgr_all = mgfe_star[mR]
    good_bg = np.isfinite(Zr_all) & np.isfinite(mgr_all)
    Zr_all = Zr_all[good_bg]
    mgr_all = mgr_all[good_bg]

    # background image using all stars in annulus
    H, _, _ = np.histogram2d(Zr_all, mgr_all, bins=[z_edges, mgfe_edges])
    colsum = H.sum(axis=1, keepdims=True)
    Hnorm = np.divide(H, colsum, out=np.zeros_like(H), where=(colsum > 0))
    img = gaussian_filter(Hnorm, sigma=smooth_sigma_img).T

    vals = img[(img > 0) & np.isfinite(img)]
    if vals.size > 0:
        vmin, vmax = np.percentile(vals, [2.0, 99.7])
    else:
        vmin, vmax = 0.0, 1.0

    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    nseg = len(seg_plot)
    nrows = int(np.ceil(nseg / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.6 * ncols, 3.8 * nrows),
        sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    pcm = None
    segment_results = []

    for j, ax in enumerate(axes.flat):
        if j >= nseg:
            ax.axis("off")
            continue

        seg_id = seg_plot[j]
        mseg = mR & segm[seg_id]
        Nseg = int(np.sum(mseg))

        pcm = ax.pcolormesh(
            z_edges, mgfe_edges, img,
            shading="auto",
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            alpha=0.9
        )

        if Nseg < minN_seg_total:
            ax.text(
                0.05, 0.95,
                f"Seg {seg_id}\nN={Nseg}\n(too few)",
                transform=ax.transAxes,
                ha="left", va="top", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
            )
            ax.axvline(0, color="k", lw=0.8, alpha=0.6)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            segment_results.append({
                "seg_id": seg_id,
                "Nseg": Nseg,
                "status": "too_few_stars",
                "zmin": np.nan
            })
            continue

        Zs = Z_star[mseg]
        Ys = mgfe_star[mseg]
        good = np.isfinite(Zs) & np.isfinite(Ys)
        Zs, Ys = Zs[good], Ys[good]

        mean_line, cnt_line = binned_mean_and_counts(Zs, Ys, z_edges)

        (a, b, c), st, z0 = quad_fit_on_trend(
            z_centers, mean_line, cnt_line,
            zmin=z_fit_min, zmax=z_fit_max,
            min_cnt=min_bin_count, min_bins=min_bins_for_fit
        )

        ax.plot(z_centers, mean_line, lw=2.0)

        if np.isfinite(a) and np.isfinite(b) and np.isfinite(c):
            zfit = np.linspace(z_fit_min, z_fit_max, 200)
            yfit = a * zfit**2 + b * zfit + c
            ax.plot(zfit, yfit, color="k", lw=2.0, alpha=0.85)

        if np.isfinite(z0) and st == "ok":
            ax.axvline(z0, color="k", ls="--", lw=1.6, alpha=0.9)

        ax.axvline(0, color="k", lw=0.8, alpha=0.6)

        ax.text(
            0.05, 0.95,
            f"Seg {seg_id}\nN={Nseg}\nfit: {st}"
            + (f"\nzmin={z0:+.3f}" if (np.isfinite(z0) and st == "ok") else ""),
            transform=ax.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        segment_results.append({
            "seg_id": seg_id,
            "Nseg": Nseg,
            "status": st,
            "zmin": z0
        })

    fig.suptitle(
        f"[Mg/Fe] vs Z — annulus {Rmin_focus:.1f}–{Rmax_focus:.1f} kpc (mean + quadratic fit)",
        fontsize=16, y=0.995
    )
    fig.supylabel("[Mg/Fe]")
    fig.supxlabel("Z [kpc]")

    fig.subplots_adjust(
        left=0.07, right=0.88, bottom=0.08, top=0.92,
        wspace=0.22, hspace=0.25
    )
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label("Column-normalized density (all stars in annulus)")

    plt.show()
    plt.close(fig)

    return segment_results

import numpy as np
import matplotlib.pyplot as plt


def plot_zmin_polar_heatmap(
    Zmap,
    theta_edges,
    R_edges_kept,
    zlim=0.6,
    figsize=(8.5, 7.5),
    title=r"Polar heatmap of quadratic-fit $z_{\min}$",
    cbar_label=r"Quadratic-fit $z_{\min}$ [kpc]",
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="lightgray", alpha=0.6)

    TH, RR = np.meshgrid(theta_edges, R_edges_kept)

    pcm = ax.pcolormesh(
        TH, RR, Zmap,
        shading="flat",
        cmap=cmap,
        vmin=-zlim,
        vmax=+zlim
    )

    cb = fig.colorbar(pcm, ax=ax, pad=0.10)
    cb.set_label(cbar_label)

    ax.set_title(title, y=1.08)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def plot_warp_comparison_3panel(Zmap_data, Zmap_model, Zmap_sub,
                                theta_edges, R_edges_plot,
                                zlim=0.6,
                                figsize=(18, 6),
                                panel_titles=None):
    """
    Plot 3 polar panels:
      1) data quadratic-fit z_min
      2) Poggio model
      3) data after subtracting model
    """
    if panel_titles is None:
        panel_titles = [
            r"Data: quadratic-fit $z_{\min}$",
            r"Poggio $m=1$ warp model",
            r"Data after subtracting Poggio model"
        ]

    all_vals = np.concatenate([
        Zmap_data[np.isfinite(Zmap_data)],
        Zmap_model[np.isfinite(Zmap_model)],
        Zmap_sub[np.isfinite(Zmap_sub)],
    ])

    if len(all_vals) > 0:
        vmax = np.nanpercentile(np.abs(all_vals), 98)
        vmax = min(vmax, zlim)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = zlim
    else:
        vmax = zlim

    fig, axes = plt.subplots(
        1, 3,
        subplot_kw={"projection": "polar"},
        figsize=figsize,
        constrained_layout=True
    )

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="lightgray", alpha=0.6)

    TH, RR = np.meshgrid(theta_edges, R_edges_plot)

    panel_maps = [Zmap_data, Zmap_model, Zmap_sub]

    pcm = None
    for ax, Zmap_i, title in zip(axes, panel_maps, panel_titles):
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)

        pcm = ax.pcolormesh(
            TH, RR, Zmap_i,
            shading="flat",
            cmap=cmap,
            vmin=-vmax,
            vmax=+vmax
        )

        ax.set_title(title, y=1.08, fontsize=13)

    cb = fig.colorbar(pcm, ax=axes, pad=0.08, shrink=0.9)
    cb.set_label(r"Vertical displacement [kpc]")

    plt.show()

def plot_warp_comparison_2x3(
    Zmap_data,
    Zmap_model,
    Zmap_sub,
    Emap_data,
    Emap_sub,
    theta_edges,
    R_edges_plot,
    zlim,
    data_title="Data: quadratic-fit $z_{min}$",
    model_title="Poggio $m=1$ warp model",
    sub_title="Data after subtracting Poggio model",
    figsize=(18, 11),
):
    """
    Plot 2x3 polar comparison:
      top row    = data / model / data-model
      bottom row = uncertainty(data) / blank / uncertainty(data-model)
    """
    all_vals = np.concatenate([
        Zmap_data[np.isfinite(Zmap_data)],
        Zmap_model[np.isfinite(Zmap_model)],
        Zmap_sub[np.isfinite(Zmap_sub)],
    ])

    if len(all_vals) > 0:
        vmax = np.nanpercentile(np.abs(all_vals), 98)
        vmax = min(vmax, zlim)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = zlim
    else:
        vmax = zlim

    all_err = np.concatenate([
        Emap_data[np.isfinite(Emap_data)],
        Emap_sub[np.isfinite(Emap_sub)],
    ])

    if len(all_err) > 0:
        emax = np.nanpercentile(all_err, 98)
        if not np.isfinite(emax) or emax == 0:
            emax = 0.1
    else:
        emax = 0.1

    fig, axes = plt.subplots(
        2, 3,
        subplot_kw={"projection": "polar"},
        figsize=figsize,
        constrained_layout=True
    )

    cmap_z = plt.cm.RdBu_r.copy()
    cmap_z.set_bad(color="lightgray", alpha=0.6)

    cmap_e = plt.cm.magma_r.copy()
    cmap_e.set_bad(color="lightgray", alpha=0.6)

    TH, RR = np.meshgrid(theta_edges, R_edges_plot)

    top_maps = [Zmap_data, Zmap_model, Zmap_sub]
    top_titles = [
    data_title,
    model_title,
    sub_title]

    pcm_top = None
    for ax, Zmap_i, title in zip(axes[0], top_maps, top_titles):
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)

        pcm_top = ax.pcolormesh(
            TH, RR, Zmap_i,
            shading="flat",
            cmap=cmap_z,
            vmin=-vmax,
            vmax=+vmax
        )
        ax.set_title(title, y=1.08, fontsize=13)

    bottom_maps = [Emap_data, None, Emap_sub]
    bottom_titles = [
        r"Uncertainty in inferred $z_{\min}$",
        "",
        r"Uncertainty in inferred $z_{\min}$"
    ]

    pcm_bottom = None
    for ax, Emap_i, title in zip(axes[1], bottom_maps, bottom_titles):
        if Emap_i is None:
            ax.set_axis_off()
            continue

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)

        pcm_bottom = ax.pcolormesh(
            TH, RR, Emap_i,
            shading="flat",
            cmap=cmap_e,
            vmin=0,
            vmax=emax
        )
        ax.set_title(title, y=1.08, fontsize=13)

    cb1 = fig.colorbar(pcm_top, ax=axes[0, :], pad=0.06, shrink=0.9)
    cb1.set_label(r"Vertical displacement [kpc]")

    cb2 = fig.colorbar(pcm_bottom, ax=[axes[1, 0], axes[1, 2]], pad=0.06, shrink=0.9)
    cb2.set_label(r"Propagated uncertainty in $z_{\min}$ [kpc]")

    plt.show()

def plot_poggio_median_z_map(Zmed, xedges, yedges,
                             vlim=0.30,
                             figsize=(5.2, 5.2),
                             title="Data (Young giants)\nMedian Z (Sun-centered)",
                             xlim=(-9, 23),
                             cmap=None):
    import matplotlib.pyplot as plt

    if cmap is None:
        cmap = plt.cm.RdBu_r

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        Zmed,
        origin="lower",
        cmap=cmap,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        vmin=-vlim, vmax=+vlim,
        interpolation="nearest",
        aspect="equal"
    )

    ax.plot(0, 0, marker="+", color="black", markersize=10, mew=1.5)

    ax.set_xlabel("X (kpc)")
    ax.set_ylabel("Y (kpc)")
    ax.set_title(title)
    ax.set_xlim(*xlim)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Median Z (kpc)")

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def plot_my_polar_vs_their_cartesian(
    Zmap, theta_edges, R_edges_kept, zlim,
    Zmed, xedges, yedges,
    bin_kpc=0.25,
    vlim_their=0.30,
    figsize=(13.5, 6.0),
    my_title="My sample (polar heatmap)",
    their_title=None,
    show_segment_labels=True
):
    """
    Side-by-side comparison:
      left  = my polar heatmap
      right = their Cartesian median-Z map
    """
    if their_title is None:
        their_title = f"Their sample (Cartesian): median Z per {bin_kpc:.2f} kpc bin"

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.25)

    # --------------------------
    # LEFT: MY POLAR HEATMAP
    # --------------------------
    ax0 = fig.add_subplot(gs[0, 0], projection="polar")
    ax0.set_theta_zero_location("E")
    ax0.set_theta_direction(1)

    cmap0 = plt.cm.RdBu_r.copy()
    cmap0.set_bad(color="lightgray", alpha=0.6)

    TH, RR = np.meshgrid(theta_edges, R_edges_kept)

    pcm0 = ax0.pcolormesh(
        TH, RR, Zmap,
        cmap=cmap0,
        vmin=-zlim,
        vmax=+zlim,
        shading="auto"
    )

    cbar_my = fig.colorbar(
        pcm0,
        ax=ax0,
        fraction=0.05,
        pad=0.08
    )
    cbar_my.set_label(r"$z_{\min}$ or vertical displacement (kpc)")
    ax0.set_title(my_title, y=1.08)

    if show_segment_labels:
        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        r_label = R_edges_kept[-1] * 1.05
        for j, th in enumerate(theta_centers):
            ax0.text(th, r_label, f"S{j+1}",
                     ha="center", va="center",
                     fontsize=9, weight="bold")

    # --------------------------
    # RIGHT: THEIR CARTESIAN MAP
    # --------------------------
    ax1 = fig.add_subplot(gs[0, 1])

    cmap_match = plt.cm.RdBu_r
    im1 = ax1.imshow(
        Zmed,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap_match,
        vmin=-vlim_their,
        vmax=+vlim_their,
        interpolation="nearest",
        aspect="equal"
    )

    ax1.set_xlim(-9, 23)
    xy_lim = max(abs(yedges[0]), abs(yedges[-1]))
    ax1.set_ylim(-xy_lim, xy_lim)

    # Galactocentric circles
    R0 = 8.2
    X_gc, Y_gc = +R0, 0.0

    radii = [2, 4, 6, 8, 10, 12, 14, 16]
    theta = np.linspace(0, 2*np.pi, 600)

    for R in radii:
        ax1.plot(
            X_gc + R * np.cos(theta),
            Y_gc + R * np.sin(theta),
            linestyle=":",
            color="black",
            lw=1.0,
            alpha=0.7
        )

    ax1.plot(X_gc, Y_gc, marker="x", color="black", ms=6, mew=1.5)

    # draw 3 radial lines using your polar wedge geometry
    phi0 = float(theta_edges[0])
    phi1 = float(theta_edges[-1])
    phi_mid = 0.5 * (phi0 + phi1)
    phi_draw = [phi0, phi_mid, phi1]

    r_max = 16.0
    for phi in phi_draw:
        x_line = X_gc + np.array([0.0, r_max]) * np.cos(phi)
        y_line = Y_gc + np.array([0.0, r_max]) * np.sin(phi)
        ax1.plot(x_line, y_line, color="k", lw=0.7, alpha=0.9, zorder=1)

    ax1.set_xlabel("X (kpc)")
    ax1.set_ylabel("Y (kpc)")
    ax1.set_title(their_title)

    cbar_their = fig.colorbar(
        im1,
        ax=ax1,
        fraction=0.046,
        pad=0.04
    )
    cbar_their.set_label("Median Z (kpc)")

    plt.subplots_adjust(wspace=0.35)
    plt.show()

from matplotlib.patches import Circle

def plot_my_polar_vs_poggio_polar(
    Zmap, theta_edges, R_edges_kept, zlim,
    Zmed_pog_polar,
    vlim_pog=0.60,
    figsize=(14.5, 6.8),
    my_title="My sample (polar heatmap)",
    poggio_title="Poggio sample in my bins"
):
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="lightgray", alpha=0.6)

    fig = plt.figure(figsize=figsize)

    axL = fig.add_subplot(1, 2, 1, projection="polar")
    axL.set_theta_zero_location("E")
    axL.set_theta_direction(1)

    TH, RR = np.meshgrid(theta_edges, R_edges_kept)
    pcmL = axL.pcolormesh(TH, RR, Zmap, shading="flat", cmap=cmap,
                          vmin=-zlim, vmax=+zlim)
    axL.set_title(my_title)

    cbL = fig.colorbar(pcmL, ax=axL, pad=0.10, fraction=0.05)
    cbL.set_label(r"My $z_{\min}$ [kpc]")

    axP = fig.add_subplot(1, 2, 2, projection="polar")
    axP.set_theta_zero_location("E")
    axP.set_theta_direction(1)

    TH2, RR2 = np.meshgrid(theta_edges, R_edges_kept)
    pcmP = axP.pcolormesh(TH2, RR2, Zmed_pog_polar, shading="flat", cmap=cmap,
                          vmin=-vlim_pog, vmax=+vlim_pog)
    axP.set_title(poggio_title)

    cbP = fig.colorbar(pcmP, ax=axP, pad=0.10, fraction=0.05)
    cbP.set_label("Poggio median Z [kpc]")

    plt.tight_layout()
    plt.show()

def plot_poggio_cartesian_guides(
    Zmed_xy, xedges, yedges, theta_edges,
    R0=8.2,
    ring_radii=(10, 12, 14),
    xlimR=(-8, 23),
    ylimR=(-16, 16),
    vlim=0.60,
    figsize=(6.3, 6.0),
    title="Poggio sample"
):
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="lightgray", alpha=0.6)

    theta_left = float(theta_edges[0])
    theta_right = float(theta_edges[-1])
    theta_mid = 0.5 * (theta_left + theta_right)
    theta_lines = [theta_left, theta_mid, theta_right]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        Zmed_xy,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap,
        vmin=-vlim, vmax=+vlim,
        interpolation="nearest",
        aspect="equal"
    )

    ax.plot(0, 0, marker="+", color="black", ms=10, mew=1.5)   # Sun
    ax.plot(R0, 0, marker="x", color="black", ms=6, mew=1.5)   # GC marker in Sun-centered XY

    for rr in ring_radii:
        ax.add_patch(Circle((R0, 0), rr, fill=False, ls=":", lw=1.0, ec="gray", alpha=0.9))

    ray_len = 40.0
    for th in theta_lines:
        x2 = R0 + ray_len * np.cos(th)
        y2 = 0  + ray_len * np.sin(th)
        ax.plot([R0, x2], [0, y2], color="black", lw=2)

    ax.set_xlim(*xlimR)
    ax.set_ylim(*ylimR)
    ax.set_xlabel("X (kpc)")
    ax.set_ylabel("Y (kpc)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Median Z (kpc)")

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def plot_quad_vs_spline_scatter(
    z_quad,
    z_spline,
    figsize=(6.5, 6.5),
    title="Spline vs quadratic midplane estimates",
    xlabel=r"Quadratic-fit $z_{\min}$ [kpc]",
    ylabel=r"Spline-fit $z_{\min}$ [kpc]"
):
    z_quad = np.asarray(z_quad, float)
    z_spline = np.asarray(z_spline, float)

    good = np.isfinite(z_quad) & np.isfinite(z_spline)
    x = z_quad[good]
    y = z_spline[good]

    if len(x) == 0:
        raise RuntimeError("No valid points to plot in spline vs quadratic scatter.")

    lim = np.nanmax(np.abs(np.concatenate([x, y])))
    lim = max(lim, 0.1)
    lim *= 1.05

    plt.figure(figsize=figsize)
    plt.scatter(x, y, s=25, alpha=0.8)
    plt.plot([-lim, lim], [-lim, lim], "k--", lw=1.5, label="1:1 line")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_difference_polar_heatmap(
    Zdiff,
    theta_edges,
    R_edges_kept,
    diff_lim=None,
    figsize=(8.5, 7.5),
    title=r"Polar heatmap of spline$-$quadratic $z_{\min}$",
    cbar_label=r"$z_{\min}^{\rm spline} - z_{\min}^{\rm quad}$ [kpc]",
    vmin=None,
    vmax=None
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="lightgray", alpha=0.6)

    good = np.isfinite(Zdiff)
    if diff_lim is None:
        if np.any(good):
            diff_lim = np.nanpercentile(np.abs(Zdiff[good]), 98)
            if not np.isfinite(diff_lim) or diff_lim == 0:
                diff_lim = 0.1
        else:
            diff_lim = 0.1

    TH, RR = np.meshgrid(theta_edges, R_edges_kept)

    pcm = ax.pcolormesh(
        TH, RR, Zdiff,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    cb = fig.colorbar(pcm, ax=ax, pad=0.10)
    cb.set_label(cbar_label)

    ax.set_title(title, y=1.08)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

from .fitting import (
    binned_mean_and_counts,
    binned_mean_count_sem,
    spline_fit_on_trend,
)


def plot_single_annulus_spline_fits(
    R_star,
    Z_star,
    mgfe_star,
    segm,
    Rmin_focus,
    Rmax_focus,
    z_edges,
    seg_plot,
    minN_seg_total=15,
    min_bin_count=5,
    z_fit_min=-1.5,
    z_fit_max=1.5,
    min_bins_for_fit=8,
    use_sem_weights=False,
    spline_k=3,
    spline_s=None,
    n_eval=400,
    max_allowed_minima=1,
    edge_buffer=0.05,
    xlim=(-3, 3),
    ylim=(-0.2, 0.6),
    ncols=4,
    figsize_per_panel=(4.6, 3.8),
):
    """
    For one annulus, show one panel per segment with:
      - binned mean Mg/Fe vs z
      - optional SEM error bars
      - smoothing spline fit
      - inferred spline minimum

    Returns
    -------
    results : list of dict
    """
    mR = (R_star >= Rmin_focus) & (R_star < Rmax_focus)
    Ntot = int(np.sum(mR))
    print(f"Annulus {Rmin_focus:.1f}–{Rmax_focus:.1f} kpc: Ntot = {Ntot}")

    if Ntot == 0:
        raise RuntimeError("No stars in that annulus.")

    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    nseg = len(seg_plot)
    nrows = int(np.ceil(nseg / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    results = []

    for j, ax in enumerate(axes.flat):
        if j >= nseg:
            ax.axis("off")
            continue

        seg_id = seg_plot[j]
        mseg = mR & segm[seg_id]
        good = mseg & np.isfinite(Z_star) & np.isfinite(mgfe_star)
        Nseg = int(np.sum(good))

        if Nseg < minN_seg_total:
            ax.text(
                0.05, 0.95,
                f"Seg {seg_id}\nN={Nseg}\n(too few stars)",
                transform=ax.transAxes,
                ha="left", va="top", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
            )
            ax.axvline(0, color="k", lw=0.8, alpha=0.6)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            results.append({
                "seg_id": seg_id,
                "Nseg": Nseg,
                "status": "too_few_stars",
                "zmin": np.nan,
            })
            continue

        Zs = Z_star[good]
        Ys = mgfe_star[good]

        if use_sem_weights:
            mean_line, cnt_line, sem_line = binned_mean_count_sem(Zs, Ys, z_edges)
            fit_obj, st, z0 = spline_fit_on_trend(
                z_centers=z_centers,
                trend=mean_line,
                cnt=cnt_line,
                sem=sem_line,
                zmin=z_fit_min,
                zmax=z_fit_max,
                min_cnt=min_bin_count,
                min_bins=min_bins_for_fit,
                spline_k=spline_k,
                spline_s=spline_s,
                n_eval=n_eval,
                max_allowed_minima=max_allowed_minima,
                edge_buffer=edge_buffer,
            )
        else:
            mean_line, cnt_line = binned_mean_and_counts(Zs, Ys, z_edges)
            sem_line = None
            fit_obj, st, z0 = spline_fit_on_trend(
                z_centers=z_centers,
                trend=mean_line,
                cnt=cnt_line,
                sem=None,
                zmin=z_fit_min,
                zmax=z_fit_max,
                min_cnt=min_bin_count,
                min_bins=min_bins_for_fit,
                spline_k=spline_k,
                spline_s=spline_s,
                n_eval=n_eval,
                max_allowed_minima=max_allowed_minima,
                edge_buffer=edge_buffer,
            )

        # plot the binned data
        ok_pts = np.isfinite(mean_line)
        if use_sem_weights and sem_line is not None:
            ax.errorbar(
                z_centers[ok_pts],
                mean_line[ok_pts],
                yerr=sem_line[ok_pts],
                fmt="o",
                ms=3.5,
                lw=1.0,
                capsize=2,
                alpha=0.9,
            )
        else:
            ax.plot(
                z_centers[ok_pts],
                mean_line[ok_pts],
                "o",
                ms=4,
                alpha=0.9,
            )

        # plot spline if fit object exists
        if fit_obj is not None and "zgrid" in fit_obj and "ygrid" in fit_obj:
            ax.plot(
                fit_obj["zgrid"],
                fit_obj["ygrid"],
                lw=2.0,
                color="k",
                alpha=0.9,
            )

        # mark minimum
        if np.isfinite(z0) and st == "ok":
            ax.axvline(z0, color="k", ls="--", lw=1.6, alpha=0.9)

        ax.axvline(0, color="k", lw=0.8, alpha=0.6)

        label = f"Seg {seg_id}\nN={Nseg}\nfit: {st}"
        if np.isfinite(z0) and st == "ok":
            label += f"\nzmin={z0:+.3f}"

        if fit_obj is not None and "n_local_min" in fit_obj:
            label += f"\nmins={fit_obj['n_local_min']}"

        ax.text(
            0.05, 0.95,
            label,
            transform=ax.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        results.append({
            "seg_id": seg_id,
            "Nseg": Nseg,
            "status": st,
            "zmin": z0,
            "n_local_min": fit_obj["n_local_min"] if fit_obj is not None and "n_local_min" in fit_obj else np.nan,
        })

    fig.suptitle(
        f"[Mg/Fe] vs Z — annulus {Rmin_focus:.1f}–{Rmax_focus:.1f} kpc (smoothing spline)",
        fontsize=16, y=0.995
    )
    fig.supylabel("[Mg/Fe]")
    fig.supxlabel("Z [kpc]")
    fig.tight_layout()

    plt.show()
    return results

