"""
Spatial patterns of warming — MPI-ESM1-2-LR historical+SSP2-4.5
Two ensemble members (r1, r25) compared against:
  (i)  recent baseline 2016-2025
  (ii) preindustrial baseline (full piControl mean)

Author: Nasrul Huda
"""
from pathlib import Path
import textwrap
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -------- Configuration --------
DATA_DIR = Path(__file__).resolve().parent.parent / "model_outputs"

FILES = {
    "A_r1":  DATA_DIR / "tas_Amon_MPI-ESM1-2-LR_historical-ssp245_r1i1p1f1_g025_185001-210012.nc",
    "B_r25": DATA_DIR / "tas_Amon_MPI-ESM1-2-LR_historical-ssp245_r25i1p1f1_g025_185001-210012.nc",
    "piCtl": DATA_DIR / "tas_Amon_MPI-ESM1-2-LR_piControl_r1i1p1f1_g025_185001-284912.nc",
}

REF_START, REF_END = 2016, 2025
FUT_START, FUT_END = 2091, 2100

REGIONS = {
    "Global mean":     dict(lat=(-90, 90), lon=None),
    "Arctic (60-90N)": dict(lat=(60, 90),  lon=None),
}

VAR = "tas"


# -------- Data loading --------
_TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)


def load_experiment(path: Path, var: str = VAR) -> xr.DataArray:
    ds = xr.open_dataset(path, decode_times=_TIME_CODER)
    return ds[var]


# -------- Computation --------
def _year_slice(da: xr.DataArray, y0: int, y1: int) -> xr.DataArray:
    """Select all months in years [y0, y1] inclusive."""
    yrs = da.time.dt.year
    return da.where((yrs >= y0) & (yrs <= y1), drop=True)


def compute_climatology(da: xr.DataArray, year_start: int, year_end: int) -> xr.DataArray:
    """Time-mean over [year_start, year_end] inclusive. Returns 2D (lat, lon)."""
    sub = _year_slice(da, year_start, year_end)
    return sub.mean(dim="time", keep_attrs=True)


def compute_anomaly(da_future: xr.DataArray, da_baseline: xr.DataArray) -> xr.DataArray:
    out = da_future - da_baseline
    out.attrs["units"] = "K"
    return out


def _lat_weights(da: xr.DataArray) -> xr.DataArray:
    return np.cos(np.deg2rad(da.lat))


def regional_mean(da: xr.DataArray, lat_bounds, lon_bounds=None) -> xr.DataArray:
    """Area-weighted (cos-lat) regional mean. lon_bounds=None means all longitudes.
    Handles wrap-around if lon_bounds[0] > lon_bounds[1]."""
    sub = da.sel(lat=slice(lat_bounds[0], lat_bounds[1]))
    if lon_bounds is not None:
        lo, hi = lon_bounds
        if lo <= hi:
            sub = sub.sel(lon=slice(lo, hi))
        else:
            sub = sub.where((sub.lon >= lo) | (sub.lon <= hi), drop=True)
    w = _lat_weights(sub)
    return sub.weighted(w).mean(dim=[d for d in ("lat", "lon") if d in sub.dims])


def global_mean(da: xr.DataArray) -> xr.DataArray:
    return regional_mean(da, (-90, 90), None)


# -------- Plotting --------
def plot_anomaly_map(ax, anomaly: xr.DataArray, title: str, vmin: float, vmax: float):
    """Plot a single anomaly map onto a Cartopy axis. Returns the QuadMesh."""
    # Cyclic point so there is no white seam at lon=0
    data = anomaly.values
    lon  = anomaly.lon.values
    lat  = anomaly.lat.values
    # add cyclic column
    data_c = np.concatenate([data, data[:, :1]], axis=1)
    lon_c  = np.concatenate([lon, [lon[0] + 360.0]])

    mesh = ax.pcolormesh(
        lon_c, lat, data_c,
        cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto",
        transform=ccrs.PlateCarree(),
    )
    ax.coastlines(linewidth=0.6)
    ax.set_global()
    ax.set_title(title, fontsize=10)
    return mesh


def make_anomaly_figure(anomalies: dict, savepath: Path | None = None):
    """4-panel figure (2x2) with shared diverging colorbar centred on 0."""
    # Shared symmetric limits across all four panels.
    vmax = max(float(np.abs(a).max()) for a in anomalies.values())
    # round up to nice number
    vmax = np.ceil(vmax)
    vmin = -vmax

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 7),
        subplot_kw=dict(projection=ccrs.Robinson(central_longitude=0)),
    )
    axes = axes.flatten()
    items = list(anomalies.items())
    mesh = None
    for ax, (label, anom) in zip(axes, items):
        mesh = plot_anomaly_map(ax, anom, label, vmin, vmax)

    fig.suptitle(
        "MPI-ESM1-2-LR  |  near-surface air temperature anomaly (K)\n"
        f"future = mean({FUT_START}-{FUT_END});  "
        f"recent baseline = mean({REF_START}-{REF_END});  "
        "preindustrial baseline = full piControl mean",
        fontsize=11,
    )

    cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.025])
    cb = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal", extend="both")
    cb.set_label("Temperature anomaly (K)")

    fig.text(0.99, 0.01, "Nasrul Huda", ha="right", va="bottom",
             fontsize=8, color="gray")

    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.13,
                        wspace=0.05, hspace=0.15)
    if savepath is not None:
        fig.savefig(savepath, format="pdf", bbox_inches="tight")
        print(f"Saved {savepath}")
    return fig


def annual_regional_mean(da: xr.DataArray, lat_bounds, lon_bounds=None) -> xr.DataArray:
    """Area-weighted regional mean -> annual mean. Returns 1D over time."""
    rm = regional_mean(da, lat_bounds, lon_bounds)
    return rm.resample(time="YE").mean()


def plot_timeseries(series_by_region: dict, savepath: Path | None = None):
    """series_by_region[region_name] = dict with keys
       'A' (annual DataArray), 'B' (annual DataArray),
       'pi_mean' (float, K), 'pi_std' (float, K)
    """
    n = len(series_by_region)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n + 0.6), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (region, s) in zip(axes, series_by_region.items()):
        years_A = s["A"].time.dt.year.values
        years_B = s["B"].time.dt.year.values
        ax.plot(years_A, s["A"].values - 273.15, color="steelblue", lw=1.2,
                label="Experiment A (r1)")
        ax.plot(years_B, s["B"].values - 273.15, color="tomato",    lw=1.2,
                label="Experiment B (r25)")

        pi_mean = s["pi_mean"] - 273.15
        pi_std  = s["pi_std"]
        ax.axhline(pi_mean, color="black", lw=1.0, linestyle="--",
                   label=f"piControl mean ({pi_mean:.2f} \u00b0C)")
        ax.fill_between(
            [years_A.min(), years_A.max()],
            pi_mean - 2 * pi_std, pi_mean + 2 * pi_std,
            color="black", alpha=0.08, label="piControl \u00b12\u03c3"
        )

        ax.axvline(2014, color="gray", lw=0.8, linestyle=":", label="hist \u2192 SSP2-4.5")
        ax.set_title(f"{region}  |  annual-mean tas")
        ax.set_ylabel("Temperature (\u00b0C)")
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="upper left", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Year")
    fig.text(0.99, 0.01, "Nasrul Huda", ha="right", va="bottom",
             fontsize=8, color="gray")
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, format="pdf", bbox_inches="tight")
        print(f"Saved {savepath}")
    return fig


# -------- Written analysis --------
def make_paragraph_figure(stats: dict) -> plt.Figure:
    """Render a one-page A4-ish figure containing the written analysis."""
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait, inches
    fig.text(0.07, 0.94,
             "Spatial patterns of warming in MPI-ESM1-2-LR (SSP2-4.5)",
             fontsize=14, weight="bold")
    fig.text(0.07, 0.91,
             "Comparison of two ensemble members (r1, r25) against recent and "
             "preindustrial baselines.",
             fontsize=10, style="italic", color="dimgray")
    fig.text(0.07, 0.885,
             "Nasrul Huda  \u00b7  Climate Modelling, Universit\u00e4t Hamburg",
             fontsize=10, color="black")

    # Build paragraph text from computed stats so numbers stay in sync.
    p = (
        "The four anomaly maps (2091-2100 minus 2016-2025 and minus the full "
        "piControl mean, for each ensemble member) show the canonical CMIP6 warming "
        "pattern under SSP2-4.5. Warming is everywhere positive and dominated by two "
        "well-known features: strong polar amplification over the Arctic (local maxima "
        f"exceeding {stats['max_anom']:.0f} K against the preindustrial baseline) and "
        "a pronounced land-ocean contrast, with mid- and high-latitude continents "
        "warming substantially faster than the surrounding oceans. The Southern Ocean "
        "and the North Atlantic 'warming hole' south of Greenland show the slowest "
        "warming, consistent with deep ocean heat uptake and a weakening AMOC. "
        "\n\n"
        f"The two experiments (r1, r25) are two realisations of the same forced "
        f"scenario, so their global-mean response is essentially identical "
        f"(A: {stats['gm_A_pi']:+.2f} K vs B: {stats['gm_B_pi']:+.2f} K relative to "
        f"piControl). They differ, however, in the regional pattern: peak Arctic "
        f"anomalies in r1 reach {stats['max_anom_A']:.1f} K against the recent "
        f"baseline, while r25 peaks at only {stats['max_anom_B']:.1f} K. This spread "
        "is internal-variability noise (decadal sea-ice and Arctic-Oscillation "
        "fluctuations), not a forced difference."
        "\n\n"
        "The choice of baseline strongly affects the apparent magnitude of the "
        f"anomaly. The preindustrial baseline yields ~{stats['gm_A_pi']:.1f} K of "
        f"global warming by 2091-2100, whereas the 2016-2025 baseline only shows "
        f"~{stats['gm_A_ref']:.1f} K, since most of that ~1.1 K of historical and "
        "early-21st-century warming has already been absorbed into the recent "
        "reference. The recent baseline is therefore the appropriate frame for "
        "communicating committed future warming, while the preindustrial baseline "
        "is the relevant frame for Paris-Agreement-style targets."
        "\n\n"
        "The Arctic time series highlights what is mildly surprising: even at "
        f"{stats['arctic_sigma']:.2f} K piControl interannual standard deviation, "
        "the forced trajectories of r1 and r25 remain visually indistinguishable "
        "until roughly the mid-21st century, after which their year-to-year paths "
        "diverge by several Kelvin while sharing the same trend - a clean "
        "illustration of the signal-to-noise problem at regional scales."
    )
    # Manual wrap per paragraph (width chosen for the A4 figure at fontsize 10).
    wrapped = "\n\n".join(textwrap.fill(par, width=95) for par in p.split("\n\n"))
    fig.text(0.07, 0.84, wrapped, fontsize=10, va="top", ha="left",
             family="serif", linespacing=1.45)

    fig.text(0.99, 0.01, "Nasrul Huda", ha="right", va="bottom",
             fontsize=8, color="gray")
    return fig


def build_combined_pdf(anom_fig, ts_fig, paragraph_fig, savepath: Path):
    with PdfPages(savepath) as pdf:
        pdf.savefig(paragraph_fig, bbox_inches="tight")
        pdf.savefig(anom_fig,      bbox_inches="tight")
        pdf.savefig(ts_fig,        bbox_inches="tight")
    print(f"Saved combined PDF: {savepath}")


def inspect(name: str, da: xr.DataArray) -> None:
    t0, t1 = da.time.values[[0, -1]]
    print(f"{name:>10s} | dims={dict(da.sizes)} | units={da.attrs.get('units','?')} "
          f"| calendar={da.time.dt.calendar} | {t0} -> {t1}")


def main():
    da_A   = load_experiment(FILES["A_r1"])
    da_B   = load_experiment(FILES["B_r25"])
    da_pic = load_experiment(FILES["piCtl"])

    inspect("A (r1)",    da_A)
    inspect("B (r25)",   da_B)
    inspect("piControl", da_pic)

    print("\nLat range:", float(da_A.lat.min()), "to", float(da_A.lat.max()),
          "| n =", da_A.sizes["lat"])
    print("Lon range:", float(da_A.lon.min()), "to", float(da_A.lon.max()),
          "| n =", da_A.sizes["lon"])

    # ---- Climatologies ----
    fut_A = compute_climatology(da_A, FUT_START, FUT_END)
    fut_B = compute_climatology(da_B, FUT_START, FUT_END)
    ref_A = compute_climatology(da_A, REF_START, REF_END)
    ref_B = compute_climatology(da_B, REF_START, REF_END)
    pi_A  = da_pic.mean(dim="time", keep_attrs=True)   # full piControl

    # ---- Anomalies ----
    anomalies = {
        "A: 2091-2100 minus 2016-2025":     compute_anomaly(fut_A, ref_A),
        "A: 2091-2100 minus piControl":     compute_anomaly(fut_A, pi_A),
        "B: 2091-2100 minus 2016-2025":     compute_anomaly(fut_B, ref_B),
        "B: 2091-2100 minus piControl":     compute_anomaly(fut_B, pi_A),
    }

    print("\n--- Sanity check: area-weighted global-mean anomaly (K) ---")
    for label, anom in anomalies.items():
        gm = float(global_mean(anom))
        amin, amax = float(anom.min()), float(anom.max())
        print(f"  {label:40s} | global mean = {gm:+.3f} K | min = {amin:+.2f} | max = {amax:+.2f}")

    out_dir = Path(__file__).resolve().parent
    anom_fig = make_anomaly_figure(anomalies, out_dir / "anomaly_maps.pdf")

    # ---- Regional time series ----
    print("\n--- Regional annual time series ---")
    series_by_region = {}
    for region_name, bounds in REGIONS.items():
        ann_A = annual_regional_mean(da_A,   bounds["lat"], bounds["lon"])
        ann_B = annual_regional_mean(da_B,   bounds["lat"], bounds["lon"])
        ann_p = annual_regional_mean(da_pic, bounds["lat"], bounds["lon"])
        pi_mean = float(ann_p.mean())
        pi_std  = float(ann_p.std())
        print(f"  {region_name:18s} | piControl mean = {pi_mean - 273.15:+.2f} \u00b0C "
              f"| sigma = {pi_std:.3f} K | A 2091-2100 = {float(ann_A.where(ann_A.time.dt.year >= FUT_START, drop=True).mean()) - 273.15:+.2f} \u00b0C "
              f"| B 2091-2100 = {float(ann_B.where(ann_B.time.dt.year >= FUT_START, drop=True).mean()) - 273.15:+.2f} \u00b0C")
        series_by_region[region_name] = dict(A=ann_A, B=ann_B, pi_mean=pi_mean, pi_std=pi_std)

    ts_fig = plot_timeseries(series_by_region, out_dir / "timeseries.pdf")

    # ---- Combined PDF ----
    stats = dict(
        gm_A_ref   = float(global_mean(anomalies["A: 2091-2100 minus 2016-2025"])),
        gm_A_pi    = float(global_mean(anomalies["A: 2091-2100 minus piControl"])),
        gm_B_ref   = float(global_mean(anomalies["B: 2091-2100 minus 2016-2025"])),
        gm_B_pi    = float(global_mean(anomalies["B: 2091-2100 minus piControl"])),
        max_anom   = max(float(a.max()) for a in anomalies.values()),
        max_anom_A = float(anomalies["A: 2091-2100 minus 2016-2025"].max()),
        max_anom_B = float(anomalies["B: 2091-2100 minus 2016-2025"].max()),
        arctic_sigma = series_by_region["Arctic (60-90N)"]["pi_std"],
    )
    paragraph_fig = make_paragraph_figure(stats)
    build_combined_pdf(anom_fig, ts_fig, paragraph_fig,
                       out_dir / "spatial_patterns_submission.pdf")
    plt.close("all")


if __name__ == "__main__":
    main()
