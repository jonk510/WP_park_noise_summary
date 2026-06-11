"""
Wind Turbine Noise Contour Analyser
=====================================
Estimates A-weighted noise contours for a given wind turbine layout
using the ISO 9613-2 simplified propagation model.

Inputs
------
  WTG layout        CSV  — columns X, Y (projected metres)
  Hub height        metres above local ground
  Sound power data  octave-band or 1/3-octave-band Lw (dB re 1 pW),
                    entered manually or loaded from a CSV file
  Terrain elevation local XYZ/CSV file  OR  SRTM auto-download (OpenTopoData)
  EPSG code         projected coordinate system of the WTG file

Propagation model  (ISO 9613-2 simplified method)
--------------------------------------------------
  Per octave band f:
    Lp_f  = Lw_f − Adiv − Aatm_f − Agr_f

    Adiv  = 20·log10(d_slant) + 11       [dB]  geometric divergence
    Aatm  = α_f · d_slant / 1000         [dB]  atmospheric absorption (ISO 9613-1)
    Agr   = As + Am + Ar                 [dB]  ground effect (ISO 9613-2:1996 §7.3 Table 3)
              As, Ar  frequency-specific height functions a′–d′ (125–1000 Hz),
                      constant −1.5(1−G) at 2000–8000 Hz, −1.5 at 63 Hz
              Am = −3q(1−G),  q = max(1 − 30(hs+hr)/dp, 0)
              For tall turbines (hs >> 10 m) the height terms vanish and
              q = 0 for dp < 30(hs+hr), giving Agr ≈ −3(1−G) dB

  A-weighted SPL per band:
    Lp_A_f = Lp_f + ΔA_f

  Total at receiver (energy sum over bands and over all turbines):
    Lp_A = 10·log10( Σ_turbines Σ_bands 10^(Lp_A_f/10) )

  Terrain shielding  ISO 9613-2 §8 optional — dominant ridge barrier:
                     A_bar = 10·log₁₀(3 + 20·Nf), Nf = 2δf/c, capped 20 dB
                     Disabled by default (use_shielding=False).

Outputs
-------
  wind_noise_results.png   — 4-panel figure (satellite map, terrain map,
                             distance-decay curve, octave-band spectrum)
  wind_noise_levels.csv    — full noise grid  (X, Y, Lp_A_dBA columns)

Dependencies
------------
  pip install numpy pandas scipy matplotlib requests pyproj
  pip install contextily          # satellite / Bing tile basemaps
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as mpe
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath

from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ── optional dependencies ────────────────────────────────────────────────────
try:
    import requests
    from pyproj import Transformer as _ProjTransformer
    _HAS_PYPROJ = True
except Exception as _pyproj_err:
    _HAS_PYPROJ = False
    print(f"  Warning: requests/pyproj import failed ({type(_pyproj_err).__name__}: {_pyproj_err})")

try:
    import contextily as cx
    _HAS_CTX = True
except ImportError:
    _HAS_CTX = False


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these defaults before running
# ══════════════════════════════════════════════════════════════════════════════

# Turbine / site
DEFAULT_HUB_HEIGHT_M    = 150     # hub height above local ground (m)
DEFAULT_RECEIVER_HT_M   = 4.0    # receptor height above local ground (m) — SA Guidelines 2021
DEFAULT_GROUND_FACTOR   = 0.5    # G: 0 = hard (paved/water) … 1 = soft (grass/crops) — SA Guidelines 2021: G=0.5

# Coordinate system
DEFAULT_EPSG            = 7850  # EPSG code for projected WTG coordinates

# Noise grid
DEFAULT_GRID_RESOLUTION = 200    # grid points per side (N × N)
DEFAULT_GRID_BUFFER_M   = 3000   # buffer around turbine bounding box (m)
DEFAULT_CONTOUR_LEVELS  = [25, 30, 35, 40, 45, 50, 55, 60]  # dB(A) contour levels to draw

# SRTM terrain download (when no local XYZ file is provided)
SRTM_BUFFER_M           = 5000   # download area buffer beyond layout (m)
SRTM_GRID_N             = 40     # elevation grid points per axis


# ══════════════════════════════════════════════════════════════════════════════
# Acoustic constants
# ══════════════════════════════════════════════════════════════════════════════

OCTAVE_BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000]  # Hz

# A-weighting corrections (dB) per octave-band centre frequency
A_WEIGHTING = {
    63: -26.2, 125: -16.1, 250: -8.6,  500: -3.2,
    1000:  0.0, 2000:  1.2, 4000:  1.0, 8000: -1.1,
}

# Atmospheric absorption α (dB/km) at 10 °C, 80 % RH — ISO 9613-1 (SA Guidelines 2021)
ALPHA_ATM = {
    63: 0.1, 125: 0.4,  250: 1.0,   500: 2.0,
    1000: 3.6, 2000: 8.8, 4000: 29.0, 8000: 104.6,
}

# 1/3-octave → parent octave band
# Note: int(12.5)=12 and int(31.5)=31, so float inputs from spreadsheets resolve correctly
THIRD_OCT_TO_OCT = {
    # Sub-44.5 Hz bands are below ISO 9613-2's range; rolled into the 63 Hz octave.
    # int() handles both 31.5 Hz (→31) and 32 Hz nominal variants.
    10: 63,  12: 63,  16: 63,  20: 63,  25: 63,  31: 63,  32: 63,  40: 63,
     50: 63,   63: 63,   80: 63,
    100: 125,  125: 125, 160: 125,
    200: 250,  250: 250, 315: 250,
    400: 500,  500: 500, 630: 500,
    800: 1000, 1000: 1000, 1250: 1000,
    1600: 2000, 2000: 2000, 2500: 2000,
    3150: 4000, 4000: 4000, 5000: 4000,
    6300: 8000, 8000: 8000, 10000: 8000,
}

# A-weighting corrections (dB) at 1/3-octave centre frequencies (IEC 61672-1)
# Integer keys match int(freq) convention used in THIRD_OCT_TO_OCT (12→12.5 Hz, 31→31.5 Hz)
A_WEIGHTING_THIRD_OCT = {
    10: -70.4, 12: -63.4, 16: -56.7, 20: -50.5, 25: -44.7, 31: -39.4, 32: -39.1, 40: -34.6,
    50: -30.2,  63: -26.2,  80: -22.5, 100: -19.1, 125: -16.1, 160: -13.4,
    200: -10.9, 250:  -8.6, 315:  -6.6, 400:  -4.8, 500:  -3.2, 630:  -1.9,
    800:  -0.8, 1000:  0.0, 1250:  0.6, 1600:   1.0, 2000:  1.2, 2500:  1.3,
    3150:  1.2, 4000:  1.0, 5000:  0.5, 6300:  -0.1, 8000: -1.1, 10000: -2.5,
}

# Default 1/3-octave reduction vs. octave band for manual entry
_THIRD_OCT_DEFAULT_OFFSET = -4.8   # ≈ 10·log10(1/3)

# GE 164-6.0 octave-band SWL defaults (derived from GE 164_6.0_sound_power_levels.xlsx)
_DEFAULT_LW = {
    63: 88.1, 125: 93.6, 250: 98.1,  500: 100.7,
    1000: 102.3, 2000: 100.1, 4000: 92.6, 8000: 76.8,
}

# Noise contour fill colours (green → yellow → orange → red → purple)
_NOISE_COLOURS = [
    "#27ae60", "#a9d65d", "#f9ca24", "#f0932b", "#eb4d4b", "#8e44ad", "#2d3436",
]


# ══════════════════════════════════════════════════════════════════════════════
# Band conversion utilities
# ══════════════════════════════════════════════════════════════════════════════

def third_oct_to_octave(third_oct: dict, a_weighted: bool = False) -> dict:
    """Energy-sum 1/3-octave levels into octave bands.

    a_weighted=True: inputs are Lwa (A-weighted) per band — each value is
    un-A-weighted before summing so the returned dict contains unweighted Lw,
    ready for the ISO 9613-2 propagation model which applies A-weighting itself.
    """
    energy = {f: 0.0 for f in OCTAVE_BANDS}
    for freq, val in third_oct.items():
        parent = THIRD_OCT_TO_OCT.get(int(freq))
        if parent is not None:
            lw = float(val)
            if a_weighted:
                lw -= A_WEIGHTING_THIRD_OCT.get(int(freq), 0.0)
            energy[parent] += 10.0 ** (lw / 10.0)
    return {f: 10.0 * np.log10(max(e, 1e-30)) for f, e in energy.items()}


def overall_lwa(Lw_bands: dict) -> float:
    """Overall A-weighted sound power level from octave-band dict."""
    lwa_energy = sum(
        10.0 ** ((Lw_bands[f] + A_WEIGHTING[f]) / 10.0)
        for f in OCTAVE_BANDS if f in Lw_bands)
    return 10.0 * np.log10(max(lwa_energy, 1e-30))


# ══════════════════════════════════════════════════════════════════════════════
# ISO 9613-2 ground effect — vectorised
# ══════════════════════════════════════════════════════════════════════════════

def _ground_effect_all_bands(dp: np.ndarray, hs: np.ndarray,
                               hr: float, G: float) -> dict:
    """
    ISO 9613-2:1996 §7.3 ground effect — Table 3 formulas.

    Uses the standard frequency-specific height functions a′–d′ and the
    distance-dependent q factor for Am.  For tall wind turbines (hs >> 10 m)
    the exponential height terms vanish and q = 0 while dp < 30(hs+hr),
    so As ≈ Ar ≈ −1.5(1−G) and Am = 0, giving Agr ≈ −3(1−G) dB.

    Parameters
    ----------
    dp  : (N,) horizontal source-to-receiver distance (m)
    hs  : (N,) source height above local ground (m)
    hr  : receiver height above local ground (m)
    G   : ground factor  0 = hard (paved/water)  →  1 = soft (grass/crops)

    Returns
    -------
    dict {freq_hz: Agr_array (N,)}  — negative = ground enhancement (higher noise)
    """
    dp = np.maximum(dp, 1.0)

    # Middle-region q factor — zero while source/receiver regions overlap
    q = np.maximum(1.0 - 30.0 * (hs + hr) / dp, 0.0)

    # Table 3 height functions (h in metres, dp in metres).
    # Exponentials vanish for h >> 10 m so tall-turbine As/Ar → −1.5(1−G).
    def a_prime(h):
        return (1.5
                + 3.0 * np.exp(-0.12 * (h - 5.0)) * (1.0 - np.exp(-dp / 50.0))
                + 5.7 * np.exp(-0.09 * h ** 2)    * (1.0 - np.exp(-2.8e-6 * dp ** 2)))

    def b_prime(h):
        return 1.5 + 8.6  * np.exp(-0.09 * h ** 2) * (1.0 - np.exp(-dp / 50.0))

    def c_prime(h):
        return 1.5 + 14.0 * np.exp(-0.46 * h ** 2) * (1.0 - np.exp(-dp / 50.0))

    def d_prime(h):
        return 1.5 + 5.0  * np.exp(-0.5  * h ** 2) * (1.0 - np.exp(-dp / 50.0))

    ones = np.ones_like(dp)
    result = {}
    for f in OCTAVE_BANDS:
        if f == 63:
            # No G-dependence at 63 Hz per Table 3
            As = -1.5 * ones
            Ar = -1.5 * ones
            Am = -3.0 * q
        elif f == 125:
            As = -1.5 + G * a_prime(hs)
            Ar = (-1.5 + G * a_prime(hr)) * ones
            Am = -3.0 * q * (1.0 - G)
        elif f == 250:
            As = -1.5 + G * b_prime(hs)
            Ar = (-1.5 + G * b_prime(hr)) * ones
            Am = -3.0 * q * (1.0 - G)
        elif f == 500:
            As = -1.5 + G * c_prime(hs)
            Ar = (-1.5 + G * c_prime(hr)) * ones
            Am = -3.0 * q * (1.0 - G)
        elif f == 1000:
            As = -1.5 + G * d_prime(hs)
            Ar = (-1.5 + G * d_prime(hr)) * ones
            Am = -3.0 * q * (1.0 - G)
        else:  # 2000, 4000, 8000 Hz
            As = -1.5 * (1.0 - G) * ones
            Ar = -1.5 * (1.0 - G) * ones
            Am = -3.0 * q * (1.0 - G)
        result[f] = As + Am + Ar
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ISO 9613-2 §8 terrain shielding — vectorised
# ══════════════════════════════════════════════════════════════════════════════

def _shielding_attenuation(tx: float, ty: float, zs: float,
                            rx_flat: np.ndarray, ry_flat: np.ndarray,
                            zr: np.ndarray, d_slant: np.ndarray,
                            elev_grid: np.ndarray,
                            x0: float, y0: float,
                            dx_grid: float, dy_grid: float,
                            n_samples: int = 30) -> dict:
    """
    ISO 9613-2 §8 barrier attenuation for terrain shielding, all receivers at once.

    Samples terrain at n_samples points along each source→receiver path,
    finds the highest terrain point above the line of sight (the dominant
    barrier), computes the Fresnel path-length difference δ, and returns
    A_bar per octave band.  Returns zero where terrain does not break LOS.

    Parameters
    ----------
    tx, ty    : source (turbine) XY position
    zs        : source height (turbine elevation + hub height, m AMSL)
    rx_flat   : (N,) receiver X positions
    ry_flat   : (N,) receiver Y positions
    zr        : (N,) receiver heights (terrain elev + hr, m AMSL)
    d_slant   : (N,) 3-D source→receiver distance (m)
    elev_grid : (M, M) terrain elevation on the noise mesh
    x0, y0    : origin of elev_grid (xx[0,0], yy[0,0])
    dx_grid   : column spacing of elev_grid (m)
    dy_grid   : row spacing of elev_grid (m)

    Returns
    -------
    dict {freq_hz: A_bar (N,)} — barrier attenuation per octave band (dB)
    """
    N_pts = len(rx_flat)

    # Convert XY positions to fractional row/col indices in elev_grid
    row_r = (ry_flat - y0) / dy_grid  # (N,)
    col_r = (rx_flat - x0) / dx_grid  # (N,)
    row_s = (ty - y0) / dy_grid       # scalar
    col_s = (tx - x0) / dx_grid       # scalar

    # Intermediate sample fractions (exclude endpoints — those are source/receiver)
    t_vals = np.linspace(0.05, 0.95, n_samples)  # (K,)

    # Fractional indices for all sample points × all receivers  →  (K, N)
    rows = row_s + t_vals[:, None] * (row_r[None, :] - row_s)
    cols = col_s + t_vals[:, None] * (col_r[None, :] - col_s)

    # Bilinear interpolation on the elevation grid (clamped at edges)
    terrain = map_coordinates(
        elev_grid, [rows.ravel(), cols.ravel()],
        order=1, mode="nearest",
    ).reshape(n_samples, N_pts)  # (K, N) — terrain elev at each sample

    # Straight line-of-sight elevation at each sample fraction
    los = zs + t_vals[:, None] * (zr[None, :] - zs)  # (K, N)

    # Terrain height above line of sight (positive = terrain breaks LOS)
    excess = terrain - los  # (K, N)
    max_excess = excess.max(axis=0)         # (N,)
    k_bar      = excess.argmax(axis=0)      # (N,) — dominant barrier index

    barrier_mask = max_excess > 0.0         # True where terrain blocks LOS

    # Barrier position in plan and elevation
    t_bar = t_vals[k_bar]                            # (N,) — fraction along path
    z_bar = terrain[k_bar, np.arange(N_pts)]         # (N,) — barrier top elevation

    xb = tx + t_bar * (rx_flat - tx)
    yb = ty + t_bar * (ry_flat - ty)

    # 3-D path lengths source→barrier and barrier→receiver
    d_sb = np.sqrt((xb - tx) ** 2 + (yb - ty) ** 2 + (z_bar - zs) ** 2)
    d_br = np.sqrt((rx_flat - xb) ** 2 + (ry_flat - yb) ** 2 + (zr - z_bar) ** 2)

    # Path-length difference δ (non-negative by the triangle inequality)
    delta = np.maximum(d_sb + d_br - d_slant, 0.0)

    # ISO 9613-2 §8.2: A_bar = 10·log10(3 + 20·Nf), max 20 dB per single edge
    result = {}
    for f in OCTAVE_BANDS:
        Nf = 2.0 * delta * float(f) / 340.0
        A_bar = np.where(
            barrier_mask,
            np.minimum(10.0 * np.log10(np.maximum(3.0 + 20.0 * Nf, 1e-10)), 20.0),
            0.0,
        )
        result[f] = A_bar
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Noise grid computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_noise_grid(wtg_xy: np.ndarray,
                        wtg_elevs: np.ndarray,
                        Lw_bands: dict,
                        hub_height: float,
                        xx: np.ndarray,
                        yy: np.ndarray,
                        elev_grid: np.ndarray,
                        hr: float = 1.5,
                        G: float = 0.5,
                        use_shielding: bool = False) -> np.ndarray:
    """
    A-weighted noise grid (energy sum from all turbines) — ISO 9613-2.

    Parameters
    ----------
    wtg_xy        : (N_wtg, 2)  turbine projected XY coordinates
    wtg_elevs     : (N_wtg,)    terrain elevation at each turbine
    Lw_bands      : {freq_hz: Lw_dB}  octave-band SWL (same for all turbines)
    hub_height    : hub height above local ground (m)
    xx, yy        : (M, M) output mesh grids (projected CRS)
    elev_grid     : (M, M) terrain elevation at grid points (m)
    hr            : receiver height above ground (m)
    G             : ground factor  0 = hard  …  1 = soft
    use_shielding : apply ISO 9613-2 §8 terrain barrier attenuation

    Returns
    -------
    noise_grid : (M, M) A-weighted SPL in dB(A)
    """
    shape        = xx.shape
    grid_pts     = np.column_stack([xx.ravel(), yy.ravel()])
    g_elev_flat  = elev_grid.ravel()
    energy_total = np.zeros(len(grid_pts))

    # Precompute grid metadata needed for terrain profile sampling
    if use_shielding:
        _x0  = float(xx[0, 0])
        _y0  = float(yy[0, 0])
        _dx  = float(xx[0, 1] - xx[0, 0]) if xx.shape[1] > 1 else 1.0
        _dy  = float(yy[1, 0] - yy[0, 0]) if yy.shape[0] > 1 else 1.0

    for i, (pos, w_elev) in enumerate(zip(wtg_xy, wtg_elevs)):
        diffs = grid_pts - pos
        dp    = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        dp    = np.maximum(dp, 1.0)

        # Effective hub height above receiver-level terrain
        delta_z = float(w_elev) - g_elev_flat      # positive when turbine is uphill
        hs_eff  = np.maximum(hub_height + delta_z, 1.0)

        # Slant distance source → receiver
        d_slant = np.sqrt(dp ** 2 + np.maximum(hs_eff - hr, 0.0) ** 2)
        d_slant = np.maximum(d_slant, 1.0)

        Agr = _ground_effect_all_bands(dp, hs_eff, hr, G)

        # Terrain shielding — ISO 9613-2 §8 barrier attenuation
        if use_shielding:
            zs_src  = float(w_elev) + hub_height
            zr_recv = g_elev_flat + hr
            A_bar   = _shielding_attenuation(
                float(pos[0]), float(pos[1]), zs_src,
                grid_pts[:, 0], grid_pts[:, 1], zr_recv, d_slant,
                elev_grid, _x0, _y0, _dx, _dy)

        band_energy = np.zeros(len(grid_pts))
        for f in OCTAVE_BANDS:
            Lw = Lw_bands.get(f)
            if Lw is None:
                continue
            Adiv   = 20.0 * np.log10(d_slant) + 11.0
            Aatm   = ALPHA_ATM[f] * d_slant / 1000.0
            Atotal = Adiv + Aatm + Agr[f]
            if use_shielding:
                Atotal = Atotal + A_bar[f]
            Lp_A   = Lw - Atotal + A_WEIGHTING[f]
            band_energy += 10.0 ** (Lp_A / 10.0)

        energy_total += band_energy
        print(f"    Turbine {i + 1}/{len(wtg_xy)} computed…")

    noise_flat = 10.0 * np.log10(np.maximum(energy_total, 1e-30))
    return noise_flat.reshape(shape)


# ══════════════════════════════════════════════════════════════════════════════
# Terrain elevation — download or interpolate
# ══════════════════════════════════════════════════════════════════════════════

def fetch_srtm_elevation(wtg_xy: np.ndarray, epsg_code: int,
                          buffer_m: float = 5000.0,
                          grid_n: int = 40) -> pd.DataFrame:
    """
    Download SRTM 30 m elevation over the WTG bounding box via OpenTopoData
    (free public API, no key required).  Returns DataFrame with X, Y, Z columns.
    """
    if not _HAS_PYPROJ:
        raise ImportError(
            "Install 'requests' and 'pyproj':\n"
            "  pip install requests pyproj")

    xmin = wtg_xy[:, 0].min() - buffer_m
    xmax = wtg_xy[:, 0].max() + buffer_m
    ymin = wtg_xy[:, 1].min() - buffer_m
    ymax = wtg_xy[:, 1].max() + buffer_m

    xi = np.linspace(xmin, xmax, grid_n)
    yi = np.linspace(ymin, ymax, grid_n)
    gxx, gyy = np.meshgrid(xi, yi)
    grid_xy = np.column_stack([gxx.ravel(), gyy.ravel()])

    trans = _ProjTransformer.from_crs(
        f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    lons, lats = trans.transform(grid_xy[:, 0], grid_xy[:, 1])

    if not (np.all(np.isfinite(lons)) and np.all(np.isfinite(lats))):
        raise ValueError(
            f"Coordinate transform failed (EPSG:{epsg_code} → WGS84). "
            "Verify the EPSG code matches your WTG file.")

    elevations = []
    n_pts = len(lats)
    for start in range(0, n_pts, 100):
        blats = lats[start:start + 100]
        blons = lons[start:start + 100]
        locs  = "|".join(f"{la:.6f},{lo:.6f}" for la, lo in zip(blats, blons))
        resp  = requests.get(
            f"https://api.opentopodata.org/v1/srtm30m?locations={locs}", timeout=30)
        resp.raise_for_status()
        for r in resp.json()["results"]:
            e = r.get("elevation")
            elevations.append(float(e) if e is not None else 0.0)
        done = min(start + 100, n_pts)
        print(f"    DEM: {done}/{n_pts} elevation points downloaded…")
        if done < n_pts:
            time.sleep(1.1)

    return pd.DataFrame({"X": grid_xy[:, 0], "Y": grid_xy[:, 1],
                          "Z": np.array(elevations)})


def _build_elev_interp(xyz: pd.DataFrame):
    """Return a callable that interpolates elevation at arbitrary (N,2) XY points."""
    pts  = xyz[["X", "Y"]].values.astype(float)
    vals = xyz["Z"].values.astype(float)
    tree = cKDTree(pts)

    def _interp(xy: np.ndarray) -> np.ndarray:
        xy   = np.atleast_2d(xy).astype(float)
        elev = griddata(pts, vals, xy, method="linear")
        mask = np.isnan(elev)
        if mask.any():
            _, idx = tree.query(xy[mask])
            elev[mask] = vals[idx]
        return elev

    return _interp


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _terrain_overlay(ax, xx, yy, elev_grid, n_levels=15, alpha=0.35):
    """Filled terrain + contour lines overlay."""
    valid = ~np.isnan(elev_grid)
    if not valid.any():
        return
    lo = np.nanpercentile(elev_grid, 2)
    hi = np.nanpercentile(elev_grid, 98)
    if hi <= lo:
        return
    levels = np.linspace(lo, hi, n_levels)
    ax.contourf(xx, yy, elev_grid, levels=levels, cmap="terrain", alpha=alpha)
    ax.contour(xx, yy, elev_grid, levels=levels,
               colors="grey", linewidths=0.3, alpha=0.50)


def _format_map_axis(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
    ax.set_xlabel("Easting (m)", fontsize=22)
    ax.set_ylabel("Northing (m)", fontsize=22)
    ax.tick_params(labelsize=20)


def _noise_cmap_norm(levels):
    n      = len(levels) - 1
    colors = (_NOISE_COLOURS * ((n // len(_NOISE_COLOURS)) + 1))[:n]
    cmap   = mcolors.ListedColormap(colors)
    norm   = mcolors.BoundaryNorm(levels, len(colors))
    return cmap, norm


def _add_noise_contours(ax, xx, yy, noise_grid, levels, alpha_fill=0.55):
    """Draw filled + labelled noise contours.  Returns contourf handle."""
    cmap, norm = _noise_cmap_norm(levels)
    cf = ax.contourf(xx, yy, noise_grid, levels=levels,
                     cmap=cmap, norm=norm, alpha=alpha_fill, extend="both")
    cl = ax.contour(xx, yy, noise_grid, levels=levels,
                    colors="black", linewidths=0.8, alpha=0.9)
    ax.clabel(cl, fmt="%g dB(A)", fontsize=18, inline=True)
    return cf


def _make_wtg_marker():
    """3-bladed rotor plan-view marker — tapered blades + hub circle."""
    def _blade(angle_deg):
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        pts = np.array([
            [ 0.00,  0.00],
            [-0.10,  0.18],
            [-0.05,  1.00],
            [ 0.05,  1.00],
            [ 0.10,  0.18],
            [ 0.00,  0.00],
        ])
        rot = np.array([[c, -s], [s, c]])
        return (pts @ rot.T).tolist()

    verts, codes = [], []
    for angle in [90, 210, 330]:
        blade = _blade(angle)
        verts += blade
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(blade) - 2) + [MplPath.CLOSEPOLY]

    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    hub = np.column_stack([0.16 * np.cos(t), 0.16 * np.sin(t)])
    verts += hub.tolist() + [hub[0].tolist()]
    codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(hub) - 1) + [MplPath.CLOSEPOLY]

    return MplPath(verts, codes)

_WTG_MARKER = _make_wtg_marker()


def _scatter_turbines(ax, wtg_xy, label=True):
    ax.scatter(wtg_xy[:, 0], wtg_xy[:, 1],
               marker=_WTG_MARKER, s=400, c="white", edgecolors="black",
               linewidths=1.2, zorder=10)
    if label:
        for i, pos in enumerate(wtg_xy):
            ax.annotate(
                f"T{i + 1}", pos, xytext=(5, 4),
                textcoords="offset points", fontsize=18,
                color="white", fontweight="bold",
                path_effects=[mpe.withStroke(linewidth=2, foreground="black")])


def _scatter_receptors(ax, receptor_xy, receptor_levels, receptor_names=None):
    crit_35 = 35.0
    crit_40 = 40.0
    for i, (pos, lvl) in enumerate(zip(receptor_xy, receptor_levels)):
        colour = "#e74c3c" if lvl > crit_40 else ("#f39c12" if lvl > crit_35 else "#2ecc71")
        ax.scatter(pos[0], pos[1], marker="D", s=80, c=colour,
                   edgecolors="black", linewidths=1.0, zorder=11)
        name = receptor_names[i] if receptor_names else f"R{i + 1}"
        label = f"{name}\n{lvl:.1f} dB(A)"
        ax.annotate(
            label, pos, xytext=(6, 4), textcoords="offset points",
            fontsize=18, fontweight="bold", color="white",
            path_effects=[mpe.withStroke(linewidth=2, foreground="black")])


def _add_satellite(ax, epsg_code, bing_key=None) -> bool:
    """Try to add a satellite basemap.  Returns True if successful."""
    if not _HAS_CTX:
        return False
    try:
        if bing_key:
            source = cx.providers.Bing.Aerial(apikey=bing_key)
        else:
            source = cx.providers.Esri.WorldImagery
        cx.add_basemap(ax, crs=f"EPSG:{epsg_code}",
                       source=source, attribution=False, zoom="auto")
        return True
    except Exception as exc:
        print(f"  Warning: satellite basemap unavailable — {exc}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Main plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(wtg_xy: np.ndarray,
                 noise_grid: np.ndarray,
                 xx: np.ndarray, yy: np.ndarray,
                 elev_grid: np.ndarray,
                 Lw_bands: dict,
                 hub_height: float,
                 contour_levels: list,
                 epsg_code: int,
                 use_satellite: bool = True,
                 bing_key: str = None,
                 alpha_fill: float = 0.55,
                 save_path: str = None,
                 receptor_xy: np.ndarray = None,
                 receptor_levels: np.ndarray = None,
                 receptor_names: list = None):
    """
    Four-panel figure:
      Top-left  : satellite (or terrain) + noise contours  — main planning map
      Top-right : smooth noise-level heatmap               — technical view
      Bot-left  : noise level vs. distance from centroid
      Bot-right : input octave-band Lw spectrum + A-weighted levels
    """
    fig = plt.figure(figsize=(44, 36))
    fig.suptitle(
        f"Wind Turbine Noise Contour Analysis  ·  "
        f"Hub height {hub_height:.0f} m  ·  ISO 9613-2 simplified model",
        fontsize=30, fontweight="bold", y=0.99)

    gs = GridSpec(2, 3, figure=fig,
                  height_ratios=[3, 1], width_ratios=[1, 1, 1],
                  hspace=0.30, wspace=0.28,
                  left=0.04, right=0.97, top=0.96, bottom=0.05)

    ax_sat   = fig.add_subplot(gs[0, :])   # full-width top
    ax_ter   = fig.add_subplot(gs[1, 0])
    ax_decay = fig.add_subplot(gs[1, 1])
    ax_spec  = fig.add_subplot(gs[1, 2])

    cmap, norm = _noise_cmap_norm(contour_levels)
    xmin, xmax = float(xx.min()), float(xx.max())
    ymin, ymax = float(yy.min()), float(yy.max())

    # ── Panel 1 : satellite + noise contours ─────────────────────────────────
    ax_sat.set_xlim(xmin, xmax)
    ax_sat.set_ylim(ymin, ymax)

    sat_ok = False
    if use_satellite:
        sat_ok = _add_satellite(ax_sat, epsg_code, bing_key)
        ax_sat.set_xlim(xmin, xmax)
        ax_sat.set_ylim(ymin, ymax)

    if not sat_ok:
        _terrain_overlay(ax_sat, xx, yy, elev_grid, alpha=0.50)
    else:
        _terrain_overlay(ax_sat, xx, yy, elev_grid, alpha=0.20)

    _add_noise_contours(ax_sat, xx, yy, noise_grid, contour_levels,
                        alpha_fill=alpha_fill)
    _scatter_turbines(ax_sat, wtg_xy)

    legend_handles = [Line2D([0], [0], marker=_WTG_MARKER, color="w",
                             markerfacecolor="white", markeredgecolor="black",
                             markersize=14, label="Wind turbine")]
    if receptor_xy is not None and receptor_levels is not None:
        _scatter_receptors(ax_sat, receptor_xy, receptor_levels, receptor_names)
        legend_handles.append(
            Line2D([0], [0], marker="D", color="w",
                   markerfacecolor="yellow", markeredgecolor="black",
                   markersize=8, label="Sensitive receptor"))

    ax_sat.set_title(
        "Noise Contour Map" + (" — Satellite" if sat_ok else " — Terrain"),
        fontsize=26, fontweight="bold")
    _format_map_axis(ax_sat)
    ax_sat.legend(handles=legend_handles, loc="upper left", fontsize=20, framealpha=0.85)

    # Colourbar for Panel 1
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb1 = fig.colorbar(sm, ax=ax_sat, shrink=0.80, pad=0.02, aspect=28)
    cb1.set_label("A-weighted SPL  dB(A)", fontsize=20)
    cb1.ax.tick_params(labelsize=18)
    cb1.set_ticks(contour_levels)

    # ── Panel 2 : smooth noise heatmap ───────────────────────────────────────
    ax_ter.pcolormesh(xx, yy, noise_grid, cmap=cmap, norm=norm,
                      shading="auto", alpha=0.90)
    _terrain_overlay(ax_ter, xx, yy, elev_grid, alpha=0.25)
    _scatter_turbines(ax_ter, wtg_xy)
    if receptor_xy is not None and receptor_levels is not None:
        _scatter_receptors(ax_ter, receptor_xy, receptor_levels, receptor_names)
    ax_ter.set_title("Noise Level Heatmap — Terrain", fontsize=14, fontweight="bold")
    _format_map_axis(ax_ter)
    cb2 = fig.colorbar(sm, ax=ax_ter, shrink=0.80, pad=0.02, aspect=28)
    cb2.set_label("A-weighted SPL  dB(A)", fontsize=11)
    cb2.ax.tick_params(labelsize=10)
    cb2.set_ticks(contour_levels)

    # ── Panel 3 : distance-decay ──────────────────────────────────────────────
    centroid   = wtg_xy.mean(axis=0)
    noise_flat = noise_grid.ravel()
    grid_pts   = np.column_stack([xx.ravel(), yy.ravel()])
    r_from_cen = np.sqrt(((grid_pts - centroid) ** 2).sum(axis=1))
    r_max      = float(r_from_cen.max())

    bin_edges = np.linspace(0, r_max, 60)
    bin_r, bin_max = [], []
    for j in range(len(bin_edges) - 1):
        mask = (r_from_cen >= bin_edges[j]) & (r_from_cen < bin_edges[j + 1])
        if mask.sum() > 0:
            bin_r.append(0.5 * (bin_edges[j] + bin_edges[j + 1]))
            bin_max.append(float(noise_flat[mask].max()))

    ax_decay.plot(bin_r, bin_max, "b-", lw=2.0, label="Max dB(A)")
    for lv in contour_levels:
        ax_decay.axhline(lv, color="grey", lw=0.7, linestyle="--", alpha=0.7)
        ax_decay.text(r_max * 0.97, lv + 0.4, f"{lv:g}", fontsize=11,
                      ha="right", va="bottom", color="grey")

    # Mark approximate contour radii
    for lv in contour_levels:
        mask_lv = noise_flat >= lv
        if mask_lv.any():
            r_lv = float(r_from_cen[mask_lv].max())
            ax_decay.axvline(r_lv, color="grey", lw=0.5, linestyle=":", alpha=0.6)

    ax_decay.set_xlabel("Distance from layout centroid (m)", fontsize=13)
    ax_decay.set_ylabel("Max A-weighted SPL  dB(A)", fontsize=13)
    ax_decay.set_title("Noise vs. Distance from Layout Centroid",
                        fontsize=14, fontweight="bold")
    ax_decay.set_xlim(0, r_max)
    y_lo = max(min(contour_levels) - 5, 15)
    y_hi = max(float(noise_flat[np.isfinite(noise_flat)].max()) + 3, 60)
    ax_decay.set_ylim(y_lo, y_hi)
    ax_decay.grid(True, alpha=0.3)
    ax_decay.tick_params(labelsize=12)

    # Contour distance table as text
    table_lines = [f"{'Level':>8}  {'Max radius':>12}"]
    table_lines.append("─" * 23)
    for lv in contour_levels:
        mask_lv = noise_flat >= lv
        r_lv    = float(r_from_cen[mask_lv].max()) if mask_lv.any() else 0.0
        table_lines.append(f"{lv:>5g} dB(A)  {r_lv:>7.0f} m")
    ax_decay.text(
        0.98, 0.97, "\n".join(table_lines),
        transform=ax_decay.transAxes, fontsize=11, ha="right", va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0",
                  edgecolor="#aaaaaa", linewidth=0.8, alpha=0.92))

    # ── Panel 4 : octave-band spectrum ────────────────────────────────────────
    freqs    = [str(f) for f in OCTAVE_BANDS]
    lw_vals  = [Lw_bands.get(f, 0.0) for f in OCTAVE_BANDS]
    lwa_vals = [Lw_bands.get(f, 0.0) + A_WEIGHTING[f] for f in OCTAVE_BANDS]

    x_pos = np.arange(len(OCTAVE_BANDS))
    bar_w = 0.38
    bars1 = ax_spec.bar(x_pos - bar_w / 2, lw_vals,  bar_w, label="Lw (dB)",
                         color="#3498db", edgecolor="black", linewidth=0.5)
    bars2 = ax_spec.bar(x_pos + bar_w / 2, lwa_vals, bar_w, label="Lw,A (dB(A))",
                         color="#e67e22", edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars1, lw_vals):
        if val > 0:
            ax_spec.text(bar.get_x() + bar.get_width() / 2, val + 0.4,
                          f"{val:.0f}", ha="center", va="bottom", fontsize=11)
    for bar, val in zip(bars2, lwa_vals):
        if val > -60:
            ax_spec.text(bar.get_x() + bar.get_width() / 2,
                          max(val, ax_spec.get_ylim()[0] if ax_spec.get_ylim()[1] else 0) + 0.4,
                          f"{val:.0f}", ha="center", va="bottom", fontsize=11)

    lw_total  = 10.0 * np.log10(sum(10.0 ** (v / 10.0) for v in lw_vals if v > 0))
    lwa_total = overall_lwa(Lw_bands)

    ax_spec.set_xticks(x_pos)
    ax_spec.set_xticklabels([f"{f}\nHz" for f in OCTAVE_BANDS], fontsize=12)
    ax_spec.set_ylabel("Level (dB / dB(A))", fontsize=13)
    ax_spec.set_title("Input Sound Power Spectrum", fontsize=14, fontweight="bold")
    ax_spec.legend(fontsize=12, loc="upper right")
    ax_spec.grid(axis="y", alpha=0.3)
    ax_spec.tick_params(labelsize=12)

    note = (f"Overall Lw = {lw_total:.1f} dB re 1 pW  "
            f"  |  Overall Lw,A = {lwa_total:.1f} dB(A)  "
            f"  |  Hub height = {hub_height:.0f} m  "
            f"  |  Turbines = {len(wtg_xy)}")
    fig.text(0.5, 0.008, note, ha="center", va="bottom", fontsize=14,
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbe6",
                       edgecolor="#c8a800", linewidth=1.2, alpha=0.92))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {save_path}")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Console output
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(wtg_xy, noise_grid, xx, yy, contour_levels,
                   Lw_bands, hub_height):
    noise_flat = noise_grid.ravel()
    grid_pts   = np.column_stack([xx.ravel(), yy.ravel()])
    centroid   = wtg_xy.mean(axis=0)
    r_from_cen = np.sqrt(((grid_pts - centroid) ** 2).sum(axis=1))

    print("\n" + "=" * 72)
    print("  WIND TURBINE NOISE ANALYSIS SUMMARY")
    print(f"  Turbines   : {len(wtg_xy)}")
    print(f"  Hub height : {hub_height:.0f} m")
    print(f"  Overall Lw,A input : {overall_lwa(Lw_bands):.1f} dB(A) re 1 pW")
    print("=" * 72)

    print(f"\n  {'Distance from centroid':28}  {'Max Lp,A':>10}")
    print("  " + "─" * 42)
    for r in [200, 300, 500, 750, 1000, 1500, 2000, 3000]:
        mask = (r_from_cen >= r * 0.88) & (r_from_cen <= r * 1.12)
        if mask.sum() > 0:
            print(f"  {r:<28d}  {noise_flat[mask].max():>9.1f} dB(A)")

    print(f"\n  Contour extents (approximate):")
    for lv in contour_levels:
        pts_above = r_from_cen[noise_flat >= lv]
        if len(pts_above) > 0:
            print(f"    ≥ {lv:>3g} dB(A)  →  {pts_above.max():.0f} m from centroid")
    print("=" * 72 + "\n")


def save_noise_csv(xx, yy, noise_grid, save_path: str):
    df = pd.DataFrame({
        "X":        np.round(xx.ravel(), 1),
        "Y":        np.round(yy.ravel(), 1),
        "Lp_A_dBA": np.round(noise_grid.ravel(), 2),
    })
    df.to_csv(save_path, index=False)
    print(f"  Noise grid saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI helpers
# ══════════════════════════════════════════════════════════════════════════════

def _prompt(label, default, cast=str):
    raw = input(f"  {label} [{default}]: ").strip()
    return cast(raw) if raw else cast(str(default))


def _auto_detect(script_dir: str):
    """Scan directory for WTG layout and XYZ terrain files."""
    wtg_path = xyz_path = ""
    for fname in sorted(os.listdir(script_dir)):
        if not fname.endswith((".csv", ".txt", ".xyz")):
            continue
        fl   = fname.lower()
        path = os.path.join(script_dir, fname)
        if any(k in fl for k in ("xyz", "contour", "dem", "terrain")):
            if not xyz_path:
                xyz_path = path
        elif any(k in fl for k in ("wtg", "turbine", "layout", "wind_farm", "farm")):
            if not wtg_path:
                wtg_path = path
    return xyz_path, wtg_path


def _find_octave_csv(script_dir: str) -> str:
    """Return path to an octave/Lw CSV in script_dir, or empty string."""
    for fname in sorted(os.listdir(script_dir)):
        if fname.endswith(".csv") and any(
                k in fname.lower() for k in
                ("lw", "sound", "noise", "octave", "spectrum")):
            return os.path.join(script_dir, fname)
    return ""


def _load_Lw_csv(csv_path: str) -> dict:
    """Load octave- or 1/3-octave-band Lw from a CSV and return octave-band dict."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    freq_col = next((c for c in df.columns if "freq" in c), df.columns[0])
    lw_col   = next(
        (c for c in df.columns if any(k in c for k in ("lw", "level", "db"))),
        df.columns[1])
    raw_data = {int(r[freq_col]): float(r[lw_col]) for _, r in df.iterrows()}
    if len(raw_data) > 10:
        print("  1/3-octave data detected — summing to octave bands…")
        return third_oct_to_octave(raw_data)
    return {f: v for f, v in raw_data.items() if f in OCTAVE_BANDS}


def _load_Lw(script_dir: str) -> dict:
    """
    Interactively load octave-band or 1/3-octave-band Lw values.
    Returns {freq_hz: Lw_dB} octave-band dict.
    """
    auto_csv = _find_octave_csv(script_dir)

    if auto_csv:
        print(f"\n  Octave data detected: {os.path.basename(auto_csv)}")
        ans = input("  Press Enter to use it, or type a custom CSV path: ").strip()
        if ans == "":
            return _load_Lw_csv(auto_csv)
        if os.path.isfile(ans):
            return _load_Lw_csv(ans)
        print(f"  File not found: {ans} — falling back to {os.path.basename(auto_csv)}")
        return _load_Lw_csv(auto_csv)

    print("\n  Sound power level (Lw) input:")
    print("    1 = Enter octave-band values manually   (8 bands)")
    print("    2 = Enter 1/3-octave-band values manually  (24 bands)")
    print("    3 = Load from CSV file  [columns: freq_hz, Lw_dB]")
    choice = (input("  Choice [1]: ").strip() or "1").strip()

    # ── option 3: CSV ────────────────────────────────────────────────────────
    if choice == "3":
        raw = input("  CSV path: ").strip()
        if not raw or not os.path.isfile(raw):
            raise FileNotFoundError(f"File not found: {raw}")
        return _load_Lw_csv(raw)

    # ── options 1 & 2: manual entry ──────────────────────────────────────────
    is_third = (choice == "2")
    bands    = sorted(THIRD_OCT_TO_OCT.keys()) if is_third else OCTAVE_BANDS
    label    = "1/3-octave" if is_third else "octave"
    print(f"\n  Enter Lw (dB re 1 pW) for each {label} band.")
    print("  Press Enter to accept the default (typical 3 MW wind turbine).")

    raw_data = {}
    for f in bands:
        parent  = THIRD_OCT_TO_OCT.get(f, f) if is_third else f
        default = round(_DEFAULT_LW.get(parent, 95.0) +
                        (_THIRD_OCT_DEFAULT_OFFSET if is_third else 0.0), 1)
        val = _prompt(f"    {f:>6} Hz", default, float)
        raw_data[f] = val

    if is_third:
        return third_oct_to_octave(raw_data)
    return {int(f): float(v) for f, v in raw_data.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    auto_xyz, auto_wtg = _auto_detect(script_dir)

    print("\n" + "=" * 65)
    print("  Wind Turbine Noise Contour Analyser")
    print("  Propagation model: ISO 9613-2 simplified method")
    print("=" * 65 + "\n")

    # ── WTG layout ───────────────────────────────────────────────────────────
    if auto_wtg and os.path.isfile(auto_wtg):
        print(f"  WTG layout detected: {os.path.basename(auto_wtg)}")
        ans = input("  Press Enter to use it, or type a custom path: ").strip()
        wtg_path = ans if (ans and os.path.isfile(ans)) else auto_wtg
    else:
        wtg_path = input("  WTG layout CSV (full path): ").strip()
        if not os.path.isfile(wtg_path):
            raise FileNotFoundError(f"File not found: {wtg_path}")

    wtg = pd.read_csv(wtg_path)
    wtg.columns = [c.strip().lstrip("﻿").upper() for c in wtg.columns]
    wtg.dropna(subset=["X", "Y"], inplace=True)
    wtg_xy = wtg[["X", "Y"]].values.astype(float)
    print(f"  Loaded {len(wtg_xy)} turbines.")

    # ── hub height ───────────────────────────────────────────────────────────
    hub_height = _prompt("\nHub height (m)", DEFAULT_HUB_HEIGHT_M, float)

    # ── receiver height ──────────────────────────────────────────────────────
    hr = _prompt("Receiver height above ground (m)", DEFAULT_RECEIVER_HT_M, float)

    # ── ground type ──────────────────────────────────────────────────────────
    print("\n  Ground factor G:")
    print("    0   = Hard  (concrete, tarmac, water)")
    print("    0.5 = Mixed (typical agricultural / short grass)")
    print("    1   = Soft  (long grass, crops, scrub, forest floor)")
    G = max(0.0, min(1.0, _prompt("G", DEFAULT_GROUND_FACTOR, float)))

    # ── octave-band Lw ───────────────────────────────────────────────────────
    Lw_bands = _load_Lw(script_dir)
    lwa_in   = overall_lwa(Lw_bands)
    print(f"\n  Lw loaded — overall A-weighted Lw = {lwa_in:.1f} dB(A) re 1 pW")
    print("  Octave-band Lw (dB):  " +
          "  ".join(f"{f}: {Lw_bands.get(f, 0):.0f}" for f in OCTAVE_BANDS))

    # ── terrain data ─────────────────────────────────────────────────────────
    if auto_xyz and os.path.isfile(auto_xyz):
        ans = input(
            f"\n  XYZ terrain file detected: {os.path.basename(auto_xyz)}\n"
            "  Press Enter to use it, type 'dem' to auto-download SRTM, "
            "or enter a custom path: "
        ).strip()
        if ans.lower() in ("dem", "d"):
            xyz_path = None
        elif ans == "":
            xyz_path = auto_xyz
        elif os.path.isfile(ans):
            xyz_path = ans
        else:
            xyz_path = auto_xyz
    else:
        raw = input(
            "\n  XYZ terrain file (full path), or press Enter to "
            "auto-download SRTM DEM: "
        ).strip()
        xyz_path = raw if (raw and os.path.isfile(raw)) else None

    epsg_code = _prompt("EPSG code for WTG coordinate system", DEFAULT_EPSG, int)

    if xyz_path:
        print(f"\n  Loading terrain: {os.path.basename(xyz_path)}…")
        xyz = pd.read_csv(xyz_path)
        xyz.columns = [c.strip().lstrip("﻿").upper() for c in xyz.columns]
    else:
        print(f"\n  Downloading SRTM 30 m DEM via OpenTopoData "
              f"(EPSG:{epsg_code}, 5 km buffer)…")
        xyz = fetch_srtm_elevation(wtg_xy, epsg_code, buffer_m=SRTM_BUFFER_M, grid_n=SRTM_GRID_N)
        print(f"  DEM ready — {len(xyz):,} elevation points.\n")

    get_elev = _build_elev_interp(xyz)

    # ── satellite basemap ────────────────────────────────────────────────────
    use_satellite = False
    bing_key      = None
    if _HAS_CTX:
        print("\n  Satellite basemap:")
        print("    1 = ESRI World Imagery  (free, no API key)")
        print("    2 = Bing Aerial         (requires a Bing Maps API key)")
        print("    3 = None                (terrain contours only)")
        sat_choice = (input("  Choice [1]: ").strip() or "1").strip()
        if sat_choice == "2":
            bing_key      = input("  Bing Maps API key: ").strip()
            use_satellite = bool(bing_key)
        elif sat_choice == "1":
            use_satellite = True
    else:
        print(
            "\n  contextily not installed — satellite imagery unavailable.\n"
            "  To enable:  pip install contextily\n"
            "  Continuing with terrain contours only.")

    # ── noise grid parameters ────────────────────────────────────────────────
    resolution = _prompt("\nNoise grid resolution (points per side)", DEFAULT_GRID_RESOLUTION, int)
    buffer_m   = _prompt("Grid buffer beyond turbine layout (m)", DEFAULT_GRID_BUFFER_M, float)

    xmin = wtg_xy[:, 0].min() - buffer_m
    xmax = wtg_xy[:, 0].max() + buffer_m
    ymin = wtg_xy[:, 1].min() - buffer_m
    ymax = wtg_xy[:, 1].max() + buffer_m

    xi = np.linspace(xmin, xmax, resolution)
    yi = np.linspace(ymin, ymax, resolution)
    xx, yy = np.meshgrid(xi, yi)

    print("\n  Interpolating terrain elevation on noise grid…")
    grid_pts_flat = np.column_stack([xx.ravel(), yy.ravel()])
    elev_flat     = get_elev(grid_pts_flat)
    elev_grid     = elev_flat.reshape(xx.shape)
    wtg_elevs     = get_elev(wtg_xy)

    # ── contour levels ───────────────────────────────────────────────────────
    default_levels_str = ", ".join(str(int(x)) for x in DEFAULT_CONTOUR_LEVELS)
    custom = input(
        f"\n  Noise contour levels to draw — comma-separated dB(A)\n"
        f"  [{default_levels_str}]: "
    ).strip()
    try:
        contour_levels = sorted(float(x) for x in custom.split(",")) if custom else \
                         list(DEFAULT_CONTOUR_LEVELS)
    except ValueError:
        contour_levels = list(DEFAULT_CONTOUR_LEVELS)

    # ── run noise computation ────────────────────────────────────────────────
    print(f"\n  Computing noise grid "
          f"({resolution}×{resolution} = {resolution**2:,} points, "
          f"{len(wtg_xy)} turbines)…")
    noise_grid = compute_noise_grid(
        wtg_xy, wtg_elevs, Lw_bands, hub_height,
        xx, yy, elev_grid, hr=hr, G=G)

    print_summary(wtg_xy, noise_grid, xx, yy, contour_levels,
                  Lw_bands, hub_height)

    # ── save and display ─────────────────────────────────────────────────────
    csv_path = os.path.join(script_dir, "wind_noise_levels.csv")
    save_noise_csv(xx, yy, noise_grid, csv_path)

    png_path = os.path.join(script_dir, "wind_noise_results.png")
    plot_results(
        wtg_xy, noise_grid, xx, yy, elev_grid,
        Lw_bands, hub_height, contour_levels, epsg_code,
        use_satellite=use_satellite, bing_key=bing_key,
        save_path=png_path)
    plt.show()


if __name__ == "__main__":
    main()
