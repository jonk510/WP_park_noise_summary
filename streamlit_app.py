"""
JK's Prelim Wind Turbine Noise Contour Estimator — Streamlit Web App
"""

import io
import os
import sys
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wind_noise_analyser import (
    OCTAVE_BANDS, THIRD_OCT_TO_OCT, A_WEIGHTING, _DEFAULT_LW, _THIRD_OCT_DEFAULT_OFFSET,
    DEFAULT_HUB_HEIGHT_M, DEFAULT_RECEIVER_HT_M, DEFAULT_GROUND_FACTOR,
    DEFAULT_EPSG, DEFAULT_GRID_RESOLUTION, DEFAULT_GRID_BUFFER_M,
    DEFAULT_CONTOUR_LEVELS,
    third_oct_to_octave, overall_lwa, compute_noise_grid,
    fetch_srtm_elevation, _build_elev_interp, plot_results,
    _HAS_PYPROJ,
)

# Exact 1/3-octave centre frequencies from the GE spreadsheet (31 bands, 10–10000 Hz)
THIRD_OCT_BANDS = [
    10, 12.5, 16, 20, 25, 31.5, 40,
    50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
]

# GE 164-6.0 Lw per 1/3-octave band (dB re 1 pW) — from GE 164_6.0_sound_power_levels.xlsx
_GE_164_6_LW_3RD = {
    10: 0.0,  12.5: 52.0, 16: 58.5, 20: 63.8, 25: 68.6, 31.5: 72.9, 40: 76.9,
     50: 80.2,  63: 83.1,   80: 85.3,
    100: 87.0, 125: 88.7,  160: 90.2,
    200: 91.7, 250: 93.3,  315: 94.6,
    400: 95.1, 500: 95.9,  630: 96.6,
    800: 97.0, 1000: 97.5, 1250: 98.0,
    1600: 96.7, 2000: 95.3, 2500: 93.3,
    3150: 90.6, 4000: 86.8, 5000: 82.6,
    6300: 76.3, 8000: 66.6, 10000: 53.9,
}
# Octave-band equivalent — un-weight from Lwa before summing (GE data is A-weighted)
_GE_164_6_LW_OCT = third_oct_to_octave(_GE_164_6_LW_3RD, a_weighted=True)

# ── Shapefile loader ──────────────────────────────────────────────────────────
def _load_shapefile_points(uploaded_files, target_epsg: int):
    """Write uploaded shapefile parts to a temp dir, read with geopandas, return (xy, names)."""
    try:
        import geopandas as gpd
    except ImportError:
        st.error("Install `geopandas` to use shapefile upload.")
        return None, None
    with tempfile.TemporaryDirectory() as tmp:
        for f in uploaded_files:
            with open(os.path.join(tmp, f.name), "wb") as fh:
                fh.write(f.read())
        shp_files = [p for p in os.listdir(tmp) if p.endswith(".shp")]
        if not shp_files:
            st.error("No .shp file found in the uploaded set.")
            return None, None
        gdf = gpd.read_file(os.path.join(tmp, shp_files[0]))
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[gdf.geometry.geom_type.isin(["Point", "MultiPoint"])]
    if gdf.empty:
        st.error("Shapefile contains no point features.")
        return None, None
    gdf = gdf.to_crs(epsg=target_epsg)
    xy = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    name_col = next((c for c in gdf.columns if c.lower() in ("name", "label", "id", "receptor")), None)
    names = gdf[name_col].astype(str).tolist() if name_col else [f"R{i+1}" for i in range(len(xy))]
    return xy, names


# ── KMZ / KML loader ─────────────────────────────────────────────────────────
def _load_kmz_points(uploaded_file, target_epsg: int):
    """Parse a KMZ or KML file and return (xy, names) reprojected to target_epsg."""
    import zipfile
    import xml.etree.ElementTree as ET
    from pyproj import Transformer

    raw = uploaded_file.read()
    # KMZ = zip archive containing a .kml file; KML = plain XML
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                st.error("No KML found inside KMZ.")
                return None, None
            kml_bytes = z.read(kml_names[0])
    else:
        kml_bytes = raw

    root = ET.fromstring(kml_bytes)

    # Strip all namespace prefixes so tags become plain local names
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    lons, lats, names = [], [], []
    for pm in root.iter("Placemark"):
        pt = pm.find(".//Point")
        if pt is None:
            continue
        coords_el = pt.find("coordinates")
        if coords_el is None or not coords_el.text:
            continue
        parts = coords_el.text.strip().split(",")
        lons.append(float(parts[0]))
        lats.append(float(parts[1]))
        name_el = pm.find("name")
        names.append(name_el.text.strip() if name_el is not None and name_el.text else f"P{len(lons)}")

    if not lons:
        st.error("No point features found in KMZ/KML.")
        return None, None

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
    xs, ys = transformer.transform(lons, lats)
    return np.column_stack([xs, ys]), names


# ── Load turbine presets from Excel if present ────────────────────────────────
_SPECTRA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "WTG_Acoustic_Spectra_Loudest_Modes 1.xlsx")

@st.cache_data
def _load_wtg_presets():
    """Return dict {display_name: (data_dict {freq: Lwa_dB}, is_third_oct)}."""
    if not os.path.exists(_SPECTRA_FILE):
        return {}
    try:
        xl = pd.ExcelFile(_SPECTRA_FILE)
        presets = {}
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            freq_col = next((c for c in df.columns if "freq" in c.lower()), None)
            lw_col   = next((c for c in df.columns if "lw"   in c.lower()), None)
            if freq_col is None or lw_col is None:
                continue
            df = df.dropna(subset=[lw_col])
            if df.empty:
                continue
            data = {float(r[freq_col]): float(r[lw_col]) for _, r in df.iterrows()}
            _OCTAVE_SET = {63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0}
            is_third = any(f not in _OCTAVE_SET for f in data.keys())
            # Clean display name: drop suffix, underscores → spaces
            name = sheet.replace("_1-3oct", "").replace("_1-1oct", "").replace("_", " ")
            presets[name] = (data, is_third)
        return presets
    except Exception:
        return {}

_WTG_PRESETS = _load_wtg_presets()

st.set_page_config(
    page_title="Wind Turbine Noise Analyser",
    layout="wide",
)

st.title("JK's Prelim Wind Turbine Noise Contour Estimator")
st.caption("ISO 9613-2 simplified propagation model")

with st.expander("About this tool — methodology, assumptions & standards"):
    st.markdown("""
**Propagation model — ISO 9613-2 (simplified method)**

Noise levels are estimated using the simplified outdoor sound propagation method
defined in ISO 9613-2:1996. For each octave band *f* (63 Hz – 8 kHz), the
A-weighted sound pressure level at a receiver is:

> Lp,A,f = Lw,f − A_div − A_atm,f − A_gr,f + ΔA,f

where:

| Term | Description |
|---|---|
| **A_div** = 20·log₁₀(d) + 11 | Geometric spreading loss (dB) |
| **A_atm** = α·d / 1000 | Atmospheric absorption per ISO 9613-1 (10 °C, 80 % RH) |
| **A_gr** = A_s + A_m + A_r | Ground effect — source, middle and receiver regions |
| **ΔA** | A-weighting correction for each octave band |

The total A-weighted level at a receiver is the energy sum over all octave bands
and all turbines:

> Lp,A = 10·log₁₀( Σ_turbines Σ_bands 10^(Lp,A,f / 10) )

---

**South Australian Wind Farm Environmental Noise Guidelines (2021)**

This tool's defaults are configured to match the modelling parameters used in
professional wind farm noise assessments under the
*SA Wind Farm Environmental Noise Guidelines (2021)*:

| Parameter | SA Guidelines | This tool default |
|---|---|---|
| Propagation model | ISO 9613-2 (simplified method) | ✅ ISO 9613-2:1996 §7.3 Table 3 |
| Ground factor G | **0.5** (50 % acoustically porous / mixed) | ✅ G = 0.5 |
| Receiver height | **4.0 m** above ground | ✅ 4.0 m |
| Temperature | 10 °C | ✅ 10 °C |
| Relative humidity | **80 %** | ✅ 80 % RH |
| Frequency range | 1/3-octave, 20 Hz – 10 kHz | ✅ 10 Hz – 10 kHz |
| Barriers / shielding | Not modelled (conservative) | ✅ Optional — ISO 9613-2 §8 (toggle in sidebar) |

**SA EPA assessment criteria (rural):**
- **35 dB(A)** LAeq — predominantly rural living / residential areas
- **40 dB(A)** LAeq — primary production areas
- Or 5 dB(A) above the measured LA90 background level, whichever is greater

---

**Ground factor G — effect on predictions**

- **0** — hard ground (concrete, asphalt, water) — highest noise levels (most conservative)
- **0.5** — mixed ground (typical agricultural land, short grass) — *SA Guidelines default*
- **1** — soft ground (long grass, crops, scrubland, forest floor) — lowest noise levels

Low-frequency ground effect is reduced by a factor of 0.50 at 63 Hz and 0.75 at 125 Hz,
consistent with ISO 9613-2 Table 2.

---

**Assumptions & limitations**

- Terrain **shielding (A_bar) is optional** — when disabled, results are
  upper-bound (conservative) estimates suitable for early-stage screening.
  When enabled, the ISO 9613-2 §8 single-edge Maekawa formula is applied
  using the dominant terrain ridge between each source and receiver.
- A uniform sound power spectrum is applied to all turbines.
- Atmospheric conditions are fixed at 10 °C / 70 % RH for absorption coefficients.
- Terrain elevation affects the effective hub height and slant distance but not
  diffraction over ridges.
- The noise grid is calculated on a uniform 2-D plane at the specified receiver height.
- This tool produces indicative results only and does not replace a full acoustic assessment.

---

**Standards referenced**

- ISO 9613-1:1993 — Attenuation of sound during propagation outdoors (atmospheric absorption)
- ISO 9613-2:1996 — General method of calculation (simplified method)
- EPA South Australia — Wind Farms Environmental Noise Guidelines (2009)
""")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Site Settings")
    hub_height = st.number_input(
        "Hub height (m)", value=float(DEFAULT_HUB_HEIGHT_M), min_value=10.0, step=5.0)
    hr = st.number_input(
        "Receiver height (m)", value=float(DEFAULT_RECEIVER_HT_M), min_value=0.5, step=0.5)
    G = st.slider(
        "Ground factor G", 0.0, 1.0, float(DEFAULT_GROUND_FACTOR), 0.05,
        help="0 = hard (concrete/water)   0.5 = mixed   1 = soft (grass/crops)")
    use_shielding = st.toggle(
        "Terrain shielding (A_bar)", value=False,
        help="ISO 9613-2 §8 — samples the terrain profile between each turbine "
             "and receiver, finds the dominant ridge, and applies Maekawa/ISO "
             "barrier attenuation (capped at 20 dB). More realistic results "
             "but slightly longer computation.")
    epsg_code = st.number_input(
        "Coordinate System's EPSG code", value=int(DEFAULT_EPSG), min_value=1000, max_value=99999, step=1)
    try:
        from pyproj import CRS
        _crs_name = CRS.from_epsg(int(epsg_code)).name
        st.caption(f"📐 {_crs_name}")
    except Exception:
        st.caption("⚠️ Unrecognised EPSG code")

    st.divider()
    st.header("Grid")
    grid_spacing_m = st.number_input(
        "Grid spacing (m)", value=50.0, min_value=10.0, max_value=500.0, step=10.0,
        help="Spacing between grid points in metres. Smaller = finer detail but slower.")
    buffer_m = st.number_input(
        "Buffer beyond layout (m)", value=float(DEFAULT_GRID_BUFFER_M), min_value=500.0, step=500.0)

    st.divider()
    st.header("Contour levels dB(A)")
    levels_str = st.text_input(
        "Comma-separated",
        value=", ".join(str(int(x)) for x in DEFAULT_CONTOUR_LEVELS))
    try:
        contour_levels = sorted(float(x.strip()) for x in levels_str.split(",") if x.strip())
    except ValueError:
        contour_levels = list(DEFAULT_CONTOUR_LEVELS)
        st.warning("Invalid levels — using defaults.")

    alpha_fill = st.slider(
        "Contour opacity", 0.10, 1.0, 0.55, 0.05,
        help="Opacity of filled noise contour bands")

# ── Input columns ─────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

# Column 1 — WTG layout
with c1:
    st.subheader("1 · Turbine Layout")
    wtg_fmt = st.radio("Format", ["CSV", "Shapefile", "KMZ / KML"], horizontal=True, key="wtg_fmt")
    wtg_xy = None
    if wtg_fmt == "CSV":
        wtg_file = st.file_uploader("CSV with X, Y columns", type=["csv", "txt"], key="wtg_csv")
        if wtg_file:
            wtg_df = pd.read_csv(wtg_file)
            wtg_df.columns = [c.strip().lstrip("﻿").upper() for c in wtg_df.columns]
            wtg_df.dropna(subset=["X", "Y"], inplace=True)
            wtg_xy = wtg_df[["X", "Y"]].values.astype(float)
            st.success(f"{len(wtg_xy)} turbines loaded")
            display_df = wtg_df[["X", "Y"]].copy()
            display_df.index = range(1, len(display_df) + 1)
            st.dataframe(display_df, use_container_width=True)
    elif wtg_fmt == "Shapefile":
        wtg_shp = st.file_uploader(
            "Shapefile parts (.shp, .shx, .dbf, .prj)",
            type=["shp", "shx", "dbf", "prj", "cpg", "qmd"],
            accept_multiple_files=True, key="wtg_shp")
        if wtg_shp:
            wtg_xy, _ = _load_shapefile_points(wtg_shp, int(epsg_code))
            if wtg_xy is not None:
                st.success(f"{len(wtg_xy)} turbines loaded")
                display_df = pd.DataFrame(wtg_xy, columns=["X", "Y"])
                display_df.index = range(1, len(display_df) + 1)
                st.dataframe(display_df, use_container_width=True)
    else:
        wtg_kmz = st.file_uploader("KMZ or KML file", type=["kmz", "kml"], key="wtg_kmz")
        if wtg_kmz:
            wtg_xy, _ = _load_kmz_points(wtg_kmz, int(epsg_code))
            if wtg_xy is not None:
                st.success(f"{len(wtg_xy)} turbines loaded")
                display_df = pd.DataFrame(wtg_xy, columns=["X", "Y"])
                display_df.index = range(1, len(display_df) + 1)
                st.dataframe(display_df, use_container_width=True)

# Column 2 — Sound power
with c2:
    st.subheader("2 · WTG's Sound Power Lw Curve")
    _source_opts = ["Manual entry", "Upload CSV"]
    if _WTG_PRESETS:
        _source_opts.insert(0, "Turbine preset")
    lw_method = st.radio("Source", _source_opts, horizontal=True)
    Lw_bands = {}
    lwa_display = None  # set to direct energy sum when input is already A-weighted

    if lw_method == "Turbine preset":
        _preset_names = list(_WTG_PRESETS.keys())
        _selected = st.selectbox("Select turbine", _preset_names)
        _data, _is_third = _WTG_PRESETS[_selected]
        if _is_third:
            Lw_bands = third_oct_to_octave(_data, a_weighted=True)
            lwa_display = 10 * np.log10(
                sum(10 ** (v / 10) for v in _data.values() if v > 0))
        else:
            Lw_bands = {int(f): v - A_WEIGHTING.get(int(f), 0.0)
                        for f, v in _data.items() if int(f) in OCTAVE_BANDS}
            lwa_display = 10 * np.log10(
                sum(10 ** (v / 10) for v in _data.values() if v > 0))
        with st.expander("Loaded spectrum"):
            _disp = {f: round(Lw_bands[f] + A_WEIGHTING[f], 1) for f in OCTAVE_BANDS if f in Lw_bands}
            st.dataframe(pd.DataFrame.from_dict(
                {"Freq (Hz)": list(_disp.keys()), "Lw,A dB(A)": list(_disp.values())}),
                hide_index=True, use_container_width=True)

    elif lw_method == "Upload CSV":
        lw_file = st.file_uploader("Lw CSV (freq_hz, Lw_dB or Lwa_dB)", type=["csv"], key="lw")
        if lw_file:
            df_lw = pd.read_csv(lw_file)
            df_lw.columns = [c.strip().lower() for c in df_lw.columns]
            fc = next((c for c in df_lw.columns if "freq" in c), df_lw.columns[0])
            lc = next(
                (c for c in df_lw.columns if any(k in c for k in ("lw", "level", "db"))),
                df_lw.columns[1])
            col_is_lwa = "lwa" in lc.lower()
            csv_a_weighted = st.toggle(
                "Values are Lwa (A-weighted)", value=col_is_lwa, key="csv_aw",
                help="Enable if your CSV column contains A-weighted sound power (Lwa). "
                     "The model will un-weight them before propagation.")
            raw = {int(r[fc]): float(r[lc]) for _, r in df_lw.iterrows()}
            if len(raw) > 10:
                st.info(f"1/3-octave data detected ({len(raw)} bands) — converting to octave bands.")
                Lw_bands = third_oct_to_octave(raw, a_weighted=csv_a_weighted)
                if csv_a_weighted:
                    lwa_display = 10 * np.log10(sum(10**(v/10) for v in raw.values() if v > 0))
            else:
                if csv_a_weighted:
                    st.info("Octave-band Lwa loaded — un-weighting before propagation.")
                    raw_oct = {f: v for f, v in raw.items() if f in OCTAVE_BANDS}
                    Lw_bands = {f: v - A_WEIGHTING.get(f, 0.0) for f, v in raw_oct.items()}
                    lwa_display = 10 * np.log10(sum(10**(v/10) for v in raw_oct.values()))
                else:
                    Lw_bands = {f: v for f, v in raw.items() if f in OCTAVE_BANDS}
                    st.info(f"Octave-band Lw loaded ({len(Lw_bands)} bands).")
            st.success("Lw loaded.")
    elif lw_method == "Manual entry":
        band_type = st.radio(
            "Band resolution",
            ["Octave (8 bands)", "1/3-Octave (31 bands)"],
            index=1,
            horizontal=True,
            help="Choose the format provided by your OEM")

        if band_type == "Octave (8 bands)":
            oct_a_weighted = st.toggle(
                "Values are Lwa (A-weighted)", value=True, key="oct_aw",
                help="Enable if your data is A-weighted per octave band (Lwa).")
            sub_cols = st.columns(2)
            raw_manual = {}
            for i, f in enumerate(OCTAVE_BANDS):
                with sub_cols[i % 2]:
                    # Defaults are Lwa (A-weighted) per octave band
                    _lwa_def = round(float(_GE_164_6_LW_OCT.get(f, 95.0)) + A_WEIGHTING.get(f, 0.0), 1)
                    raw_manual[f] = st.number_input(
                        f"{f} Hz", value=_lwa_def,
                        min_value=0.0, max_value=130.0, step=0.5, key=f"lw_oct_{f}")
            if oct_a_weighted:
                Lw_bands = {f: v - A_WEIGHTING.get(f, 0.0) for f, v in raw_manual.items()}
                lwa_display = 10 * np.log10(sum(10**(v/10) for v in raw_manual.values()))
            else:
                Lw_bands = raw_manual
        else:
            third_a_weighted = st.toggle(
                "Values are Lwa (A-weighted)", value=True, key="third_aw",
                help="Enable if your OEM data is A-weighted per 1/3-octave band (Lwa,p). "
                     "Most OEM data sheets use Lwa. The model will un-weight before propagation.")
            st.caption("Default: GE 164-6.0 — edit any value to customise")
            sub_cols = st.columns(3)
            raw_manual = {}
            for i, f in enumerate(THIRD_OCT_BANDS):
                default = float(_GE_164_6_LW_3RD.get(f, round(
                    _DEFAULT_LW.get(THIRD_OCT_TO_OCT.get(int(f), 63), 95.0) + _THIRD_OCT_DEFAULT_OFFSET, 1)))
                with sub_cols[i % 3]:
                    raw_manual[f] = st.number_input(
                        f"{f} Hz", value=default,
                        min_value=0.0, max_value=130.0, step=0.1, key=f"lw_3rd_{f}")
            Lw_bands = third_oct_to_octave(raw_manual, a_weighted=third_a_weighted)
            if third_a_weighted:
                lwa_display = 10 * np.log10(sum(10**(v/10) for v in raw_manual.values() if v > 0))

    if Lw_bands:
        display_val = lwa_display if lwa_display is not None else overall_lwa(Lw_bands)
        st.metric("Overall Lw,A", f"{display_val:.1f} dB(A)")

# Column 3 — Terrain & basemap
with c3:
    st.subheader("3 · Terrain & Basemap")
    terrain_src = st.radio(
        "Terrain source", ["Auto-download SRTM", "Upload XYZ/CSV"], horizontal=True)
    xyz_file = None
    if terrain_src == "Upload XYZ/CSV":
        xyz_file = st.file_uploader("XYZ CSV (X, Y, Z)", type=["csv", "txt", "xyz"], key="xyz")
        if xyz_file:
            st.success("Terrain file ready.")

    st.divider()

    try:
        import contextily  # noqa: F401
        _has_ctx = True
    except ImportError:
        _has_ctx = False

    use_satellite, bing_key = False, None
    if _has_ctx:
        sat_opt = st.selectbox("Basemap", ["ESRI World Imagery", "Bing Aerial", "None"],
                               index=0)
        if sat_opt == "Bing Aerial":
            bing_key = st.text_input("Bing Maps API key", type="password")
            use_satellite = bool(bing_key)
        elif sat_opt == "ESRI World Imagery":
            use_satellite = True
    else:
        st.warning("Install `contextily` to enable satellite imagery:\n"
                   "```\npip install contextily\n```")

# ── Sensitive receptors ───────────────────────────────────────────────────────
st.divider()
st.subheader("4 · Sensitive Receptors (optional)")
rec_fmt = st.radio("Format", ["CSV", "Shapefile", "KMZ / KML"], horizontal=True, key="rec_fmt")
receptor_xy, receptor_names = None, None
if rec_fmt == "CSV":
    rec_file = st.file_uploader(
        "CSV — columns: X, Y (and optionally Name)",
        type=["csv", "txt"], key="rec_csv")
    if rec_file:
        rec_df = pd.read_csv(rec_file)
        rec_df.columns = [c.strip().lstrip("﻿").upper() for c in rec_df.columns]
        rec_df.dropna(subset=["X", "Y"], inplace=True)
        receptor_xy = rec_df[["X", "Y"]].values.astype(float)
        receptor_names = rec_df["NAME"].tolist() if "NAME" in rec_df.columns else [f"R{i+1}" for i in range(len(receptor_xy))]
        st.success(f"{len(receptor_xy)} receptors loaded: {', '.join(receptor_names)}")
elif rec_fmt == "Shapefile":
    rec_shp = st.file_uploader(
        "Shapefile parts (.shp, .shx, .dbf, .prj)",
        type=["shp", "shx", "dbf", "prj", "cpg", "qmd"],
        accept_multiple_files=True, key="rec_shp")
    if rec_shp:
        receptor_xy, receptor_names = _load_shapefile_points(rec_shp, int(epsg_code))
        if receptor_xy is not None:
            st.success(f"{len(receptor_xy)} receptors loaded: {', '.join(receptor_names)}")
else:
    rec_kmz = st.file_uploader("KMZ or KML file", type=["kmz", "kml"], key="rec_kmz")
    if rec_kmz:
        receptor_xy, receptor_names = _load_kmz_points(rec_kmz, int(epsg_code))
        if receptor_xy is not None:
            st.success(f"{len(receptor_xy)} receptors loaded: {', '.join(receptor_names)}")

# ── Run button ────────────────────────────────────────────────────────────────
st.divider()
ready = (wtg_xy is not None) and bool(Lw_bands)
if not ready:
    st.info("Upload a turbine layout CSV and configure sound power levels to begin.")

if st.button("Run Noise Analysis", type="primary", disabled=not ready, use_container_width=True):
    with st.status("Running analysis…", expanded=True) as status:
        st.write("Preparing terrain elevation…")
        if terrain_src == "Upload XYZ/CSV" and xyz_file:
            xyz_file.seek(0)
            xyz = pd.read_csv(xyz_file)
            xyz.columns = [c.strip().lstrip("﻿").upper() for c in xyz.columns]
        else:
            if not _HAS_PYPROJ:
                st.error("Install `requests` and `pyproj` to use SRTM auto-download:\n"
                         "```\npip install requests pyproj\n```")
                st.stop()
            xyz = fetch_srtm_elevation(wtg_xy, int(epsg_code), buffer_m=5000.0, grid_n=40)

        get_elev = _build_elev_interp(xyz)

        xmin = wtg_xy[:, 0].min() - buffer_m
        xmax = wtg_xy[:, 0].max() + buffer_m
        ymin = wtg_xy[:, 1].min() - buffer_m
        ymax = wtg_xy[:, 1].max() + buffer_m
        nx = max(10, int(round((xmax - xmin) / grid_spacing_m)) + 1)
        ny = max(10, int(round((ymax - ymin) / grid_spacing_m)) + 1)
        xi = np.linspace(xmin, xmax, nx)
        yi = np.linspace(ymin, ymax, ny)
        xx, yy = np.meshgrid(xi, yi)
        elev_grid = get_elev(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        wtg_elevs = get_elev(wtg_xy)

        shield_note = " + terrain shielding" if use_shielding else ""
        st.write(f"Computing noise grid ({nx}×{ny} pts @ {grid_spacing_m:.0f} m, {len(wtg_xy)} turbines{shield_note})…")
        noise_grid = compute_noise_grid(
            wtg_xy, wtg_elevs, Lw_bands, hub_height,
            xx, yy, elev_grid, hr=float(hr), G=float(G),
            use_shielding=use_shielding)

        # Interpolate noise at receptor locations
        receptor_levels = None
        if receptor_xy is not None:
            from scipy.interpolate import RegularGridInterpolator
            st.write("Interpolating receptor noise levels…")
            interp = RegularGridInterpolator(
                (yi, xi), noise_grid, method="linear", bounds_error=False,
                fill_value=None)
            receptor_levels = interp(receptor_xy[:, ::-1])  # (Y, X) order

        st.write("Rendering figure…")
        fig = plot_results(
            wtg_xy, noise_grid, xx, yy, elev_grid,
            Lw_bands, hub_height, contour_levels, int(epsg_code),
            use_satellite=use_satellite, bing_key=bing_key,
            alpha_fill=alpha_fill, save_path=None,
            receptor_xy=receptor_xy, receptor_levels=receptor_levels,
            receptor_names=receptor_names)

        status.update(label="Analysis complete!", state="complete")

    st.session_state.results = dict(
        wtg_xy=wtg_xy, noise_grid=noise_grid,
        xx=xx, yy=yy, elev_grid=elev_grid, fig=fig,
        Lw_bands=Lw_bands, hub_height=hub_height,
        contour_levels=contour_levels,
        epsg_code=int(epsg_code),
        use_satellite=use_satellite, bing_key=bing_key,
        alpha_fill=alpha_fill,
        use_shielding=use_shielding,
        receptor_xy=receptor_xy, receptor_levels=receptor_levels,
        receptor_names=receptor_names,
    )

# ── Results ───────────────────────────────────────────────────────────────────
if "results" in st.session_state:
    r = st.session_state.results

    # Regenerate figure cheaply when only alpha changed (skip expensive noise recompute)
    if r["alpha_fill"] != alpha_fill:
        with st.spinner("Updating contour opacity…"):
            r["fig"] = plot_results(
                r["wtg_xy"], r["noise_grid"], r["xx"], r["yy"], r["elev_grid"],
                r["Lw_bands"], r["hub_height"], r["contour_levels"], r["epsg_code"],
                use_satellite=r["use_satellite"], bing_key=r["bing_key"],
                alpha_fill=alpha_fill, save_path=None,
                receptor_xy=r.get("receptor_xy"), receptor_levels=r.get("receptor_levels"),
                receptor_names=r.get("receptor_names"))
            r["alpha_fill"] = alpha_fill
            plt.close("all")

    noise_flat = r["noise_grid"].ravel()
    grid_pts   = np.column_stack([r["xx"].ravel(), r["yy"].ravel()])
    centroid   = r["wtg_xy"].mean(axis=0)
    r_from_cen = np.sqrt(((grid_pts - centroid) ** 2).sum(axis=1))

    st.subheader("Results")

    if r.get("receptor_levels") is not None:
        st.markdown("**Sensitive Receptor Noise Levels**")
        crit = 35.0
        rec_rows = []
        for name, lvl in zip(r["receptor_names"], r["receptor_levels"]):
            status_str = "🔴 Exceeds 40" if lvl > 40 else ("🟡 35–40" if lvl > 35 else "🟢 < 35")
            rec_rows.append({"Receptor": name, "dB(A)": round(float(lvl), 1), "SA Criterion": status_str})
        st.dataframe(pd.DataFrame(rec_rows).set_index("Receptor"), use_container_width=True)

    tab_map, tab_extents, tab_decay = st.tabs(["Map", "Contour Extents", "Distance Decay"])

    with tab_map:
        r["fig"].set_dpi(120)
        st.pyplot(r["fig"], use_container_width=True)
        dl1, dl2 = st.columns(2)
        with dl1:
            png_buf = io.BytesIO()
            r["fig"].savefig(png_buf, format="png", dpi=150, bbox_inches="tight")
            png_buf.seek(0)
            st.download_button(
                "Download PNG", png_buf,
                file_name="wind_noise_results.png", mime="image/png")
        with dl2:
            csv_str = pd.DataFrame({
                "X":        np.round(r["xx"].ravel(), 1),
                "Y":        np.round(r["yy"].ravel(), 1),
                "Lp_A_dBA": np.round(noise_flat, 2),
            }).to_csv(index=False)
            st.download_button(
                "Download noise grid CSV", csv_str,
                file_name="wind_noise_levels.csv", mime="text/csv")

    with tab_extents:
        rows = []
        for lv in r["contour_levels"]:
            pts = r_from_cen[noise_flat >= lv]
            rows.append({
                "Level dB(A)": int(lv),
                "Max radius from centroid (m)": f"{pts.max():.0f}" if len(pts) else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_decay:
        rows2 = []
        for dist in [200, 300, 500, 750, 1000, 1500, 2000, 3000]:
            mask = (r_from_cen >= dist * 0.88) & (r_from_cen <= dist * 1.12)
            if mask.sum() > 0:
                rows2.append({
                    "Distance (m)": dist,
                    "Max Lp,A dB(A)": f"{noise_flat[mask].max():.1f}",
                })
        st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)
