"""
WindPRO PARK + Noise Results Summary — Streamlit app

Upload WindPRO PARK PDF exports alongside a noise spectrum per calculation
to generate a branded PowerPoint with satellite maps, noise contours,
wake-loss charts, and a comparison table.
"""

import hashlib
import os
import re
import tempfile
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from park_summary import (
    DEFAULT_LOSSES, HAS_CTX, HAS_GPD, HAS_NOISE, _NOISE_IMPORT_ERR,
    _ordinal, apply_losses, build, compute_noise_overlay, extract, load_shapefile,
)


# ─────────────────────────────────────────────────────────────────────────────
# Password gate
# ─────────────────────────────────────────────────────────────────────────────

def _check_password() -> bool:
    def _submit():
        if st.session_state['_pw_input'] == st.secrets['app_password']:
            st.session_state['_authed'] = True
        else:
            st.session_state['_authed'] = False

    if st.session_state.get('_authed'):
        return True

    st.title('WindPRO PARK + Noise Results Summary')
    st.text_input('Password', type='password', key='_pw_input', on_change=_submit)
    if '_authed' in st.session_state and not st.session_state['_authed']:
        st.error('Incorrect password — try again.')
    st.stop()


_check_password()

_BUNDLED_TPL  = Path(__file__).parent / 'template01.pptx'
DEFAULT_TPL   = str(_BUNDLED_TPL) if _BUNDLED_TPL.exists() else None
_SPECTRA_FILE = Path(__file__).parent / 'WTG_Acoustic_Spectra_Loudest_Modes 1.xlsx'

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='PARK + Noise Summary',
    page_icon='💨',
    layout='wide',
)

st.title('WindPRO PARK + Noise Results Summary')
st.caption(
    'Upload WindPRO PARK PDF exports to generate a comparison presentation '
    'with satellite maps, noise contours, wake-loss charts, and a summary table.'
)

caps = []
if HAS_CTX:
    caps.append('🛰  Satellite basemap enabled')
else:
    caps.append('ℹ  Install `contextily` for satellite maps')
if HAS_NOISE and HAS_CTX:
    caps.append('🔊  Noise overlay enabled')
elif HAS_NOISE and not HAS_CTX:
    caps.append('ℹ  Noise overlay requires `contextily`')
else:
    caps.append('ℹ  Noise overlay unavailable — check `wind_noise_analyser` install')
st.caption('  ·  '.join(caps))

# ─────────────────────────────────────────────────────────────────────────────
# Noise preset loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def _load_wtg_presets() -> dict:
    """Return {display_name: (data_dict {freq: Lwa_dB}, is_third_oct)}."""
    if not _SPECTRA_FILE.exists():
        return {}
    try:
        xl = pd.ExcelFile(_SPECTRA_FILE)
        presets = {}
        for sheet in xl.sheet_names:
            df       = xl.parse(sheet)
            freq_col = next((c for c in df.columns if 'freq' in c.lower()), None)
            lw_col   = next((c for c in df.columns if 'lw'   in c.lower()), None)
            if freq_col is None or lw_col is None:
                continue
            df = df.dropna(subset=[lw_col])
            if df.empty:
                continue
            data     = {float(r[freq_col]): float(r[lw_col]) for _, r in df.iterrows()}
            oct_set  = {63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0}
            is_third = any(f not in oct_set for f in data.keys())
            name     = (sheet.replace('_1-3oct', '').replace('_1-1oct', '')
                             .replace('_', ' '))
            presets[name] = (data, is_third)
        return presets
    except Exception:
        return {}


_WTG_PRESETS = _load_wtg_presets()


def _best_preset_match(wtg_model: str, preset_names: list[str]) -> str | None:
    """Fuzzy-match a WTG model name to the closest noise preset."""
    if not wtg_model or not preset_names:
        return None
    norm_model = wtg_model.lower().replace('-', ' ').replace('/', ' ').replace('_', ' ')
    best_score, best_name = 0.0, None
    for name in preset_names:
        norm_name    = name.lower().replace('-', ' ').replace('_', ' ')
        seq_score    = SequenceMatcher(None, norm_model, norm_name).ratio()
        model_tokens = set(norm_model.split())
        name_tokens  = set(norm_name.split())
        token_score  = len(model_tokens & name_tokens) / max(len(model_tokens | name_tokens), 1)
        combined     = 0.4 * seq_score + 0.6 * token_score
        if combined > best_score:
            best_score, best_name = combined, name
    return best_name if best_score > 0.10 else None


def _noise_hash(wtg_coords: dict, Lw_bands: dict, hub_height: float,
                resolution: int, buffer_m: float, hr: float, G: float) -> str:
    key = f"{sorted(wtg_coords.items())}{sorted(Lw_bands.items())}{hub_height}{resolution}{buffer_m}{hr}{G}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Persistent session state
# ─────────────────────────────────────────────────────────────────────────────

if 'tmp_dir' not in st.session_state:
    st.session_state.tmp_dir = tempfile.mkdtemp()
if 'noise_cache' not in st.session_state:
    st.session_state.noise_cache = {}

TMP = st.session_state.tmp_dir


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header('Settings')

    # ── PowerPoint template ───────────────────────────────────────────────
    st.subheader('PowerPoint Template')
    if DEFAULT_TPL:
        st.caption('✅ Bundled template active (`template01.pptx`)')
    tpl_file = st.file_uploader(
        'Override template (optional)',
        type=['pptx'],
        help='Upload a different .pptx to override the bundled template.',
    )

    st.divider()

    # ── Shapefile overlay ─────────────────────────────────────────────────
    st.subheader('Shapefile Overlay')
    shapes_available: dict = {}
    if not HAS_GPD:
        st.caption('ℹ  Install `geopandas` to enable shapefile overlays.')
    else:
        shp_uploads = st.file_uploader(
            'Upload shapefiles as .zip (optional)',
            type=['zip'],
            accept_multiple_files=True,
            help='Each .zip must contain .shp + .dbf + .shx. '
                 'Select which to display per calculation in the main area.',
        )
        for idx, f in enumerate(shp_uploads or []):
            tmp_path = os.path.join(TMP, f'shp_{idx}_{f.name}')
            if not os.path.exists(tmp_path):
                with open(tmp_path, 'wb') as fp:
                    fp.write(f.read())
            gdf = load_shapefile(tmp_path)
            if gdf is not None:
                label = f.name.removesuffix('.zip')
                if label in shapes_available:
                    label = f'{label} ({idx + 1})'
                shapes_available[label] = gdf
            else:
                st.warning(f'Could not read {f.name} — check it contains a valid shapefile.')
        if shapes_available:
            st.caption(f'{len(shapes_available)} shapefile(s) loaded — select per calculation below.')

    st.divider()

    # ── Noise settings ────────────────────────────────────────────────────
    st.subheader('Noise Settings')
    if not (HAS_NOISE and HAS_CTX):
        if not HAS_NOISE:
            msg = f'ℹ  Noise import failed: `{_NOISE_IMPORT_ERR}`' if _NOISE_IMPORT_ERR else 'ℹ  `wind_noise_analyser` not importable.'
            st.caption(msg)
        else:
            st.caption('ℹ  Install `contextily` to enable noise overlays.')
        noise_enabled    = False
        noise_resolution = 120
        noise_buffer_km  = 3.0
        noise_hr         = 4.0
        noise_G          = 0.5
        noise_levels     = [35.0, 40.0, 45.0]
    else:
        noise_enabled    = st.toggle('Enable noise contour overlays', value=True)
        noise_resolution = st.slider(
            'Grid resolution (pts/side)', 50, 250, 120, 10,
            help='Higher = smoother contours but slower. 120 is a good balance.',
        )
        noise_buffer_km = st.number_input(
            'Grid buffer beyond layout (km)', 0.5, 10.0, 3.0, 0.5,
        )
        noise_hr = st.number_input(
            'Receiver height (m)', 1.0, 10.0, 4.0, 0.5,
            help='SA Guidelines 2021 default: 4 m',
        )
        noise_G = st.slider(
            'Ground factor G', 0.0, 1.0, 0.5, 0.05,
            help='0 = hard (paved/water)  →  1 = soft (long grass/crops)',
        )
        _lvl_str = st.text_input('Contour levels dB(A) — comma-separated', '35, 40, 45')
        try:
            noise_levels = sorted(float(x) for x in _lvl_str.split(',') if x.strip())
        except ValueError:
            noise_levels = [35.0, 40.0, 45.0]
            st.warning('Invalid contour levels — using 35, 40, 45 dB(A).')

    st.divider()

    # ── Presentation metadata ─────────────────────────────────────────────
    st.subheader('Presentation Details')
    cover_title       = st.text_input('Report title', value='XXWF Prelim Yield Estimates')
    cover_subtitle    = st.text_input('Subtitle (heading 2)', value='Version Y')
    _today            = date.today()
    cover_subsubtitle = st.text_input(
        'Date / sub-heading',
        value=f"{_ordinal(_today.day)} {_today.strftime('%B %Y')}",
    )

    st.divider()

    # ── Loss assumptions ──────────────────────────────────────────────────
    st.subheader('Loss Assumptions')
    st.caption('Temp derating is set per-calculation below after uploading PDFs.')

    losses: dict = {}
    losses['Temp derating loss [%]'] = None

    def _loss_input(label: str, key: str) -> float:
        return st.number_input(
            label, 0.0, 20.0,
            float(DEFAULT_LOSSES.get(key, 0.0) or 0.0),
            0.1,
        )

    losses['Availability loss [%]']        = _loss_input('Availability loss [%]',        'Availability loss [%]')
    losses['Electrical loss [%]']          = _loss_input('Electrical loss [%]',           'Electrical loss [%]')
    losses['Turbine performance loss [%]'] = _loss_input('Turbine performance loss [%]',  'Turbine performance loss [%]')
    losses['Degradation [%]']             = _loss_input('Degradation [%]',                'Degradation [%]')

    combined = 1.0
    for v in losses.values():
        if v:
            combined *= 1 - v / 100
    total_loss_pct = (1 - combined) * 100
    st.metric('Combined loss (excl. temp derating)', f'{total_loss_pct:.1f}%')


# ─────────────────────────────────────────────────────────────────────────────
# Main area — PDF upload
# ─────────────────────────────────────────────────────────────────────────────

st.subheader('Upload PARK PDFs')
uploaded = st.file_uploader(
    'Select WindPRO PARK PDF exports (up to 8 files)',
    type=['pdf'],
    accept_multiple_files=True,
    help='Column order in the output matches upload order. First file is the AEP baseline.',
)

MAX_PDFS = 8
if len(uploaded) > MAX_PDFS:
    st.warning(f'Only the first {MAX_PDFS} PDFs will be used.')
    uploaded = uploaded[:MAX_PDFS]

# ─────────────────────────────────────────────────────────────────────────────
# Extract + preview
# ─────────────────────────────────────────────────────────────────────────────

if uploaded:
    st.subheader('Extracted Data Preview')

    datasets  = []
    pdf_paths = []

    with st.spinner('Parsing PDFs…'):
        for f in uploaded:
            tmp_path = os.path.join(TMP, f.name)
            if not os.path.exists(tmp_path):
                with open(tmp_path, 'wb') as fp:
                    fp.write(f.read())
            else:
                f.seek(0)
            data = extract(tmp_path)
            apply_losses(data, losses)
            datasets.append(data)
            pdf_paths.append(tmp_path)

    rows = []
    for d in datasets:
        gross  = d.get('gross_aep_mwh')
        park   = d.get('park_yield_mwh')
        p50    = d.get('p50_aep_mwh')
        coords = len(d.get('wtg_coords', {}))
        rows.append({
            'Calculation':     d.get('calc_name', '-'),
            'Date':            d.get('calc_date', '-'),
            'WTGs':            d.get('num_wtgs', '-'),
            'Capacity (MW)':   d.get('total_mw', '-'),
            'Gross AEP (GWh)': f'{gross/1000:.1f}' if gross else '-',
            'Wake Loss (%)':   f'{d["wake_loss_pct"]:.1f}%' if d.get('wake_loss_pct') else '-',
            'Park AEP (GWh)':  f'{park/1000:.1f}' if park else '-',
            'P50 AEP (GWh)':   f'{p50/1000:.1f}'  if p50  else '-',
            'Map':             f'Satellite ({coords} WTGs)' if coords >= 1 else 'No coords',
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Noise curve selection per calculation ─────────────────────────────
    noise_Lw_per_calc: list[dict | None] = [None] * len(datasets)

    if noise_enabled:
        st.subheader('Noise Spectra')
        if _WTG_PRESETS:
            st.caption(
                'Select the sound power spectrum for each calculation. '
                'The best-matching preset is auto-selected from the WTG model name.'
            )
            preset_names = list(_WTG_PRESETS.keys())
            _none_label  = '— None (skip noise overlay) —'
            noise_cols   = st.columns(min(len(datasets), 4))

            for i, d in enumerate(datasets):
                wtg_model = d.get('wtg_model', '') or ''
                matched   = _best_preset_match(wtg_model, preset_names)
                default_i = (preset_names.index(matched) + 1) if matched else 0

                with noise_cols[i % len(noise_cols)]:
                    calc_name = d.get('calc_name', f'Calc {i+1}')
                    sel = st.selectbox(
                        calc_name[:30],
                        options=[_none_label] + preset_names,
                        index=default_i,
                        key=f'noise_preset_{i}',
                        help=f'WTG model detected: {wtg_model or "unknown"}',
                    )
                    if sel != _none_label:
                        raw_data, is_third = _WTG_PRESETS[sel]
                        if is_third:
                            from wind_noise_analyser import third_oct_to_octave as _toto
                            noise_Lw_per_calc[i] = _toto(raw_data, a_weighted=True)
                        else:
                            from wind_noise_analyser import OCTAVE_BANDS as _OB
                            noise_Lw_per_calc[i] = {int(f): v for f, v in raw_data.items()
                                                     if int(f) in _OB}
        else:
            st.info(
                'No turbine presets found. Place '
                '`WTG_Acoustic_Spectra_Loudest_Modes 1.xlsx` in the app folder '
                'to enable noise curve selection.'
            )

    # ── Compute noise grids (cached in session_state) ─────────────────────
    noise_overlays: list[dict | None] = [None] * len(datasets)

    if noise_enabled and any(lw is not None for lw in noise_Lw_per_calc):
        to_compute = [
            i for i, (d, lw) in enumerate(zip(datasets, noise_Lw_per_calc))
            if lw is not None and d.get('wtg_coords')
        ]
        if to_compute:
            with st.spinner(f'Computing noise grid(s) for {len(to_compute)} calculation(s)…'):
                for i in to_compute:
                    d   = datasets[i]
                    lw  = noise_Lw_per_calc[i]
                    hub = d.get('hub_m') or 150.0
                    cache_key = _noise_hash(
                        d['wtg_coords'], lw, hub,
                        noise_resolution, noise_buffer_km * 1000,
                        noise_hr, noise_G)
                    if cache_key not in st.session_state.noise_cache:
                        result = compute_noise_overlay(
                            d['wtg_coords'], hub, lw,
                            resolution=noise_resolution,
                            buffer_m=noise_buffer_km * 1000,
                            hr=noise_hr, G=noise_G)
                        st.session_state.noise_cache[cache_key] = result
                    cached = st.session_state.noise_cache[cache_key]
                    if cached is not None:
                        noise_overlays[i] = dict(cached, contour_levels=noise_levels)

            n_done = sum(1 for o in noise_overlays if o is not None)
            if n_done:
                summary_rows = []
                for i, (d, overlay) in enumerate(zip(datasets, noise_overlays)):
                    if overlay is not None:
                        ng = overlay['noise_grid']
                        summary_rows.append({
                            'Calculation':       d.get('calc_name', f'Calc {i+1}'),
                            'Max noise (dB(A))': f'{float(ng.max()):.1f}',
                            f'≥ {noise_levels[0]:.0f} dB(A) grid pts':
                                str(int((ng >= noise_levels[0]).sum())),
                        })
                st.caption(f'Noise grids ready for {n_done} calculation(s).')
                st.dataframe(pd.DataFrame(summary_rows),
                             use_container_width=True, hide_index=True)

    # ── Per-calculation shapefile selection ───────────────────────────────
    shapes_per_calc: list | None = None
    if shapes_available:
        st.subheader('Shapefile Overlays')
        st.caption('Choose which shapefiles to show on each calculation map.')
        shapes_per_calc = []
        shp_cols = st.columns(min(len(datasets), 4))
        for i, d in enumerate(datasets):
            with shp_cols[i % len(shp_cols)]:
                sel = st.multiselect(
                    d.get('calc_name', f'Calc {i+1}')[:30],
                    options=list(shapes_available.keys()),
                    default=list(shapes_available.keys()),
                    key=f'shp_{i}',
                )
                shapes_per_calc.append([shapes_available[k] for k in sel])

    # ── Per-calculation temperature derating ──────────────────────────────
    st.subheader('Temperature Derating')
    st.caption(
        'Temp derating varies by turbine type — set each calculation separately. '
        'Leave at 0 to exclude.'
    )
    temp_derating: list[float | None] = []
    cols = st.columns(min(len(datasets), 4))
    for i, d in enumerate(datasets):
        with cols[i % len(cols)]:
            val = st.number_input(
                d.get('calc_name', f'Calc {i+1}')[:30], 0.0, 20.0, 0.0, 0.1,
                key=f'td_{i}',
                help='Temp derating loss [%] for this calculation',
            )
            temp_derating.append(val if val > 0 else None)

    # ── Generate ──────────────────────────────────────────────────────────
    st.subheader('Generate Presentation')

    col1, col2 = st.columns([1, 3])
    with col1:
        generate = st.button('Generate PowerPoint', type='primary', use_container_width=True)

    if generate:
        if tpl_file:
            tpl_path = os.path.join(TMP, tpl_file.name)
            tpl_file.seek(0)
            with open(tpl_path, 'wb') as fp:
                fp.write(tpl_file.read())
        else:
            tpl_path = DEFAULT_TPL

        losses_per_pdf = [
            dict(losses, **{'Temp derating loss [%]': td})
            for td in temp_derating
        ]

        with st.spinner('Building presentation…'):
            pptx_bytes = build(
                pdf_paths, tpl_path, losses,
                cover_title=cover_title,
                cover_subtitle=cover_subtitle,
                cover_subsubtitle=cover_subsubtitle,
                losses_per_pdf=losses_per_pdf,
                shapes_per_calc=shapes_per_calc,
                noise_overlays=noise_overlays if noise_enabled else None,
            )

        n_noise = sum(1 for o in noise_overlays if o is not None)
        st.success(
            f'Done — cover + {len(pdf_paths)} calculation slide(s) '
            f'({n_noise} with noise contours) + summary table.'
        )

        _safe_title    = re.sub(r'[^\w\s-]', '', cover_title).strip().replace(' ', '_')
        _safe_subtitle = re.sub(r'[^\w\s-]', '', cover_subtitle).strip().replace(' ', '_')
        _file_name     = f"{_today.strftime('%Y.%m.%d')}_{_safe_title}_{_safe_subtitle}.pptx"
        st.download_button(
            label='Download PowerPoint',
            data=pptx_bytes,
            file_name=_file_name,
            mime='application/vnd.openxmlformats-officedocument.presentationml.presentation',
            use_container_width=True,
        )

else:
    st.info('Upload one or more WindPRO PARK PDFs above to get started.')
