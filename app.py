"""
WindPRO PARK Results Summary — Streamlit app

Upload up to 8 WindPRO PARK PDF exports (and optionally a branded
PowerPoint template) to generate a multi-slide summary presentation.
"""

import os
import re
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from park_summary import (DEFAULT_LOSSES, HAS_CTX, HAS_GPD,
                          _ordinal, apply_losses, build, extract, load_shapefile)


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

    st.title('WindPRO PARK Results Summary')
    st.text_input('Password', type='password', key='_pw_input', on_change=_submit)
    if '_authed' in st.session_state and not st.session_state['_authed']:
        st.error('Incorrect password — try again.')
    st.stop()


_check_password()

# Bundled template shipped with the repo
_BUNDLED_TPL = Path(__file__).parent / 'template01.pptx'
DEFAULT_TPL   = str(_BUNDLED_TPL) if _BUNDLED_TPL.exists() else None

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='WindPRO PARK Summary',
    page_icon='💨',
    layout='wide',
)

st.title('WindPRO PARK Results Summary')
st.caption(
    'Upload up to 8 WindPRO PARK PDF exports to generate a branded PowerPoint '
    'comparison presentation. The first PDF is used as the AEP baseline.'
)

if HAS_CTX:
    st.caption('🛰  Satellite basemap enabled (contextily + ESRI World Imagery)')
else:
    st.caption('ℹ  Install `contextily` to enable satellite maps: `pip install contextily pyproj`')

# ─────────────────────────────────────────────────────────────────────────────
# Persistent temp directory for this session
# ─────────────────────────────────────────────────────────────────────────────

if 'tmp_dir' not in st.session_state:
    st.session_state.tmp_dir = tempfile.mkdtemp()

TMP = st.session_state.tmp_dir


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — template + presentation settings + loss assumptions
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
                 'A .prj file is recommended so the CRS is known. '
                 'Select which to display per calculation in the main area.',
        )
        for f in (shp_uploads or []):
            tmp_path = os.path.join(TMP, f'shp_{f.name}')
            if not os.path.exists(tmp_path):
                with open(tmp_path, 'wb') as fp:
                    fp.write(f.read())
            gdf = load_shapefile(tmp_path)
            if gdf is not None:
                label = f.name.removesuffix('.zip')
                shapes_available[label] = gdf
            else:
                st.warning(f'Could not read {f.name} — check it contains a valid shapefile.')

        if shapes_available:
            st.caption(f'{len(shapes_available)} shapefile(s) loaded — select per calculation below.')

    st.divider()

    # ── Presentation metadata ─────────────────────────────────────────────
    st.subheader('Presentation Details')
    cover_title = st.text_input(
        'Report title',
        value='XXWF Prelim Yield Estimates',
    )
    cover_subtitle = st.text_input(
        'Subtitle (heading 2)',
        value='Version Y',
    )
    _today = date.today()
    cover_subsubtitle = st.text_input(
        'Date / sub-heading',
        value=f"{_ordinal(_today.day)} {_today.strftime('%B %Y')}",
    )

    st.divider()

    # ── Loss assumptions (common to all calculations) ─────────────────────
    st.subheader('Loss Assumptions')
    st.caption('Temp derating is set per-calculation below after uploading PDFs.')

    losses: dict = {}
    losses['Temp derating loss [%]'] = None  # set per-calculation in main area

    def _loss_input(label: str, key: str) -> float:
        return st.number_input(
            label, 0.0, 20.0,
            float(DEFAULT_LOSSES.get(key, 0.0) or 0.0),
            0.1,
        )

    losses['Availability loss [%]']            = _loss_input('Availability loss [%]',            'Availability loss [%]')
    losses['Electrical loss [%]']              = _loss_input('Electrical loss [%]',               'Electrical loss [%]')
    losses['Turbine performance loss [%]']     = _loss_input('Turbine performance loss [%]',      'Turbine performance loss [%]')
    losses['Degradation [%]']                  = _loss_input('Degradation [%]',                   'Degradation [%]')

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

    # Preview table
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

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    # ── Per-calculation shapefile selection ──────────────────────────────
    shapes_per_calc: list | None = None
    if shapes_available:
        st.subheader('Shapefile Overlays')
        st.caption('Choose which shapefiles to show on each calculation map.')
        shapes_per_calc = []
        shp_cols = st.columns(min(len(datasets), 4))
        for i, d in enumerate(datasets):
            name = d.get('calc_name', f'Calc {i+1}')
            with shp_cols[i % len(shp_cols)]:
                sel = st.multiselect(
                    name[:30],
                    options=list(shapes_available.keys()),
                    default=list(shapes_available.keys()),
                    key=f'shp_{i}',
                )
                shapes_per_calc.append([shapes_available[k] for k in sel])

    # ── Per-calculation temperature derating ─────────────────────────────
    st.subheader('Temperature Derating')
    st.caption(
        'Temp derating varies by turbine type — set each calculation separately. '
        'Leave at 0 to exclude.'
    )
    temp_derating: list[float | None] = []
    cols = st.columns(min(len(datasets), 4))
    for i, d in enumerate(datasets):
        name = d.get('calc_name', f'Calc {i+1}')
        with cols[i % len(cols)]:
            val = st.number_input(
                name[:30], 0.0, 20.0, 0.0, 0.1,
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
        # User upload takes priority; fall back to bundled template01.pptx
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
            )

        st.success(f'Done — cover + {len(pdf_paths)} calculation slide(s) + summary table.')
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
