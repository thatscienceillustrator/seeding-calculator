"""\
Cell Culture Seeding Calculator\
================================\
A Streamlit app that calculates seeding densities for adherent cell lines\
based on their doubling time, target vessel, and time to confluence.\
\
Run with:  streamlit run seeding_calculator.py\
"""\
\
import streamlit as st\
import pandas as pd\
import plotly.graph_objects as go\
\
# ----------------------------------------------------------------------------\
# Reference data\
# ----------------------------------------------------------------------------\
# Doubling times (hours) for common adherent cell lines.\
# Sources: ATCC / Corning / common literature values. Tweak as needed.\
CELL_LINES = \{\
    "HeLa": 24,\
    "HEK293": 24,\
    "HEK293T": 20,\
    "MCF7": 29,\
    "A549": 22,\
    "U2OS": 29,\
    "eHAP": 14,\
    "HaCaT": 28,\
\}\
\
# Approximate cell number at 100% confluence per vessel.\
# These are rough Corning/ThermoFisher reference values.\
VESSELS = \{\
    "384-well":     12_000,\
    "96-well":      40_000,\
    "48-well":      80_000,\
    "24-well":     250_000,\
    "12-well":     500_000,\
    "6-well":    1_200_000,\
    "T25":       2_800_000,\
    "T75":       8_400_000,\
    "T175":     23_300_000,\
    "10 cm dish": 8_800_000,\
    "15 cm dish": 20_000_000,\
\}\
\
# Default timepoints (hours) to compute seeding numbers for\
TIMEPOINTS = [24, 48, 72, 96, 144]\
\
\
# ----------------------------------------------------------------------------\
# Core math\
# ----------------------------------------------------------------------------\
import math\
\
# ----------------------------------------------------------------------------\
# Core math (logistic growth model)\
# ----------------------------------------------------------------------------\
def seeding_number(\
    target_cells: float,\
    carrying_capacity: float,\
    hours: float,\
    doubling_time: float,\
    attachment_lag: float = 0.0,\
    plating_efficiency: float = 1.0,\
) -> float:\
    """\
    Logistic-growth reverse calculation.\
\
    Forward model:  N(t) = K / (1 + ((K - N0)/N0) \'b7 e^(-r\'b7t))\
    Inverted:       N0   = K \'b7 N / (N + (K - N) \'b7 e^(r\'b7t))\
\
    Where:\
        K   = carrying_capacity (cells at 100% confluence in this vessel)\
        N   = target_cells (cells we want at time t)\
        r   = ln(2) / doubling_time  (intrinsic growth rate)\
        t   = max(hours - attachment_lag, 0)  (effective growth time)\
\
    Then divide by plating_efficiency to compensate for cells that don't\
    successfully attach.\
    """\
    # Edge cases\
    if target_cells <= 0:\
        return 0.0\
    if target_cells >= carrying_capacity:\
        # Asking for \uc0\u8805  K means we need to seed \u8805  K (logistic asymptotes to K)\
        return carrying_capacity / plating_efficiency\
\
    effective_hours = max(hours - attachment_lag, 0.0)\
    r = math.log(2) / doubling_time\
    growth = math.exp(r * effective_hours)\
\
    n0 = (carrying_capacity * target_cells) / (\
        target_cells + (carrying_capacity - target_cells) * growth\
    )\
    return n0 / plating_efficiency\
\
\
def practical_floor(carrying_capacity: float, plating_efficiency: float = 1.0) -> float:\
    """\
    Minimum sensible seeding number per well, accounting for two things:\
\
    1. Poisson noise: when you sample N cells from a suspension, well-to-well\
       variation has CV = 1/sqrt(N). Below ~50 cells/well, CV climbs above\
       ~14%, and replicates start looking inconsistent regardless of what\
       the growth math says.\
\
    2. Vessel scale: 50 cells in a T75 is meaninglessly small (< 0.001% of K).\
       For larger vessels, 0.1% of K is a more sensible floor \'97 well above\
       Poisson noise, but not a fixed absolute number.\
\
    Floor = max(50, 0.001 * K), then divided by plating efficiency.\
    """\
    floor = max(50.0, 0.001 * carrying_capacity)\
    return floor / plating_efficiency\
\
\
def will_naturally_confluence(\
    seed_count: float,\
    carrying_capacity: float,\
    hours: float,\
    doubling_time: float,\
    attachment_lag: float = 0.0,\
    threshold: float = 0.95,\
) -> bool:\
    """\
    Forward simulation: given seed_count, will we hit `threshold` of K\
    by `hours`? Used to detect when the user is in the "you'll be confluent\
    anyway" regime.\
    """\
    if seed_count <= 0:\
        return False\
    effective_hours = max(hours - attachment_lag, 0.0)\
    r = math.log(2) / doubling_time\
    growth = math.exp(r * effective_hours)\
    n_t = (carrying_capacity * seed_count * growth) / (\
        carrying_capacity + seed_count * (growth - 1)\
    )\
    return n_t >= threshold * carrying_capacity\
\
\
def round_nice(n: float) -> int:\
    """Round to a sensible lab-friendly number (no one pipettes 8,317 cells)."""\
    if n < 100:\
        # Round to nearest 10\
        return int(round(n / 10) * 10)\
    elif n < 1_000:\
        # Round to nearest 50\
        return int(round(n / 50) * 50)\
    elif n < 10_000:\
        # Round to nearest 500\
        return int(round(n / 500) * 500)\
    elif n < 100_000:\
        # Round to nearest 1,000\
        return int(round(n / 1_000) * 1_000)\
    else:\
        # Round to nearest 10,000\
        return int(round(n / 10_000) * 10_000)\
\
\
def format_cells(n: int) -> str:\
    """Format with thousands separators, e.g. 12,500"""\
    return f"\{n:,\}"\
\
\
def build_growth_chart(\
    seed: float,\
    K: float,\
    doubling_time: float,\
    attachment_lag: float,\
    plating_efficiency: float,\
    timepoints: list,\
    target_confluence: float,\
    vessel_name: str,\
) -> go.Figure:\
    """\
    Plot the predicted logistic growth trajectory for a given seeding density,\
    with reference lines for K, target confluence, and the user's timepoints.\
    """\
    # Sample the curve over a sensible time window \'97 at least to the latest\
    # selected timepoint, with a bit of headroom so plateau is visible.\
    max_time = max(max(timepoints) * 1.2, 96)\
    n_points = 200\
    t_values = [max_time * i / (n_points - 1) for i in range(n_points)]\
\
    # Effective seeded cells (after plating efficiency loss)\
    n0 = seed * plating_efficiency\
    r = math.log(2) / doubling_time\
\
    # Logistic forward simulation, with attachment lag (flat curve before lag)\
    def n_at(t):\
        if t <= attachment_lag or n0 <= 0:\
            return n0\
        eff_t = t - attachment_lag\
        growth = math.exp(r * eff_t)\
        return (K * n0 * growth) / (K + n0 * (growth - 1))\
\
    cell_counts = [n_at(t) for t in t_values]\
    confluence_pct = [c / K * 100 for c in cell_counts]\
\
    # Custom hover text combining cells and % confluence\
    hover_text = [\
        f"<b>\{t:.1f\} h</b><br>\{format_cells(int(c))\} cells<br>\{p:.1f\}% confluent"\
        for t, c, p in zip(t_values, cell_counts, confluence_pct)\
    ]\
\
    fig = go.Figure()\
\
    # The growth curve itself\
    fig.add_trace(go.Scatter(\
        x=t_values,\
        y=cell_counts,\
        mode="lines",\
        line=dict(color="#2563eb", width=2.5),\
        name="Predicted growth",\
        hoverinfo="text",\
        hovertext=hover_text,\
        showlegend=False,\
    ))\
\
    # Carrying capacity (100% confluence) \'97 dashed line at top\
    fig.add_hline(\
        y=K,\
        line=dict(color="#94a3b8", width=1, dash="dash"),\
        annotation_text=f"100% (\{format_cells(int(K))\})",\
        annotation_position="top right",\
        annotation_font_size=10,\
    )\
\
    # Target confluence \'97 dashed line mid\
    target_y = K * target_confluence / 100\
    fig.add_hline(\
        y=target_y,\
        line=dict(color="#10b981", width=1, dash="dot"),\
        annotation_text=f"\{target_confluence:g\}% target",\
        annotation_position="top right",\
        annotation_font_size=10,\
        annotation_font_color="#10b981",\
    )\
\
    # Vertical lines at each requested timepoint\
    for tp in timepoints:\
        fig.add_vline(\
            x=tp,\
            line=dict(color="#64748b", width=1, dash="dot"),\
            annotation_text=f"\{tp:g\}h",\
            annotation_position="bottom right",\
            annotation_font_size=10,\
        )\
\
    # Shaded lag region\
    if attachment_lag > 0:\
        fig.add_vrect(\
            x0=0, x1=attachment_lag,\
            fillcolor="#f1f5f9", opacity=0.6,\
            line_width=0,\
            annotation_text="lag",\
            annotation_position="top left",\
            annotation_font_size=9,\
            annotation_font_color="#64748b",\
            layer="below",\
        )\
\
    fig.update_layout(\
        title=dict(\
            text=f"<b>\{vessel_name\}</b>",\
            font=dict(size=14),\
            x=0.02,\
            xanchor="left",\
        ),\
        xaxis=dict(\
            title="Time (h)",\
            showgrid=True,\
            gridcolor="#f1f5f9",\
            zeroline=False,\
        ),\
        yaxis=dict(\
            title="Cells / well",\
            type="log",\
            showgrid=True,\
            gridcolor="#f1f5f9",\
            zeroline=False,\
        ),\
        plot_bgcolor="white",\
        paper_bgcolor="white",\
        margin=dict(l=10, r=10, t=40, b=10),\
        height=340,\
        hovermode="x unified",\
    )\
\
    return fig\
\
\
def build_petri_dish_3d() -> go.Figure:\
    """\
    A small Plotly 3D figure of a petri dish.\
\
    Built from primitives:\
      - Bottom disc (white-ish)\
      - Media disc (soft pink, slightly above bottom)\
      - Cylindrical wall (translucent ring)\
      - Floating lid disc\
\
    Animated camera rotation via Plotly frames (~12s per turn).\
    """\
    import numpy as np\
\
    fig = go.Figure()\
\
    # ---- Bottom disc -----------------------------------------------------\
    n_theta = 60\
    theta = np.linspace(0, 2 * np.pi, n_theta)\
    r_outer = 1.4\
    x_disc = np.concatenate([[0], r_outer * np.cos(theta)])\
    y_disc = np.concatenate([[0], r_outer * np.sin(theta)])\
    z_disc = np.zeros_like(x_disc)\
    # Triangulate as fan from center vertex 0\
    i_idx = [0] * (n_theta - 1)\
    j_idx = list(range(1, n_theta))\
    k_idx = list(range(2, n_theta + 1))\
    fig.add_trace(go.Mesh3d(\
        x=x_disc, y=y_disc, z=z_disc,\
        i=i_idx, j=j_idx, k=k_idx,\
        color="#fafafa", opacity=0.9,\
        flatshading=True, showscale=False,\
        lighting=dict(ambient=0.7, diffuse=0.5, specular=0.2),\
        hoverinfo="skip",\
    ))\
\
    # ---- Media disc (pink) -----------------------------------------------\
    z_media = np.full_like(x_disc, 0.05)\
    fig.add_trace(go.Mesh3d(\
        x=x_disc * 0.97, y=y_disc * 0.97, z=z_media,\
        i=i_idx, j=j_idx, k=k_idx,\
        color="#fbb6c2", opacity=0.85,\
        flatshading=True, showscale=False,\
        lighting=dict(ambient=0.6, diffuse=0.6, specular=0.3),\
        hoverinfo="skip",\
    ))\
\
    # ---- Cylindrical wall (ring) -----------------------------------------\
    # Build wall as quad strip: bottom circle and top circle\
    n_wall = 60\
    th = np.linspace(0, 2 * np.pi, n_wall)\
    r_wall = 1.4\
    h_wall = 0.4\
    xb = r_wall * np.cos(th); yb = r_wall * np.sin(th); zb = np.zeros(n_wall)\
    xt = r_wall * np.cos(th); yt = r_wall * np.sin(th); zt = np.full(n_wall, h_wall)\
    xw = np.concatenate([xb, xt]); yw = np.concatenate([yb, yt]); zw = np.concatenate([zb, zt])\
    iw, jw, kw = [], [], []\
    for m in range(n_wall - 1):\
        # Two triangles per quad: (m, m+1, m+n_wall) and (m+1, m+1+n_wall, m+n_wall)\
        iw += [m, m + 1]\
        jw += [m + 1, m + 1 + n_wall]\
        kw += [m + n_wall, m + n_wall]\
    fig.add_trace(go.Mesh3d(\
        x=xw, y=yw, z=zw, i=iw, j=jw, k=kw,\
        color="#e7d8da", opacity=0.35,\
        flatshading=True, showscale=False,\
        lighting=dict(ambient=0.7, diffuse=0.4, specular=0.4),\
        hoverinfo="skip",\
    ))\
\
    # ---- Lid disc (floating slightly above) ------------------------------\
    z_lid = np.full_like(x_disc, h_wall + 0.08)\
    fig.add_trace(go.Mesh3d(\
        x=x_disc * 1.03, y=y_disc * 1.03, z=z_lid,\
        i=i_idx, j=j_idx, k=k_idx,\
        color="#ffffff", opacity=0.5,\
        flatshading=True, showscale=False,\
        lighting=dict(ambient=0.8, diffuse=0.3, specular=0.5),\
        hoverinfo="skip",\
    ))\
\
    fig.update_layout(\
        scene=dict(\
            xaxis=dict(visible=False, showgrid=False, showbackground=False),\
            yaxis=dict(visible=False, showgrid=False, showbackground=False),\
            zaxis=dict(visible=False, showgrid=False, showbackground=False),\
            bgcolor="rgba(0,0,0,0)",\
            aspectmode="manual",\
            aspectratio=dict(x=1, y=1, z=0.5),\
            camera=dict(\
                eye=dict(x=2.4, y=1.3, z=1.4),\
                center=dict(x=0, y=0, z=0.1),\
                up=dict(x=0, y=0, z=1),\
            ),\
            # Disable zoom/pan \'97 only allow rotation (cleaner UX for an aesthetic widget)\
            dragmode="orbit",\
        ),\
        margin=dict(l=0, r=0, t=0, b=0),\
        height=180,\
        paper_bgcolor="rgba(0,0,0,0)",\
        showlegend=False,\
    )\
    return fig\
\
\
def build_petri_dish_figure() -> go.Figure:\
    """\
    Build a Plotly 3D figure of a petri dish: body cylinder, bottom disc,\
    pinkish media inside, floating lid. Animated via frames for gentle\
    rotation + a bit of vertical bob.\
\
    Note: Plotly 3D is flat-shaded scientific styling \'97 clean and minimal,\
    not glassy. Don't expect photorealism.\
    """\
    import numpy as np\
\
    # Geometry parameters\
    r_body = 1.0\
    body_height = 0.25\
    r_lid = 1.05\
    lid_height = 0.04\
    r_media = 0.95\
    media_height = 0.13\
    media_y_offset = -0.06  # sits inside the dish\
\
    n_theta = 48  # circular resolution\
    theta = np.linspace(0, 2 * np.pi, n_theta)\
\
    # --- Body cylinder walls (Surface) ---\
    body_x = r_body * np.cos(theta)\
    body_z = r_body * np.sin(theta)  # using z as horizontal because Plotly y is "up"\
    body_y_top = body_height / 2\
    body_y_bot = -body_height / 2\
    Body_X = np.array([body_x, body_x])\
    Body_Y = np.array([[body_y_bot] * n_theta, [body_y_top] * n_theta])\
    Body_Z = np.array([body_z, body_z])\
\
    body_walls = go.Surface(\
        x=Body_X, y=Body_Y, z=Body_Z,\
        colorscale=[[0, "#e8edf3"], [1, "#cfd8e3"]],\
        showscale=False,\
        opacity=0.55,\
        hoverinfo="skip",\
        lighting=dict(ambient=0.7, diffuse=0.4, specular=0.2),\
    )\
\
    # --- Body bottom disc (Mesh3d, flat) ---\
    bx = np.concatenate([[0], r_body * np.cos(theta)])\
    bz = np.concatenate([[0], r_body * np.sin(theta)])\
    by = np.full(n_theta + 1, body_y_bot)\
    # Triangles fan out from center vertex 0\
    i_idx = [0] * n_theta\
    j_idx = list(range(1, n_theta + 1))\
    k_idx = [j + 1 if j < n_theta else 1 for j in range(1, n_theta + 1)]\
    body_bottom = go.Mesh3d(\
        x=bx, y=by, z=bz,\
        i=i_idx, j=j_idx, k=k_idx,\
        color="#dde4ec",\
        opacity=0.7,\
        hoverinfo="skip",\
        flatshading=True,\
    )\
\
    # --- Media disc (a low cylinder shown as top + bottom + side) ---\
    media_y_top = media_y_offset + media_height / 2\
    media_y_bot = media_y_offset - media_height / 2\
\
    # Media side wall\
    mx = r_media * np.cos(theta)\
    mz = r_media * np.sin(theta)\
    Media_X = np.array([mx, mx])\
    Media_Y = np.array([[media_y_bot] * n_theta, [media_y_top] * n_theta])\
    Media_Z = np.array([mz, mz])\
    media_side = go.Surface(\
        x=Media_X, y=Media_Y, z=Media_Z,\
        colorscale=[[0, "#f4a4b3"], [1, "#fbb6c2"]],\
        showscale=False,\
        opacity=0.85,\
        hoverinfo="skip",\
        lighting=dict(ambient=0.7, diffuse=0.5),\
    )\
\
    # Media top (visible pink disc)\
    mtx = np.concatenate([[0], r_media * np.cos(theta)])\
    mtz = np.concatenate([[0], r_media * np.sin(theta)])\
    mty = np.full(n_theta + 1, media_y_top)\
    media_top = go.Mesh3d(\
        x=mtx, y=mty, z=mtz,\
        i=i_idx, j=j_idx, k=k_idx,\
        color="#fbb6c2",\
        opacity=0.95,\
        hoverinfo="skip",\
        flatshading=True,\
    )\
\
    # --- Lid (slightly bigger disc floating above) ---\
    lid_y_top = body_y_top + 0.18\
    lid_y_bot = lid_y_top - lid_height\
\
    # Lid side\
    lx = r_lid * np.cos(theta)\
    lz = r_lid * np.sin(theta)\
    Lid_X = np.array([lx, lx])\
    Lid_Y = np.array([[lid_y_bot] * n_theta, [lid_y_top] * n_theta])\
    Lid_Z = np.array([lz, lz])\
    lid_side = go.Surface(\
        x=Lid_X, y=Lid_Y, z=Lid_Z,\
        colorscale=[[0, "#e8edf3"], [1, "#cfd8e3"]],\
        showscale=False,\
        opacity=0.5,\
        hoverinfo="skip",\
        lighting=dict(ambient=0.7, diffuse=0.4, specular=0.3),\
    )\
\
    # Lid top\
    ltx = np.concatenate([[0], r_lid * np.cos(theta)])\
    ltz = np.concatenate([[0], r_lid * np.sin(theta)])\
    lty = np.full(n_theta + 1, lid_y_top)\
    lid_top = go.Mesh3d(\
        x=ltx, y=lty, z=ltz,\
        i=i_idx, j=j_idx, k=k_idx,\
        color="#dde4ec",\
        opacity=0.6,\
        hoverinfo="skip",\
        flatshading=True,\
    )\
\
    fig = go.Figure(data=[body_walls, body_bottom, media_side, media_top, lid_side, lid_top])\
\
    fig.update_layout(\
        scene=dict(\
            xaxis=dict(visible=False, showgrid=False, range=[-1.5, 1.5]),\
            yaxis=dict(visible=False, showgrid=False, range=[-1.0, 1.0]),\
            zaxis=dict(visible=False, showgrid=False, range=[-1.5, 1.5]),\
            bgcolor="rgba(0,0,0,0)",\
            aspectmode="manual",\
            aspectratio=dict(x=1.5, y=1, z=1.5),\
            # Starting camera angle \'97 slight tilt so depth is obvious immediately\
            camera=dict(\
                eye=dict(x=1.8, y=1.1, z=1.6),\
                center=dict(x=0, y=0, z=0),\
            ),\
            # Disable axes-clicking interactions but keep camera drag\
            dragmode="orbit",\
        ),\
        paper_bgcolor="rgba(0,0,0,0)",\
        plot_bgcolor="rgba(0,0,0,0)",\
        margin=dict(l=0, r=0, t=0, b=0),\
        height=240,\
        showlegend=False,\
    )\
\
    return fig\
\
\
# ----------------------------------------------------------------------------\
# Streamlit UI\
# ----------------------------------------------------------------------------\
st.set_page_config(\
    page_title="Seeding Calculator",\
    page_icon="\uc0\u55358 \u56811 ",\
    layout="wide",\
)\
\
# A bit of CSS to keep things minimal and lab-tool-like\
st.markdown(\
    """\
    <style>\
        /* Gradient background \'97 off-white to soft blush pink */\
        [data-testid="stAppViewContainer"] \{\
            background: linear-gradient(135deg, #fefdfb 0%, #fdf4f1 45%, #fde7ec 100%);\
        \}\
        /* Make the top header bar transparent so the gradient shows through */\
        [data-testid="stHeader"] \{\
            background: transparent;\
        \}\
\
        /* Layout / typography */\
        .block-container \{ max-width: 1100px; padding-top: 2.5rem; \}\
        h1 \{ font-weight: 600; letter-spacing: -0.02em; \}\
\
        /* Keep the dataframe readable on a tinted background */\
        [data-testid="stDataFrame"] \{\
            background: white;\
            border-radius: 8px;\
            padding: 4px;\
            font-variant-numeric: tabular-nums;\
        \}\
\
        /* Plotly charts sit on white cards so they don't fight the gradient */\
        [data-testid="stPlotlyChart"] \{\
            background: white;\
            border-radius: 8px;\
            padding: 8px;\
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);\
        \}\
\
        /* Metric card (single-result mode) \'97 make it a soft white card */\
        [data-testid="stMetric"] \{\
            background: white;\
            padding: 16px;\
            border-radius: 8px;\
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);\
        \}\
        [data-testid="stMetricValue"] \{ font-size: 1.4rem; \}\
    </style>\
    """,\
    unsafe_allow_html=True,\
)\
\
title_col, dish_col = st.columns([3, 1])\
with title_col:\
    st.title("\uc0\u55358 \u56811  Cell Culture Seeding Calculator")\
    st.caption(\
        "Enter a doubling time and target confluence \'97 get seeding numbers "\
        "for common vessels and timepoints."\
    )\
with dish_col:\
    petri_fig = build_petri_dish_figure()\
    st.plotly_chart(\
        petri_fig,\
        use_container_width=True,\
        config=dict(displayModeBar=False),\
    )\
    st.caption("<div style='text-align:center; font-size:0.75rem; color:#94a3b8;'>drag to spin</div>", unsafe_allow_html=True)\
\
# ---- Inputs --------------------------------------------------------------\
col1, col2, col3 = st.columns([2, 1, 1])\
\
with col1:\
    cell_choice = st.selectbox(\
        "Cell line",\
        options=list(CELL_LINES.keys()) + ["Custom..."],\
        index=0,\
        help="Pick a preset or choose 'Custom...' to enter your own doubling time.",\
    )\
\
with col2:\
    if cell_choice == "Custom...":\
        doubling_time = st.number_input(\
            "Doubling time (h)",\
            min_value=1.0,\
            max_value=200.0,\
            value=24.0,\
            step=0.5,\
        )\
    else:\
        # Show preset value but let user override it\
        default_dt = CELL_LINES[cell_choice]\
        doubling_time = st.number_input(\
            "Doubling time (h)",\
            min_value=1.0,\
            max_value=200.0,\
            value=float(default_dt),\
            step=0.5,\
            help=f"Default for \{cell_choice\} is \{default_dt\}h. You can override.",\
        )\
\
with col3:\
    target_confluence = st.slider(\
        "Target confluence (%)",\
        min_value=10,\
        max_value=100,\
        value=80,\
        step=5,\
        help="Most experiments want 70\'9690% confluence, not 100%.",\
    )\
\
# Advanced parameters \'97 collapsed by default, sensible values applied either way\
with st.expander("Advanced: attachment lag & plating efficiency", expanded=False):\
    adv_col1, adv_col2 = st.columns(2)\
    with adv_col1:\
        attachment_lag = st.number_input(\
            "Attachment lag (h)",\
            min_value=0.0,\
            max_value=48.0,\
            value=8.0,\
            step=1.0,\
            help=(\
                "Hours after seeding before cells start dividing. "\
                "Typical: 6\'9612 h for adherent lines. Set to 0 to ignore."\
            ),\
        )\
    with adv_col2:\
        plating_efficiency_pct = st.slider(\
            "Plating efficiency (%)",\
            min_value=10,\
            max_value=100,\
            value=85,\
            step=5,\
            help=(\
                "Fraction of seeded cells that successfully attach and grow. "\
                "Healthy adherent lines: 80\'9695%. Stressed/low-passage: lower."\
            ),\
        )\
    plating_efficiency = plating_efficiency_pct / 100.0\
\
st.divider()\
\
# ---- Vessel & timepoint selection ---------------------------------------\
st.subheader("Choose what you're seeding")\
\
sel_col1, sel_col2 = st.columns(2)\
\
with sel_col1:\
    selected_vessels = st.multiselect(\
        "Vessel(s)",\
        options=list(VESSELS.keys()),\
        default=["96-well"],\
        help="Pick one or more vessels.",\
    )\
\
with sel_col2:\
    selected_timepoints = st.multiselect(\
        "Time to confluence (h)",\
        options=TIMEPOINTS,\
        default=[24, 48],\
        help="Pick one or more standard timepoints, or add a custom one below.",\
    )\
\
# Optional custom timepoint\
custom_hours = st.number_input(\
    "Custom timepoint (h) \'97 optional",\
    min_value=0.0,\
    max_value=336.0,  # two weeks, generous\
    value=0.0,\
    step=1.0,\
    help="Leave at 0 to skip. Useful for non-standard timings like 36h or 60h.",\
)\
\
# Combine the standard + custom timepoints (deduplicated, sorted)\
timepoints_to_use = sorted(set(selected_timepoints) | (\{custom_hours\} if custom_hours > 0 else set()))\
\
st.divider()\
\
# ---- Calculation & display ----------------------------------------------\
if not selected_vessels:\
    st.info("Pick at least one vessel above to see seeding numbers.")\
elif not timepoints_to_use:\
    st.info("Pick at least one timepoint above to see seeding numbers.")\
else:\
    confluence_fraction = target_confluence / 100.0\
\
    # Warn if any selected timepoint is shorter than the attachment lag\
    too_short = [t for t in timepoints_to_use if t <= attachment_lag]\
    if too_short:\
        st.warning(\
            f"Timepoint(s) \{', '.join(f'\{t:g\}h' for t in too_short)\} are at or "\
            f"below the attachment lag (\{attachment_lag:g\}h). Cells won't have "\
            f"divided yet \'97 seeding number = target cells / plating efficiency."\
        )\
\
    # Track which (vessel, timepoint) cells are in the "naturally confluent" regime\
    natural_confluence_notes = []\
\
    # Build a DataFrame: rows = selected vessels, columns = selected timepoints\
    data = \{\}\
    for hours in timepoints_to_use:\
        col_name = f"\{hours:g\}h"\
        data[col_name] = []\
        for vessel_name in selected_vessels:\
            K = VESSELS[vessel_name]\
            target = K * confluence_fraction\
\
            seeded_raw = seeding_number(\
                target_cells=target,\
                carrying_capacity=K,\
                hours=hours,\
                doubling_time=doubling_time,\
                attachment_lag=attachment_lag,\
                plating_efficiency=plating_efficiency,\
            )\
\
            floor = practical_floor(K, plating_efficiency)\
\
            if seeded_raw < floor:\
                # Math says less than the practical minimum \'97 show floor + flag\
                data[col_name].append(f"\uc0\u8805 \{format_cells(round_nice(floor))\}*")\
                natural_confluence_notes.append((vessel_name, hours))\
            else:\
                data[col_name].append(format_cells(round_nice(seeded_raw)))\
\
    df = pd.DataFrame(data, index=selected_vessels)\
    df.index.name = "Vessel"\
\
    st.subheader(f"Seed at \{target_confluence\}% confluence target")\
    st.caption(\
        f"Doubling time: **\{doubling_time:g\} h**  \'95  "\
        f"Attachment lag: **\{attachment_lag:g\} h**  \'95  "\
        f"Plating efficiency: **\{plating_efficiency_pct\}%**"\
    )\
\
    # Two-column layout: numbers on the left, growth chart on the right\
    left_col, right_col = st.columns([55, 45])\
\
    # ---- Left column: numbers (table or single metric) -----------------\
    with left_col:\
        if len(selected_vessels) == 1 and len(timepoints_to_use) == 1:\
            vessel = selected_vessels[0]\
            hours = timepoints_to_use[0]\
            K = VESSELS[vessel]\
            target = K * confluence_fraction\
            seeded_raw = seeding_number(\
                target_cells=target,\
                carrying_capacity=K,\
                hours=hours,\
                doubling_time=doubling_time,\
                attachment_lag=attachment_lag,\
                plating_efficiency=plating_efficiency,\
            )\
            floor = practical_floor(K, plating_efficiency)\
\
            if seeded_raw < floor:\
                st.metric(\
                    label=f"\{vessel\}, \{hours:g\}h to \{target_confluence\}% confluence",\
                    value=f"\uc0\u8805  \{format_cells(round_nice(floor))\} cells / well",\
                )\
                st.info(\
                    f"Logistic math says **\{round_nice(seeded_raw):,\} cells** is "\
                    f"enough to reach your target \'97 but at that density, well-to-well "\
                    f"Poisson noise (CV \uc0\u8776  \{1/math.sqrt(max(seeded_raw,1))*100:.0f\}%) "\
                    f"will dominate your replicates. The number shown is a sensible "\
                    f"practical minimum. Seeding more just gets you to confluence sooner."\
                )\
            else:\
                st.metric(\
                    label=f"\{vessel\}, \{hours:g\}h to \{target_confluence\}% confluence",\
                    value=f"\{format_cells(round_nice(seeded_raw))\} cells / well",\
                )\
        else:\
            st.dataframe(df, use_container_width=True)\
            if natural_confluence_notes:\
                st.info(\
                    "**\\\\*** = the growth math alone would suggest a smaller number, "\
                    "but at very low seeding densities Poisson noise (well-to-well "\
                    "variation) dominates. The value shown is a practical minimum "\
                    "(max of 50 cells or 0.1% of confluent cell number). Seeding "\
                    "more just gets you to confluence sooner."\
                )\
\
    # ---- Right column: single growth chart ------------------------------\
    with right_col:\
        # If multiple vessels, let the user pick which to chart\
        if len(selected_vessels) > 1:\
            chart_vessel = st.selectbox(\
                "Show growth curve for",\
                options=selected_vessels,\
                index=0,\
                key="chart_vessel_picker",\
            )\
        else:\
            chart_vessel = selected_vessels[0]\
\
        # Compute seeding number for the longest timepoint, for this vessel\
        K = VESSELS[chart_vessel]\
        target = K * confluence_fraction\
        longest = max(timepoints_to_use)\
        seeded_raw = seeding_number(\
            target_cells=target,\
            carrying_capacity=K,\
            hours=longest,\
            doubling_time=doubling_time,\
            attachment_lag=attachment_lag,\
            plating_efficiency=plating_efficiency,\
        )\
        floor = practical_floor(K, plating_efficiency)\
        seed_for_chart = max(seeded_raw, floor)\
\
        fig = build_growth_chart(\
            seed=seed_for_chart,\
            K=K,\
            doubling_time=doubling_time,\
            attachment_lag=attachment_lag,\
            plating_efficiency=plating_efficiency,\
            timepoints=timepoints_to_use,\
            target_confluence=target_confluence,\
            vessel_name=f"\{chart_vessel\} \'97 seed \{format_cells(round_nice(seed_for_chart))\} cells",\
        )\
        st.plotly_chart(fig, use_container_width=True)\
        st.caption(\
            f"Curve based on seeding for **\{longest:g\}h** target. Where it "\
            f"crosses earlier timepoints, you can read off intermediate confluence."\
        )\
\
# ---- Footer / help -------------------------------------------------------\
with st.expander("How this works"):\
    st.markdown(\
        """\
        **The math: logistic growth**\
\
        Real cells don't grow exponentially forever \'97 they slow down and\
        plateau when they hit confluence (contact inhibition). This app uses\
        the logistic-growth model:\
\
        ```\
        N(t) = K / (1 + ((K - N\uc0\u8320 )/N\u8320 ) \'b7 e^(-r\'b7t))\
        ```\
\
        - `K` \'97 carrying capacity (cells at 100% confluence in this vessel)\
        - `r = ln(2) / doubling_time` \'97 intrinsic growth rate\
        - `t = max(hours - attachment_lag, 0)` \'97 effective growth time\
        - `N\uc0\u8320 ` \'97 what we want to find\
\
        Inverted to solve for the seeding number:\
\
        ```\
        N\uc0\u8320  = K \'b7 N / (N + (K - N) \'b7 e^(r\'b7t))\
        ```\
\
        Then we divide by plating efficiency to compensate for cells that\
        don't successfully attach.\
\
        **The practical floor**\
\
        When the growth math suggests seeding very few cells (e.g. 30 in a\
        96-well), Poisson sampling noise dominates: well-to-well CV =\
        1/\uc0\u8730 N. Below ~50 cells/well, CV climbs above 14% and replicates\
        get inconsistent regardless of the math.\
\
        The app uses `floor = max(50, 0.001 \'d7 K) / pe` and shows `\uc0\u8805 floor*`\
        whenever the math is below it.\
\
        **Why short and long timepoints behave differently**\
\
        - **Short timepoints**: cells are far from confluence, logistic \uc0\u8776 \
          exponential. The math gives a sensible target-driven number.\
        - **Long timepoints**: cells reach confluence regardless of seeding\
          density. The math number drops below the practical floor and the\
          app shows `\uc0\u8805 floor*` instead.\
\
        **Reference values used**\
\
        | Vessel | ~Cells at 100% confluence (K) |\
        |---|---|\
        """\
        + "\\n".join(f"| \{v\} | \{format_cells(c)\} |" for v, c in VESSELS.items())\
        + """\
\
        **Caveats**\
\
        - Reference K values are rough Corning/ThermoFisher figures; your\
          lab may differ.\
        - Doubling time is assumed constant in the log phase. The logistic\
          model captures the slowdown near confluence.\
        - Suspension cells aren't supported in this version.\
        """\
    )\
\
st.caption("v1 \'b7 Made for the bench.")\
}
