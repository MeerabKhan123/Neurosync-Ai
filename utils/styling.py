
"""
NeuroSync AI - Styling Utilities
-------------------------------------
White + Purple premium theme. Sidebar navigation uses native Streamlit
buttons (not a third-party component) so it always renders reliably.
"""

import streamlit as st

THEME_CSS = """
<style>
:root {
    --bg-primary: #FFFFFF;
    --bg-panel: #F8F6FC;
    --purple-deep: #3B1878;
    --purple-primary: #7C3AED;
    --purple-mid: #9D6BF0;
    --purple-light: #C4B0F5;
    --lilac: #F1ECFB;
    --text-primary: #1F1147;
    --text-muted: #6E6089;
    --border-soft: rgba(124, 58, 237, 0.18);
    --danger: #D0286B;
}

html, body, [class*="css"] {
    font-family: 'Segoe UI', 'Inter', system-ui, sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #FFFFFF 0%, #FBFAFE 100%);
    color: var(--text-primary);
}

/* ---------------- Sidebar ---------------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F5F1FC 100%);
    border-right: 1px solid var(--border-soft);
}
section[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    color: var(--text-primary);
    border: 1px solid transparent;
    border-radius: 10px;
    text-align: left;
    font-weight: 500;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.15rem;
    box-shadow: none;
    justify-content: flex-start;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--lilac);
    border-color: var(--border-soft);
    color: var(--purple-deep);
    transform: none;
    box-shadow: none;
}
/* active/primary nav button */
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(90deg, var(--purple-primary), var(--purple-mid));
    color: #FFFFFF;
    font-weight: 600;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    transform: none;
}

/* ---------------- Cards ---------------- */
.glass-card {
    background: #FFFFFF;
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 6px 24px rgba(124, 58, 237, 0.08);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    margin-bottom: 1rem;
}
.glass-card:hover {
    box-shadow: 0 10px 30px rgba(124, 58, 237, 0.16);
    transform: translateY(-2px);
}

.kpi-value {
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--purple-deep), var(--purple-primary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-label {
    color: var(--text-muted);
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--purple-deep);
    border-left: 4px solid var(--purple-primary);
    padding-left: 0.7rem;
    margin: 1.2rem 0 0.8rem 0;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: var(--purple-deep);
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}

.badge {
    display: inline-block;
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.badge-low { background: var(--lilac); color: var(--purple-primary); border: 1px solid var(--purple-light); }
.badge-medium { background: #EAD9FF; color: var(--purple-deep); border: 1px solid var(--purple-mid); }
.badge-high { background: #FDE3EE; color: var(--danger); border: 1px solid #F3A6C6; }

div[data-testid="stMetric"] {
    background: var(--bg-panel);
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: 0.8rem 1rem;
}
div[data-testid="stMetric"] label { color: var(--text-muted) !important; }

/* Main-area buttons (forms, predict, download etc.) */
div.block-container .stButton > button,
div.block-container .stDownloadButton > button,
div.block-container .stFormSubmitButton > button {
    background: linear-gradient(90deg, var(--purple-primary), var(--purple-deep));
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    font-weight: 600;
}
div.block-container .stButton > button:hover,
div.block-container .stDownloadButton > button:hover,
div.block-container .stFormSubmitButton > button:hover {
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.35);
    transform: translateY(-1px);
}

hr { border-color: var(--border-soft); }

#MainMenu, footer {visibility: hidden;}

/* Keep header visible (it contains the sidebar collapse/expand arrow),
   but hide the toolbar bits inside it (deploy button, menu, status). */
header[data-testid="stHeader"] {
    visibility: visible !important;
    display: block !important;
    background: transparent;
}
header[data-testid="stHeader"] [data-testid="stToolbar"] {
    visibility: hidden;
}
/* Force the sidebar's open/close arrow to always stay visible & clickable,
   across different Streamlit versions (testids changed in 1.38+). */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarContent"] button[kind="header"],
header button[kind="header"],
button[kind="headerNoPadding"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}
</style>
"""

SPLASH_HTML = """
<div style="
    height: 92vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center; text-align:center;">
    <div style="
        width: 92px; height: 92px; border-radius: 24px; margin-bottom: 1.4rem;
        background: linear-gradient(135deg, #7C3AED, #C4B0F5);
        display:flex; align-items:center; justify-content:center;
        box-shadow: 0 0 50px rgba(124,58,237,0.35);
        animation: pulse 1.8s ease-in-out infinite;">
        <span style="font-size: 2.2rem; font-weight: 800; color: white;">N</span>
    </div>
    <div style="font-size: 2.2rem; font-weight: 800; color: #1F1147; letter-spacing: 0.03em;">
        NeuroSync <span style="color: #7C3AED;">AI</span>
    </div>
    <div style="color: #6E6089; margin-top: 0.4rem; font-size: 0.95rem;">
        Intelligent Lifestyle, Burnout &amp; Productivity Analytics
    </div>
    <div style="margin-top: 2rem; width: 160px; height: 3px; background: #F1ECFB;
        border-radius: 3px; overflow: hidden;">
        <div style="height: 100%; width: 40%; border-radius:3px;
            background: linear-gradient(90deg, #7C3AED, #C4B0F5);
            animation: loadbar 1.4s ease-in-out infinite;"></div>
    </div>
</div>
<style>
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.06); }
}
@keyframes loadbar {
    0% { transform: translateX(-160px); }
    100% { transform: translateX(400px); }
}
</style>
"""


def inject_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def glass_card_open():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)


def glass_card_close():
    st.markdown('</div>', unsafe_allow_html=True)


def risk_badge(label):
    css_class = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}.get(label, "badge-medium")
    return f'<span class="badge {css_class}">{label} Risk</span>'


NAV_PAGES = [
    "Home", "Dashboard Overview", "Dataset Viewer",
    "Exploratory Data Analysis", "Model Comparison",
    "Prediction", "Batch CSV Prediction",
    "AI Wellness Assistant", "About Project",
]


def sidebar_nav():
    """Native, reliable sidebar navigation (replaces streamlit-option-menu)."""
    st.sidebar.markdown(
        '<div style="text-align:center; padding: 0.5rem 0 1rem 0;">'
        '<div class="hero-title" style="font-size:1.5rem;">NeuroSync AI</div>'
        '<div style="color:#6E6089; font-size:0.75rem;">Analytics Platform</div>'
        '</div>', unsafe_allow_html=True,
    )

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    for option in NAV_PAGES:
        is_active = st.session_state.page == option
        if st.sidebar.button(
            option, key=f"nav_{option}", use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = option

    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    return st.session_state.page