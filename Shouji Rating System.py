# streamlit_app.py
import streamlit as st
import numpy as np

# ───────────────────────────────────────────────
# Page config & custom styling
# ───────────────────────────────────────────────
st.set_page_config(page_title="Shouji Rating System", layout="wide")

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&display=swap" rel="stylesheet">
    <style>
    :root {
        --main:   #6464ff;     /* text, borders, accents */
        --bg:     #000064;     /* backgrounds only */
        --shadow: #e10f0f;     /* shadows + text-glow */
        --input-bg: #0a0a4a;   /* inputs & card backgrounds */
    }
    .stApp {
        background-color: var(--bg) !important;
        font-family: 'Orbitron', sans-serif !important;
    }
    * {
        color: var(--main) !important;
    }
    h1, h2, h3, h4, h5, h6,
    .stSubheader, label, .stRadio > div > label {
        text-shadow: 0 0 8px var(--shadow),
                     0 0 4px var(--shadow),
                     1px 1px 3px rgba(0,0,0,0.7) !important;
    }
    .stMetric > div > div > div,
    .stMetric label {
        text-shadow: 0 0 7px var(--shadow),
                     0 0 3px var(--shadow) !important;
    }
    .stButton > button {
        background-color: var(--input-bg) !important;
        color: var(--main) !important;
        border: 1px solid var(--main) !important;
        box-shadow: 0 3px 10px var(--shadow),
                    inset 0 1px 4px rgba(225,15,15,0.25) !important;
        text-shadow: 0 0 8px var(--shadow),
                     1px 1px 3px rgba(0,0,0,0.8) !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 600;
        transition: all 0.25s ease;
    }
    .stButton > button:hover {
        background-color: #101070 !important;
        border-color: #9a9aff !important;
        box-shadow: 0 6px 18px var(--shadow),
                    inset 0 1px 6px rgba(225,15,15,0.4) !important;
        text-shadow: 0 0 12px var(--shadow) !important;
    }
    .stNumberInput > div > div > input {
        background-color: var(--input-bg) !important;
        color: var(--main) !important;
        border: 1px solid var(--main) !important;
        box-shadow: inset 0 1px 5px rgba(225,15,15,0.2) !important;
        font-family: 'Orbitron', sans-serif !important;
    }
    .stSelectbox > div > div > select,
    .stRadio > div {
        background-color: var(--input-bg) !important;
        color: var(--main) !important;
        border: 1px solid var(--main) !important;
    }
    .stExpander {
        border: 1px solid var(--main) !important;
        background-color: var(--input-bg) !important;
        box-shadow: 0 3px 12px var(--shadow) !important;
    }
    hr {
        border-color: var(--main) !important;
        opacity: 0.35;
        margin: 1.2rem 0;
    }
    .stSuccess {
        background-color: rgba(100,100,255,0.10) !important;
        border: 1px solid var(--main) !important;
        box-shadow: 0 3px 12px var(--shadow) !important;
    }
    .stMetric {
        background-color: var(--input-bg) !important;
        border: 1px solid var(--main) !important;
        border-radius: 8px;
        padding: 10px 12px;
        box-shadow: 0 3px 10px var(--shadow),
                    0 -1px 4px rgba(225,15,15,0.15) !important;
        margin: 8px 0;
    }
    .stContainer[border="true"] {
        border: 1px solid var(--main) !important;
        border-radius: 10px;
        background-color: var(--input-bg) !important;
        box-shadow: 0 4px 12px var(--shadow),
                    0 -1px 4px rgba(225,15,15,0.15) !important;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ───────────────────────────────────────────────
# Global / default values
# ───────────────────────────────────────────────
DEFAULT = {
    "M": 1500.0,
    "D": 400.0,
    "C": 1.5,
    "F": 1.0,
    "T": 2.4,
    "W": 15.0,
    "N": 15.0,
}

# ───────────────────────────────────────────────
# Session state initialization
# ───────────────────────────────────────────────
if "player_a" not in st.session_state:
    st.session_state.player_a = {
        "r": 1500.0, "u": 64.0, "q": 1500.0, "v": 0.07,
        "r_new": None, "u_new": None, "v_new": None
    }
if "player_b" not in st.session_state:
    st.session_state.player_b = {
        "r": 1500.0, "u": 64.0, "q": 1500.0, "v": 0.07,
        "r_new": None, "u_new": None, "v_new": None
    }
if "global_vals" not in st.session_state:
    st.session_state.global_vals = DEFAULT.copy()

# ───────────────────────────────────────────────
# Functions
# ───────────────────────────────────────────────
def compute_mu(r, M, D): return (r - M) / D
def compute_sigma(u, D): return u / D
def compute_beta(q, M, D): return (q - M) / D

def compute_g(C, sigma_opp):
    return C / (C + sigma_opp)

def compute_e(mu_me, mu_opp, sigma_opp):
    diff = mu_me - mu_opp
    denom = sigma_opp + abs(diff)
    return 0.5 + 0.5 * diff / denom if denom != 0 else 0.5

def compute_p(beta_me, mu_opp, sigma_opp):
    diff = beta_me - mu_opp
    denom = sigma_opp + abs(diff)
    return 0.5 + 0.5 * diff / denom if denom != 0 else 0.5

def update_mu(mu, g, sigma, s, e, p):
    return mu + g * (sigma * (s - e) + sigma * (p - e))

def update_sigma(sigma, v, F, T, W):
    part1 = sigma * F / (sigma + F)
    part2 = v * (1 - sigma) / (v + W)
    return part1 + part2

def update_v(v, T, N, p, e):
    abs_diff = abs(p - e)
    part1 = v * T / (v + T)
    part2 = abs_diff * (1 - v) / (N + abs_diff)
    return part1 + part2

def run_update():
    g = st.session_state.global_vals
    M, D, C, F, T, W, N = g["M"], g["D"], g["C"], g["F"], g["T"], g["W"], g["N"]
    
    A = st.session_state.player_a
    B = st.session_state.player_b

    muA    = compute_mu(A["r"], M, D)
    muB    = compute_mu(B["r"], M, D)
    sigmaA = compute_sigma(A["u"], D)
    sigmaB = compute_sigma(B["u"], D)
    betaA  = compute_beta(A["q"], M, D)
    betaB  = compute_beta(B["q"], M, D)

    gA = compute_g(C, sigmaB)
    gB = compute_g(C, sigmaA)

    eA = compute_e(muA, muB, sigmaB)
    eB = compute_e(muB, muA, sigmaA)
    pA = compute_p(betaA, muB, sigmaB)
    pB = compute_p(betaB, muA, sigmaA)

    # Default to 1 if not set, but radio will override
    sA = st.session_state.get("outcome", 1.0)
    sB = 1 - sA

    mu_newA = update_mu(muA, gA, sigmaA, sA, eA, pA)
    sigma_newA = update_sigma(sigmaA, A["v"], F, T, W)
    v_newA = update_v(A["v"], T, N, pA, eA)

    mu_newB = update_mu(muB, gB, sigmaB, sB, eB, pB)
    sigma_newB = update_sigma(sigmaB, B["v"], F, T, W)
    v_newB = update_v(B["v"], T, N, pB, eB)

    A["r_new"] = M + D * mu_newA
    A["u_new"] = D * sigma_newA
    A["v_new"] = v_newA

    B["r_new"] = M + D * mu_newB
    B["u_new"] = D * sigma_newB
    B["v_new"] = v_newB

# ───────────────────────────────────────────────
# UI
# ───────────────────────────────────────────────
st.title("Shouji Rating System")
st.caption("A 2026 papered MMR system.")

col_global, col_outcome = st.columns([2, 1])

with col_global:
    with st.expander("Global Parameters", expanded=False):
        cols = st.columns(4)
        for i, k in enumerate(["M", "D", "C", "F"]):
            st.session_state.global_vals[k] = cols[i].number_input(
                k, value=DEFAULT[k], step=0.1, format="%.2f", key=f"g_{k}"
            )
        cols = st.columns(3)
        for i, k in enumerate(["T", "W", "N"]):
            st.session_state.global_vals[k] = cols[i].number_input(
                k, value=DEFAULT[k], step=0.1, format="%.2f", key=f"g2_{k}"
            )

colA, colB = st.columns(2)

def player_card(player_key, title):
    p = st.session_state[player_key]
    with st.container(border=True):
        st.subheader(title)
        c1, c2 = st.columns(2)
        p["r"] = c1.number_input("Rating (r)", value=p["r"], step=1.0, format="%.1f", key=f"{player_key}_r")
        p["u"] = c2.number_input("RD (u)", value=p["u"], step=1.0, min_value=1.0, format="%.1f", key=f"{player_key}_u")
        p["q"] = c1.number_input("Performance rating (q)", value=p["q"], step=10.0, format="%.1f", key=f"{player_key}_q")
        p["v"] = c2.number_input("Volatility (v)", value=p["v"], step=0.001, format="%.4f", key=f"{player_key}_v")

        if p["r_new"] is not None:
            st.markdown("**After match**")
            st.metric("New Rating", f"{p['r_new']:.0f}", delta=f"{p['r_new'] - p['r']:+.0f}")
            st.metric("New RD", f"{p['u_new']:.1f}", delta=f"{p['u_new'] - p['u']:+.1f}")
            st.metric("New Volatility", f"{p['v_new']:.4f}")

with colA:
    player_card("player_a", "Player A")

with colB:
    player_card("player_b", "Player B")

st.markdown("### Match Outcome")
outcome = st.radio(
    "Result",
    ["A wins", "Draw", "B wins"],
    horizontal=True,
    index=0   # ← default = A wins
)

if st.button("Calculate new ratings", type="primary", use_container_width=True):
    if outcome == "A wins":
        st.session_state.outcome = 1.0
    elif outcome == "Draw":
        st.session_state.outcome = 0.5
    else:
        st.session_state.outcome = 0.0
    
    run_update()
    st.expander("Ratings updated!")

st.markdown("---")
st.caption("Shouji Rating System • 2026")
