# streamlit_app.py
import streamlit as st
import numpy as np

# ───────────────────────────────────────────────
# Page config & styling
# ───────────────────────────────────────────────
st.set_page_config(page_title="Shouji Rating System", layout="wide")

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&display=swap" rel="stylesheet">

<style>
:root {
    --main:   #6464ff;
    --bg:     #000064;
    --shadow: #e10f0f;
    --input-bg: #0a0a4a;
}

/* Base */
.stApp {
    background:
        radial-gradient(circle at 20% 30%, rgba(100,100,255,0.10), transparent 45%),
        radial-gradient(circle at 80% 70%, rgba(225,15,15,0.08), transparent 45%),
        radial-gradient(circle at 50% 50%, rgba(100,100,255,0.04), transparent 60%),
        var(--bg) !important;
    font-family: 'Orbitron', sans-serif !important;
}

* {
    color: var(--main) !important;
    transition: all 0.2s ease;
}

/* Glow animation */
@keyframes pulseGlow {
    0%   { text-shadow: 0 0 6px var(--shadow); }
    50%  { text-shadow: 0 0 14px var(--shadow); }
    100% { text-shadow: 0 0 6px var(--shadow); }
}

h1, h2, h3, h4, h5, h6, label {
    animation: pulseGlow 3s infinite ease-in-out;
}

/* Cards */
.stContainer[border="true"],
.stMetric,
.stExpander {
    backdrop-filter: blur(10px);
    border-radius: 14px;
    border: 1px solid rgba(100,100,255,0.6) !important;

    background:
        linear-gradient(145deg, rgba(10,10,74,0.95), rgba(10,10,74,0.65));

    box-shadow:
        0 8px 24px var(--shadow),
        inset 0 0 14px rgba(225,15,15,0.15);
}

.stContainer[border="true"]:hover,
.stMetric:hover,
.stExpander:hover {
    transform: translateY(-3px) scale(1.01);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(145deg, #0a0a4a, #101070) !important;
    border-radius: 10px;
    border: 1px solid var(--main) !important;
    position: relative;
    overflow: hidden;
}

.stButton > button::after {
    content: "";
    position: absolute;
    left: -120%;
    width: 100%;
    height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.2), transparent);
}

.stButton > button:hover::after {
    left: 120%;
    transition: 0.5s;
}

/* Inputs */
.stNumberInput input,
.stSelectbox select {
    background-color: var(--input-bg) !important;
    border-radius: 8px;
    border: 1px solid var(--main) !important;
}

/* Divider */
hr {
    height: 2px;
    background: linear-gradient(to right, transparent, var(--main), transparent);
}

/* Success */
.stSuccess {
    border-radius: 10px;
}
</style>
    """,
    unsafe_allow_html=True
)

# ───────────────────────────────────────────────
# Defaults
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
# Session state
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
# Math functions
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
    return sigma * F / (sigma + F) + v * (1 - sigma) / (v + W)

def update_v(v, T, N, p, e):
    diff = abs(p - e)
    return v * T / (v + T) + diff * (1 - v) / (N + diff)

# ───────────────────────────────────────────────
# Update logic
# ───────────────────────────────────────────────
def run_update():
    g = st.session_state.global_vals
    A = st.session_state.player_a
    B = st.session_state.player_b

    muA = compute_mu(A["r"], g["M"], g["D"])
    muB = compute_mu(B["r"], g["M"], g["D"])

    sigmaA = compute_sigma(A["u"], g["D"])
    sigmaB = compute_sigma(B["u"], g["D"])

    betaA = compute_beta(A["q"], g["M"], g["D"])
    betaB = compute_beta(B["q"], g["M"], g["D"])

    gA = compute_g(g["C"], sigmaB)
    gB = compute_g(g["C"], sigmaA)

    eA = compute_e(muA, muB, sigmaB)
    eB = compute_e(muB, muA, sigmaA)

    pA = compute_p(betaA, muB, sigmaB)
    pB = compute_p(betaB, muA, sigmaA)

    sA = st.session_state.get("outcome", 1.0)
    sB = 1 - sA

    mu_newA = update_mu(muA, gA, sigmaA, sA, eA, pA)
    mu_newB = update_mu(muB, gB, sigmaB, sB, eB, pB)

    A["r_new"] = g["M"] + g["D"] * mu_newA
    B["r_new"] = g["M"] + g["D"] * mu_newB

    A["u_new"] = g["D"] * update_sigma(sigmaA, A["v"], g["F"], g["T"], g["W"])
    B["u_new"] = g["D"] * update_sigma(sigmaB, B["v"], g["F"], g["T"], g["W"])

    A["v_new"] = update_v(A["v"], g["T"], g["N"], pA, eA)
    B["v_new"] = update_v(B["v"], g["T"], g["N"], pB, eB)

# ───────────────────────────────────────────────
# UI
# ───────────────────────────────────────────────
st.title("Shouji Rating System")
st.caption("A 2026 papered MMR system.")

colA, colB = st.columns(2)

def player_card(key, title):
    p = st.session_state[key]
    with st.container(border=True):
        st.subheader(title)
        c1, c2 = st.columns(2)

        p["r"] = c1.number_input("Rating", value=p["r"])
        p["u"] = c2.number_input("RD", value=p["u"])
        p["q"] = c1.number_input("Performance", value=p["q"])
        p["v"] = c2.number_input("Volatility", value=p["v"])

        if p["r_new"] is not None:
            st.metric("New Rating", f"{p['r_new']:.0f}", delta=f"{p['r_new'] - p['r']:+.0f}")
            st.metric("New RD", f"{p['u_new']:.1f}")
            st.metric("Volatility", f"{p['v_new']:.4f}")

with colA:
    player_card("player_a", "Player A")

with colB:
    player_card("player_b", "Player B")

outcome = st.radio("Result", ["A wins", "Draw", "B wins"], horizontal=True)

if st.button("Calculate", use_container_width=True):
    st.session_state.outcome = {"A wins":1.0,"Draw":0.5,"B wins":0.0}[outcome]
    run_update()
    st.success("Ratings updated!")

st.markdown("---")
st.caption("Shouji Rating System • 2026")
