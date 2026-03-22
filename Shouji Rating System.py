# streamlit_app.py
import streamlit as st
import numpy as np

# ───────────────────────────────────────────────
#  Page config & custom styling
# ───────────────────────────────────────────────
st.set_page_config(page_title="Shouji Rating System", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --main: #6464ff;
        --bg: #000064;
        --shadow: #e10f0f;
    }
    .stApp {
        background-color: var(--bg) !important;
        color: var(--main) !important;
    }
    .stButton > button {
        background-color: var(--bg) !important;
        color: var(--main) !important;
        border: 1px solid var(--main) !important;
        box-shadow: 0 4px 8px var(--shadow) !important;
    }
    .stButton > button:hover {
        background-color: var(--bg) !important;
        border-color: var(--main) !important;
    }
    .stNumberInput > div > div > input {
        background-color: var(--bg) !important;
        color: #e0e0ff !important;
        border: 1px solid var(--main) !important;
    }
    .stSelectbox > div > div > select {
        background-color: #0a0a7a !important;
        color: #e0e0ff !important;
        border: 1px solid var(--main) !important;
    }
    h1, h2, h3 {
        color: var(--main) !important;
    }
    .stExpander {
        border: 1px solid var(--main) !important;
        background-color: var(--bg) !important;
    }
    hr {
        border-color: var(--main) !important;
    }
    .stSuccess {
        background-color: var(--bg) !important;
        border: 1px solid var(--main) !important;
        box-shadow: 1px solid var(--shadow) !important;
    }
    .stError {
        background-color: var(--bg) !important;
        border: 1px solid var(--main) !important;
        box-shadow: 1px solid var(--shadow) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ───────────────────────────────────────────────
#  Global / default values
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
#  Session state initialization
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
#  Functions
# ───────────────────────────────────────────────
def compute_mu(r, M, D): return (r - M) / D
def compute_sigma(u, D):  return u / D
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

    # Pre-match values
    muA = compute_mu(A["r"], M, D)
    muB = compute_mu(B["r"], M, D)
    sigmaA = compute_sigma(A["u"], D)
    sigmaB = compute_sigma(B["u"], D)
    betaA = compute_beta(A["q"], M, D)
    betaB = compute_beta(B["q"], M, D)

    gA = compute_g(C, sigmaB)
    gB = compute_g(C, sigmaA)

    eA = compute_e(muA, muB, sigmaB)
    eB = compute_e(muB, muA, sigmaA)
    pA = compute_p(betaA, muB, sigmaB)
    pB = compute_p(betaB, muA, sigmaA)

    # Outcome
    sA = st.session_state.get("outcome", 0.5)
    sB = 1 - sA

    # Update A
    mu_newA = update_mu(muA, gA, sigmaA, sA, eA, pA)
    sigma_newA = update_sigma(sigmaA, A["v"], F, T, W)
    v_newA = update_v(A["v"], T, N, pA, eA)

    # Update B
    mu_newB = update_mu(muB, gB, sigmaB, sB, eB, pB)
    sigma_newB = update_sigma(sigmaB, B["v"], F, T, W)
    v_newB = update_v(B["v"], T, N, pB, eB)

    # Save new values
    A["r_new"] = M + D * mu_newA
    A["u_new"] = D * sigma_newA
    A["v_new"] = v_newA

    B["r_new"] = M + D * mu_newB
    B["u_new"] = D * sigma_newB
    B["v_new"] = v_newB

# ───────────────────────────────────────────────
#  UI
# ───────────────────────────────────────────────
st.title("Shouji Rating System")
st.markdown("A 2026 papered mmr system.")

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
outcome = st.radio("Result", ["A wins", "Draw", "B wins"], horizontal=True, index=1)

if st.button("Calculate new ratings", type="primary", use_container_width=True):
    if outcome == "A wins":
        st.session_state.outcome = 1.0
    elif outcome == "Draw":
        st.session_state.outcome = 0.5
    else:
        st.session_state.outcome = 0.0

    run_update()
    st.success("Ratings updated!")
    st.rerun()

st.markdown("---")
st.caption("A 2026 papered mmr system.")
