"""
DS 4420 Final Project — Streamlit App (Extra Credit)
Tanishi Datta & Shruthi Palaniappan

Two-tab app:
  Tab 1: Project overview / landing page
  Tab 2: Interactive ingredient substitution explorer
         (uses the CF model's precomputed similarity data)

Run with:
  streamlit run app.py

Prerequisites:
  pip install streamlit pandas numpy plotly
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Ingredient Substitution Explorer",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #FFFFFF;
        color: #1B3A2D;
    }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; color: #1B3A2D; }

    /* ── Tab bar ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #F5F0E4;
        border-radius: 10px;
        padding: 4px 6px;
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #3A5A40;
        font-weight: 500;
        padding: 6px 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2D6A4F !important;
        color: #FFFFFF !important;
    }

    /* ── Hero ── */
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem;
        color: #1B3A2D;
        margin-bottom: 0.25rem;
        line-height: 1.15;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #5A7A60;
        margin-bottom: 2rem;
    }

    /* ── Section headers ── */
    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        color: #1B3A2D;
        border-bottom: 2px solid #C9B840;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #F9F5E4;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #2D6A4F;
        margin-bottom: 1rem;
    }
    .metric-value { font-size: 1.8rem; font-weight: 600; color: #2D6A4F; }
    .metric-sub   { color: #6B7C69; font-size: 0.9rem; }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #2D6A4F;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #1B4332;
        color: #F5F0E4;
    }

    /* ── Sidebar / selectbox / slider accents ── */
    .stSlider [data-baseweb="slider"] > div { background: #2D6A4F; }

    /* ── Divider ── */
    hr { border-color: #E8E0C8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_resource
def load_data():
    try:
        with open("vocab_top1000.txt") as f:
            vocab = [line.strip() for line in f.readlines()]
        cos_sim  = np.load("cos_sim_top1000.npy")
        ppmi_mat = np.load("ppmi_top1000.npy")
        ing2idx  = {ing: idx for idx, ing in enumerate(vocab)}
        return vocab, cos_sim, ppmi_mat, ing2idx
    except FileNotFoundError:
        return None, None, None, None


vocab, cos_sim, ppmi_mat, ing2idx = load_data()
data_loaded = vocab is not None


# ─────────────────────────────────────────────
# SUBSTITUTION LOGIC
# ─────────────────────────────────────────────

def get_substitutes(target, context, top_n=8):
    if not data_loaded or target not in ing2idx:
        return pd.DataFrame()
    t_idx    = ing2idx[target]
    ctx_idxs = [ing2idx[i] for i in context if i in ing2idx]
    exclude  = {target} | set(context)
    rows = []
    for ing, idx in ing2idx.items():
        if ing in exclude:
            continue
        sim   = float(cos_sim[t_idx, idx])
        ctx_s = float(np.mean(ppmi_mat[idx, ctx_idxs])) if ctx_idxs else 0.0
        rows.append({"ingredient": ing, "similarity": sim,
                     "context_fit": ctx_s, "score": 0.6 * sim + 0.4 * ctx_s})
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2 = st.tabs(["🏠  Project Overview", "🔍  Substitution Explorer"])

# ══════════════════════════════════════════════
# TAB 1 — LANDING PAGE
# ══════════════════════════════════════════════

with tab1:
    st.markdown('<div class="hero-title">Ingredient Substitution<br>Using Machine Learning</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">DS 4420 · Tanishi Datta & Shruthi Palaniappan · Spring 2026</div>',
                unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown('<div class="section-header">The Problem</div>', unsafe_allow_html=True)
        st.write("""
        Cooking often requires adapting recipes when certain ingredients are unavailable,
        restricted by diet, or simply not preferred. Good substitutions must preserve the
        **flavor**, **texture**, **functionality**, and **nutritional value** of the original
        ingredient — a genuinely multi-dimensional challenge.

        With large online recipe databases, we can analyze thousands of recipes to learn which
        ingredients tend to appear in the same contexts, revealing which ones play *similar
        roles* in cooking.
        """)

        st.markdown('<div class="section-header">Our Approach</div>', unsafe_allow_html=True)
        st.write("""
        We apply two machine learning methods to the **Food.com dataset** (230k+ recipes):

        **Model 1 — Item-Based Collaborative Filtering** *(Python, manual)*
        Build a recipe × ingredient binary matrix and compute Positive PMI embeddings.
        Rank candidates by a weighted combination of cosine similarity (how similar the
        candidate is to the target) and context fit (how well the candidate pairs with the
        rest of the recipe).

        **Model 2 — Bayesian Logistic Regression** *(R, brms/Stan)*
        Treat substitutability as a binary prediction problem. Estimate the posterior
        probability that a candidate is a good substitute using cosine similarity, PPMI score,
        and context overlap as features, with weakly-informative Normal(0,1) priors.
        """)

    with col2:
        st.markdown('<div class="section-header">Dataset & Results</div>', unsafe_allow_html=True)
        for label, value, sub in [
            ("Food.com Recipes", "230,000+", "recipes analyzed"),
            ("Vocabulary", "~1,500", "unique ingredients (≥20 occurrences)"),
            ("CF Model Hit@5", "~0.65+", "proxy leave-one-out evaluation"),
        ]:
            st.markdown(f"""
            <div class="metric-card">
                <b>{label}</b><br>
                <span class="metric-value">{value}</span><br>
                <span class="metric-sub">{sub}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("**Dataset:** [Food.com on Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)")
        st.markdown("**GitHub:** [tan15h1/4420-project](https://github.com/tan15h1/4420-project)")

    st.divider()
    st.markdown('<div class="section-header">References</div>', unsafe_allow_html=True)
    st.markdown("""
    1. H. M., M. G., and P. S., "A Comprehensive Framework for Nutrition Analysis and Ingredient Substitution Using Machine Learning," *ICDSAAI 2025*, doi: 10.1109/ICDSAAI65575.2025.11011623.
    2. E. Oz and F. Oz, "AI-Enabled Ingredient Substitution in Food Systems," *Foods*, vol. 14, 2025, doi: 10.3390/foods14223919.
    3. P. Fermín Cueto et al., "Completing Partial Recipes Using Item-Based Collaborative Filtering," *arXiv:1907.11250*, 2020.
    """)


# ══════════════════════════════════════════════
# TAB 2 — INTERACTIVE EXPLORER
# ══════════════════════════════════════════════

with tab2:
    st.markdown('<div class="hero-title">Substitution Explorer</div>', unsafe_allow_html=True)
    st.markdown("Find ingredient substitutes based on your recipe context using the collaborative filtering model.")
    st.divider()

    if not data_loaded:
        st.error("""
        ⚠️ Precomputed similarity data not found.
        Please run `model1_collaborative_filtering.py` first to generate:
        `cos_sim_top1000.npy`, `ppmi_top1000.npy`, `vocab_top1000.txt`
        """)
    else:
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.subheader("Configure Your Recipe")
            target_ing = st.selectbox(
                "🎯 Ingredient to substitute",
                options=sorted(vocab),
                index=sorted(vocab).index("butter") if "butter" in vocab else 0,
            )
            context_ings = st.multiselect(
                "📋 Other ingredients in your recipe",
                options=[v for v in sorted(vocab) if v != target_ing],
                default=[v for v in ["flour", "sugar", "eggs", "vanilla extract"] if v in vocab],
            )
            top_n = st.slider("Substitutes to show", 3, 15, 8)
            st.button("Find Substitutes →", type="primary", use_container_width=True)

        with right:
            results = get_substitutes(target_ing, context_ings, top_n=top_n)

            if results.empty:
                st.warning("No results found.")
            else:
                st.subheader(f"Top substitutes for **{target_ing}**")

                fig = go.Figure(go.Bar(
                    y=results["ingredient"][::-1],
                    x=results["score"][::-1],
                    orientation="h",
                    marker=dict(
                        color=results["score"][::-1],
                        colorscale=[[0, "#F5EFA0"], [0.5, "#52B788"], [1, "#1B4332"]],
                    ),
                    text=[f"{s:.3f}" for s in results["score"][::-1]],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>",
                ))
                fig.update_layout(
                    xaxis_title="Score (0.6 × cosine_sim + 0.4 × context_fit)",
                    height=max(300, top_n * 38),
                    margin=dict(l=10, r=80, t=20, b=40),
                    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                    font=dict(family="DM Sans", size=13, color="#1B3A2D"),
                    xaxis=dict(showgrid=True, gridcolor="#E8E0C8"),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📊 Score breakdown"):
                    display_df = results.rename(columns={
                        "ingredient": "Ingredient", "similarity": "Cosine Similarity",
                        "context_fit": "Context Fit", "score": "Final Score"
                    }).round(4)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                if context_ings and len(results) >= 3:
                    st.subheader("Context Similarity Heatmap")
                    st.caption("PPMI affinity between top substitutes and your context ingredients")
                    top_cands = results["ingredient"].head(8).tolist()
                    ctx_show  = context_ings[:6]
                    heat_data = np.zeros((len(top_cands), len(ctx_show)))
                    for i, cand in enumerate(top_cands):
                        if cand in ing2idx:
                            for j, ctx in enumerate(ctx_show):
                                if ctx in ing2idx:
                                    heat_data[i, j] = ppmi_mat[ing2idx[cand], ing2idx[ctx]]
                    fig2 = px.imshow(
                        heat_data, x=ctx_show, y=top_cands, aspect="auto",
                        color_continuous_scale=["#FFFFFF", "#F5EFA0", "#52B788", "#1B4332"],
                        labels=dict(color="PPMI"),
                    )
                    fig2.update_layout(
                        height=300,
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="#FFFFFF",
                        font=dict(family="DM Sans", size=12, color="#1B3A2D"),
                    )
                    st.plotly_chart(fig2, use_container_width=True)
