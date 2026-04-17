import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="SwapSmart", layout="centered")

@st.cache_resource
def load_data():
    with open("vocab_top1000.txt") as f:
        vocab = [line.strip() for line in f.readlines()]
    cos_sim = np.load("cos_sim_top1000.npy")
    ppmi    = np.load("ppmi_top1000.npy")
    ing2idx = {ing: idx for idx, ing in enumerate(vocab)}
    return vocab, cos_sim, ppmi, ing2idx

vocab, cos_sim, ppmi, ing2idx = load_data()

tab1, tab2 = st.tabs(["About", "Find Substitutes"])

with tab1:
    st.title("🍳 SwapSmart")
    st.write("**Swap Smart: Ingredient Substitution Based on Recipe Co-Occurrence**")
    st.write("DS 4420 Final Project — Tanishi Datta & Shruthi Palaniappan")
    st.write("*Ever been mid-recipe and realize you're out of an ingredient? We used machine learning to help with that.*")

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Recipes analyzed", "230,000+")
    col2.metric("Unique ingredients", "~1,500")
    col3.metric("Models built", "2")

    st.divider()

    st.subheader("How it works")
    st.write("""
    We trained on the Food.com recipe dataset to learn which ingredients tend to appear
    in the same cooking contexts. If two ingredients show up in similar recipes, they are
    likely interchangeable.

    **Model 1 - Collaborative Filtering (Python)**
    Treats recipes like users and ingredients like items. Ranks substitutes by cosine
    similarity and how well they fit the rest of your recipe.

    **Model 2 - Bayesian Logistic Regression (R)**
    Predicts whether two ingredients are substitutes using cosine similarity, PPMI score,
    and context overlap as features.
    """)

with tab2:
    st.title("Find Substitutes")

    target = st.selectbox("Ingredient to substitute", sorted(vocab))

    context = st.multiselect(
        "Other ingredients in your recipe (optional)",
        [v for v in sorted(vocab) if v != target]
    )

    top_n = st.slider("How many substitutes to show", 3, 10, 5)

    if st.button("Search"):
        t_idx = ing2idx[target]
        exclude = {target} | set(context)
        ctx_idxs = [ing2idx[i] for i in context if i in ing2idx]

        rows = []
        for ing, idx in ing2idx.items():
            if ing in exclude:
                continue
            sim = float(cos_sim[t_idx, idx])
            ctx = float(np.mean(ppmi[idx, ctx_idxs])) if ctx_idxs else 0.0
            rows.append({"Ingredient": ing, "Score": round(0.6 * sim + 0.4 * ctx, 4)})

        results = pd.DataFrame(rows).sort_values("Score", ascending=False).head(top_n).reset_index(drop=True)
        results.index += 1

        st.subheader(f"Top substitutes for '{target}'")
        st.dataframe(results, use_container_width=True)
