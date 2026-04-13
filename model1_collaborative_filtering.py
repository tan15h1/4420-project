"""
DS 4420 Final Project - Model 1: Item-Based Collaborative Filtering
Tanishi Datta & Shruthi Palaniappan

Manual implementation (no sklearn or other modeling packages).
Uses a recipe x ingredient binary rating matrix with PMI and cosine similarity
to recommend ingredient substitutes given a partial recipe.

Dataset: RAW_recipes.csv from Food.com (Kaggle)
"""

import pandas as pd
import numpy as np
import ast
import random
from collections import Counter

# ──────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────

print("Loading data...")
# RAW_recipes.csv uses a non-standard format: each physical line is wrapped
# in outer double-quotes, with multi-line descriptions spanning multiple lines.
# Direct pd.read_csv mis-parses it, so we extract ingredient lists via regex.
import re

with open("RAW_recipes.csv", "rb") as _f:
    _raw = _f.read()

# Each complete record ends with: ""['ing1', 'ing2', ...]"",N"
# where N is n_ingredients and the trailing " closes the outer-quoted row.
_pattern = rb'""(\[[^\]]*\])"",(\d+)"'
_matches = re.findall(_pattern, _raw)

_ingredients_data = []
for _ing_bytes, _n_bytes in _matches:
    try:
        _ing_str = _ing_bytes.decode("utf-8", errors="replace")
        _ing_list = ast.literal_eval(_ing_str)
        _n = int(_n_bytes)
        if isinstance(_ing_list, list) and 1 <= _n <= 100 and len(_ing_list) == _n:
            _ingredients_data.append(_ing_list)
    except (ValueError, SyntaxError):
        pass

df = pd.DataFrame({"ingredients": _ingredients_data})
print(f"  Raw records extracted: {len(df)}")
_n = len(df)

# ── Cleaning pipeline ──────────────────────────────────────────────────────

# 1. Drop rows where the ingredient list is null or empty
df = df.dropna(subset=["ingredients"])
df = df[df["ingredients"].apply(lambda x: len(x) > 0)].reset_index(drop=True)
print(f"  Step 1 – drop null/empty lists:       removed {_n - len(df):>5}, {len(df)} remaining")
_n = len(df)

# 2. Remove garbled rows: any ingredient containing a tab character.
#    These appear as 'por\tk spare\tribs' or 'fres\th ga\trl\tic' and are
#    artifacts of the non-standard CSV encoding.
df = df[~df["ingredients"].apply(
    lambda ings: any("\t" in ing for ing in ings)
)].reset_index(drop=True)
print(f"  Step 2 – drop tab-garbled rows:       removed {_n - len(df):>5}, {len(df)} remaining")
_n = len(df)

# 3. Per-entry cleaning within each ingredient list:
#    - strip whitespace and lowercase
#    - drop empty strings and lone-punctuation entries
#    - drop purely numeric tokens (e.g. '1', '42') — not real ingredient names
def _clean_list(ings):
    cleaned = []
    for ing in ings:
        ing = ing.strip().lower()
        if len(ing) > 1 and any(c.isalpha() for c in ing):
            cleaned.append(ing)
    return cleaned

df["ingredients"] = df["ingredients"].apply(_clean_list)
print(f"  Step 3 – strip/lowercase/drop numeric: {len(df)} rows (entries cleaned in-place)")

# 4. Remove recipes with fewer than 3 usable ingredients after per-entry cleaning
df = df[df["ingredients"].apply(len) >= 3].reset_index(drop=True)
print(f"  Step 4 – require ≥3 ingredients:      removed {_n - len(df):>5}, {len(df)} remaining")
_n = len(df)

# 5. Deduplicate: drop recipes whose ingredient sets are identical
df["_ing_key"] = df["ingredients"].apply(lambda x: frozenset(x))
df = df.drop_duplicates(subset=["_ing_key"]).drop(columns=["_ing_key"]).reset_index(drop=True)
print(f"  Step 5 – deduplicate by ingredient set: removed {_n - len(df):>5}, {len(df)} remaining")

print(f"  ── After cleaning: {len(df)} recipes ──")

# Sample for tractability
df = df.sample(n=min(20_000, len(df)), random_state=42).reset_index(drop=True)

# Build vocabulary: keep ingredients appearing in >= 20 recipes
ingredient_counts = Counter(ing for recipe in df["ingredients"] for ing in recipe)
vocab = [ing for ing, cnt in ingredient_counts.items() if cnt >= 20]
ing2idx = {ing: idx for idx, ing in enumerate(vocab)}
V = len(vocab)
print(f"Vocabulary size: {V} ingredients | Recipes: {len(df)}")

# ──────────────────────────────────────────────
# 2. BUILD RECIPE x INGREDIENT RATING MATRIX
# ──────────────────────────────────────────────
# Binary: R[recipe_i, ing_j] = 1 if ingredient j appears in recipe i

print("Building rating matrix...")
R = np.zeros((len(df), V), dtype=np.float32)
for row_idx, row in df.iterrows():
    for ing in row["ingredients"]:
        if ing in ing2idx:
            R[row_idx, ing2idx[ing]] = 1.0

# ──────────────────────────────────────────────
# 3. PMI SIMILARITY (MANUAL)
# ──────────────────────────────────────────────
# PMI(i, j) = log[ P(i,j) / (P(i) * P(j)) ]
# P(i,j) = number of recipes containing both i and j / total recipes
# P(i)   = number of recipes containing i / total recipes
# We clamp to PPMI (positive PMI only)

print("Computing PMI similarity matrix (this may take a minute)...")
N = len(df)
freq_i = R.sum(axis=0)           # shape (V,)  — how many recipes each ingredient appears in
co_occ = R.T @ R                  # shape (V, V) — co-occurrence counts (matrix multiply)

p_i = freq_i / N                  # marginal probabilities
p_ij = co_occ / N                 # joint probabilities

# PPMI: max(log(P(i,j) / P(i)*P(j)), 0)
# Avoid log(0): mask zero entries
with np.errstate(divide="ignore", invalid="ignore"):
    outer = np.outer(p_i, p_i)   # P(i)*P(j) for all pairs
    pmi = np.where(
        (p_ij > 0) & (outer > 0),
        np.log(np.where(outer > 0, p_ij / outer, 1.0)),
        0.0
    )
ppmi = np.maximum(pmi, 0).astype(np.float32)

# ──────────────────────────────────────────────
# 4. COSINE SIMILARITY ON PPMI VECTORS (MANUAL)
# ──────────────────────────────────────────────
# Each ingredient is represented by its row in the PPMI matrix.
# cos_sim[i, j] = dot(ppmi[i], ppmi[j]) / (||ppmi[i]|| * ||ppmi[j]||)

print("Computing cosine similarity...")
norms = np.linalg.norm(ppmi, axis=1, keepdims=True)  # shape (V, 1)
norms[norms == 0] = 1e-10  # avoid division by zero
ppmi_normed = ppmi / norms
cos_sim = ppmi_normed @ ppmi_normed.T   # shape (V, V)

# ──────────────────────────────────────────────
# 5. INGREDIENT SUBSTITUTION FUNCTION
# ──────────────────────────────────────────────

def get_substitutes(target_ingredient: str, context: list[str], top_n: int = 5) -> list[tuple]:
    """
    Recommend top_n substitutes for target_ingredient given a list of context ingredients.

    Scoring (weighted combination):
      - 60% cosine similarity between target and candidate (PPMI vectors)
      - 40% mean PPMI affinity between candidate and context ingredients
        (measures how well the candidate fits the existing recipe context)

    Parameters
    ----------
    target_ingredient : str
    context           : list of other ingredients in the recipe
    top_n             : number of substitutes to return

    Returns
    -------
    List of (ingredient, score) tuples, sorted descending by score.
    """
    if target_ingredient not in ing2idx:
        print(f"  '{target_ingredient}' not in vocabulary.")
        return []
    
    target_idx   = ing2idx[target_ingredient]
    context_idxs = [ing2idx[i] for i in context if i in ing2idx]
    
    scores = {}
    for ing, idx in ing2idx.items():
        if ing == target_ingredient or ing in context:
            continue
        
        sim_score = float(cos_sim[target_idx, idx])
        
        if context_idxs:
            context_score = float(np.mean(ppmi[idx, context_idxs]))
        else:
            context_score = 0.0
        
        scores[ing] = 0.6 * sim_score + 0.4 * context_score
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


# ──────────────────────────────────────────────
# 6. EVALUATION
# ──────────────────────────────────────────────
# Leave-one-out: for each recipe, remove one ingredient at random,
# try to predict it from the remaining context.
#
# Since there's no canonical ground-truth "substitute" list,
# we use a proxy: check whether the prediction is among the
# top-20 nearest neighbours of the held-out ingredient in PPMI space.
#
# Metrics:
#   Hit@5  — is any of the top-5 predictions a near-neighbour?
#   MRR    — mean reciprocal rank of the first near-neighbour hit

print("\nRunning evaluation (leave-one-out, n=300)...")
random.seed(99)

eval_recipes = df[df["ingredients"].apply(
    lambda x: sum(1 for i in x if i in ing2idx) >= 4
)].sample(n=300, random_state=99)

hits_at_5 = 0
mrr_total = 0.0
eval_count = 0

for _, row in eval_recipes.iterrows():
    recipe = [i for i in row["ingredients"] if i in ing2idx]
    if len(recipe) < 4:
        continue
    
    missing = random.choice(recipe)
    partial = [i for i in recipe if i != missing]
    missing_idx = ing2idx[missing]
    
    # Top-20 nearest neighbours of missing ingredient (proxy ground truth)
    sim_scores = cos_sim[missing_idx].copy()
    sim_scores[missing_idx] = -1.0
    top20_neighbors = set(vocab[i] for i in np.argsort(sim_scores)[-20:])
    
    preds = [p[0] for p in get_substitutes(missing, partial, top_n=5)]
    
    if any(p in top20_neighbors for p in preds):
        hits_at_5 += 1
    
    for rank, p in enumerate(preds, start=1):
        if p in top20_neighbors:
            mrr_total += 1.0 / rank
            break
    
    eval_count += 1

print(f"Evaluation on {eval_count} recipes:")
print(f"  Hit@5 : {hits_at_5 / eval_count:.3f}")
print(f"  MRR   : {mrr_total / eval_count:.3f}")

# ──────────────────────────────────────────────
# 7. DEMO
# ──────────────────────────────────────────────

demo_cases = [
    ("butter",        ["flour", "sugar", "eggs", "vanilla extract"]),
    ("soy sauce",     ["garlic", "ginger", "sesame oil", "rice"]),
    ("heavy cream",   ["onion", "garlic", "pasta", "parmesan cheese"]),
    ("eggs",          ["flour", "sugar", "butter", "baking powder"]),
    ("olive oil",     ["garlic", "tomatoes", "basil", "pasta"]),
]

print("\n" + "="*55)
print("DEMO: Top-5 substitutes")
print("="*55)
for target, context in demo_cases:
    subs = get_substitutes(target, context, top_n=5)
    print(f"\nSubstitutes for '{target}' in context {context}:")
    for ing, score in subs:
        print(f"  {ing:<32} {score:.4f}")


# ──────────────────────────────────────────────
# 8. SAVE DATA FOR STREAMLIT APP + SUMMARY
# ──────────────────────────────────────────────

top1000_ings = [ing for ing, _ in ingredient_counts.most_common(1000) if ing in ing2idx]
top1000_idxs = [ing2idx[i] for i in top1000_ings]
cos_sim_top  = cos_sim[np.ix_(top1000_idxs, top1000_idxs)]
ppmi_top     = ppmi[np.ix_(top1000_idxs, top1000_idxs)]

np.save("cos_sim_top1000.npy", cos_sim_top)
np.save("ppmi_top1000.npy",    ppmi_top)
with open("vocab_top1000.txt", "w") as f:
    for ing in top1000_ings:
        f.write(ing + "\n")
print("\nSaved: cos_sim_top1000.npy, ppmi_top1000.npy, vocab_top1000.txt")

print("\n" + "="*55)
print("TOP-1000 SIMILARITY MATRIX SUMMARY")
print("="*55)
print(f"  Ingredients in top-1000 slice : {len(top1000_ings)}")
print(f"  Cosine sim — min: {cos_sim_top.min():.4f}  max: {cos_sim_top.max():.4f}  mean: {cos_sim_top.mean():.4f}")
print(f"  PPMI        — min: {ppmi_top.min():.4f}  max: {ppmi_top.max():.4f}  mean: {ppmi_top.mean():.4f}")
print(f"\n  Top-10 ingredients by frequency:")
for ing, cnt in ingredient_counts.most_common(10):
    print(f"    {ing:<30} {cnt:>5} recipes")
print("\nDone.")
