"""
DS 4420 Final Project - Model 1: Item-Based Collaborative Filtering
Tanishi Datta & Shruthi Palaniappan

Manual implementation
Uses a recipe x ingredient rating matrix with PMI and cosine similarity
to recommend ingredient substitutes given a partial recipe.

Dataset: RAW_recipes.csv from Food.com (Kaggle)
"""

import pandas as pd
import numpy as np
import random
from collections import Counter

# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────
# Run preprocess.py first to generate recipes_clean.pkl

df = pd.read_pickle("recipes_clean.pkl")

# Sample down from ~170,000 to 20,000 recipes
df = df.sample(n=min(20_000, len(df)), random_state=42).reset_index(drop=True)

# Build vocabulary - keep ingredients that appear in at least 20 recipes
ingredient_counts = Counter()
for recipe in df["ingredients"]:
    for i in recipe:
        ingredient_counts[i] += 1

vocab = [i for i, cnt in ingredient_counts.items() if cnt >= 20]
ing2idx = {i: idx for idx, i in enumerate(vocab)}
V = len(vocab)
print(f"Vocabulary size: {V} ingredients | Recipes: {len(df)}")

# ──────────────────────────────────────────────
# 2. BUILD RECIPE x INGREDIENT RATING MATRIX
# ──────────────────────────────────────────────
# Binary: R[recipe_i, ing_j] = 1 if ingredient j appears in recipe i

R = np.zeros((len(df), V), dtype=np.float32)
for row_idx, row in df.iterrows():
    for ing in row["ingredients"]:
        if ing in ing2idx:
            R[row_idx, ing2idx[ing]] = 1.0

# ──────────────────────────────────────────────
# 3. PPMI SIMILARITY
# ──────────────────────────────────────────────
# PMI(i, j) = log( P(i,j) / (P(i) * P(j)) )
n = len(df)

# How often each ingredient appears across all recipes
freq = R.sum(axis=0)
p_i = freq / n

# How often each pair of ingredients appears together
# R.T @ R gives a V x V matrix 
co_occ = R.T @ R
p_ij = co_occ / n

pmi = np.log((p_ij + 1e-10) / (np.outer(p_i, p_i) + 1e-10))

# Keep only positive pmi values
ppmi = np.maximum(pmi, 0)

# ──────────────────────────────────────────────
# 4. COSINE SIMILARITY ON PPMI VECTORS
# ──────────────────────────────────────────────
# cos_sim[i, j] = i * j / (||i|| x ||j||)
# Normalize each row to unit length 
# then dot product gives all pairwise cosines

norms = np.linalg.norm(ppmi, axis=1)
norms[norms == 0] = 1e-10

ppmi_normed = ppmi / norms.reshape(-1, 1)
cos_sim = ppmi_normed @ ppmi_normed.T

# ──────────────────────────────────────────────
# 5. TAG-INGREDIENT co-occurrence matrix
# ──────────────────────────────────────────────
# Build tag vocabulary - keep tags that appear in at least 20 recipes
tag_counts = Counter()
for tags in df["tags"]:
    for tag in tags:
        tag_counts[tag] += 1

tag_vocab = [t for t, cnt in tag_counts.items() if cnt >= 20]
tag2idx = {t: idx for idx, t in enumerate(tag_vocab)}
T = len(tag_vocab)
print(f"Tag vocabulary: {T} tags")

# tag_ing[t, i] = number of recipes that have both tag t and ingredient i
tag_ing = np.zeros((T, V), dtype=np.float32)
for _, row in df.iterrows():
    for tag in row["tags"]:
        if tag in tag2idx:
            for ing in row["ingredients"]:
                if ing in ing2idx:
                    tag_ing[tag2idx[tag], ing2idx[ing]] += 1

# Compute tag-ingredient PPMI
# p_i is already computed in section 3 (ingredient marginal probabilities)
p_tag = tag_ing.sum(axis=1) / len(df)
p_tag_ing = tag_ing / len(df)

tag_ing_ppmi = np.log((p_tag_ing + 1e-10) / (np.outer(p_tag, p_i) + 1e-10))
tag_ing_ppmi = np.maximum(tag_ing_ppmi, 0)

# ──────────────────────────────────────────────
# 6. INGREDIENT SUBSTITUTION FUNCTION
# ──────────────────────────────────────────────
def get_substitutes(target_ingredient, context_ingredients, recipe_tags=None, top_n=5):
    if target_ingredient not in ing2idx:
        print(f"'{target_ingredient}' not in vocabulary.")
        return []

    target_idx = ing2idx[target_ingredient]
    context_idxs = [ing2idx[i] for i in context_ingredients if i in ing2idx]
    tag_idxs = [tag2idx[t] for t in (recipe_tags or []) if t in tag2idx]

    scores = {}
    for ing, idx in ing2idx.items():
        # skip the target itself and anything already in the recipe
        if ing == target_ingredient or ing in context_ingredients:
            continue

        # how similar is the candidate to the target ingredient
        sim_score = float(cos_sim[target_idx, idx])

        # how well does the candidate fit the rest of the recipe (context)
        context_score = float(np.mean(ppmi[idx, context_idxs])) if context_idxs else 0.0

        # how well does the candidate fit the recipe's tags
        tag_score = float(np.mean(tag_ing_ppmi[tag_idxs, idx])) if tag_idxs else 0.0

        # penalty if candidate clashes with the target
        pmi_penalty = min(float(pmi[target_idx, idx]), 0)

        scores[ing] = 0.5 * sim_score + 0.3 * context_score + 0.2 * tag_score + pmi_penalty

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

# ──────────────────────────────────────────────
# 6. EVALUATION
# ──────────────────────────────────────────────
random.seed(99)

# Collect recipes with at least 4 known ingredients
valid_recipes = []
for _, row in df.iterrows():
    known = [i for i in row["ingredients"] if i in ing2idx]
    if len(known) >= 4:
        valid_recipes.append(row)

eval_recipes = random.sample(valid_recipes, 300)

# ── Evaluation 1: Ingredient recovery ─────────────────────────────────────
# Hide one ingredient from each recipe and 
# score all vocab ingredients by how well they fit the remaining context 

# Metrics:
# Exact Hit@5 - did the missing ingredient appear in our top 5?
# Mean cos sim - how similar was our top prediction to the missing ingredient?

hits_at_5 = 0
total_similarity = 0.0

for row in eval_recipes:
    recipe = [i for i in row["ingredients"] if i in ing2idx]

    missing = random.choice(recipe)
    partial = [i for i in recipe if i != missing]
    partial_idxs = [ing2idx[i] for i in partial]
    missing_idx = ing2idx[missing]

    # Score every ingredient by how well it fits the context
    scores = {}
    for ing, idx in ing2idx.items():
        if ing in partial:
            continue
        scores[ing] = float(np.mean(ppmi[idx, partial_idxs]))

    top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_names = [t[0] for t in top5]

    if missing in top5_names:
        hits_at_5 += 1

    if top5_names:
        top_pred_idx = ing2idx[top5_names[0]]
        total_similarity += cos_sim[missing_idx, top_pred_idx]

print(f"  Exact Hit@5: {hits_at_5 / len(eval_recipes):.3f}")
print(f"  Mean cosine similarity: {total_similarity / len(eval_recipes):.3f}")

# ── Evaluation 2: Substitution quality ────────────────────────────────────
# Call get_substitutes directly and measure how similar the returned substitutes 
# are to the target ingredient using cosine similarity
total_sim = 0.0
count = 0

for row in eval_recipes:
    recipe = [i for i in row["ingredients"] if i in ing2idx]
    if len(recipe) < 2:
        continue

    target = random.choice(recipe)
    context = [i for i in recipe if i != target]
    target_idx = ing2idx[target]

    subs = get_substitutes(target, context, top_n=5)
    for sub, _ in subs:
        total_sim += cos_sim[target_idx, ing2idx[sub]]
        count += 1

print(f"  Mean cosine similarity (substitutes vs target): {total_sim / count:.3f}")

# ──────────────────────────────────────────────
# 7. DEMO
# ──────────────────────────────────────────────

demo_cases = [
    ("butter",      ["flour", "sugar", "eggs", "vanilla extract"], ["desserts", "cakes"]),
    ("soy sauce",   ["garlic", "ginger", "sesame oil", "rice"],    ["asian", "chinese"]),
    ("heavy cream", ["onion", "garlic", "pasta", "parmesan cheese"], ["italian", "pasta"]),
    ("eggs",        ["flour", "sugar", "butter", "baking powder"], ["baking", "desserts"]),
    ("olive oil",   ["garlic", "tomatoes", "basil", "pasta"],      ["italian", "mediterranean"]),
]

print("\nDEMO: Top-5 substitutes")
for target, context, tags in demo_cases:
    subs = get_substitutes(target, context, recipe_tags=tags, top_n=5)
    print(f"\nSubstitutes for '{target}' (tags: {tags}):")
    for ing, score in subs:
        print(f"  {ing} ({score:.4f})")


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
