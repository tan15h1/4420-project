"""
DS 4420 Final Project
Model 1: Item-Item-Based Collaborative Filtering
Tanishi Datta & Shruthi Palaniappan

Uses a recipe x ingredient rating matrix with PMI and cosine similarity
to recommend ingredient substitutes given a partial recipe.
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────
df = pd.read_pickle("recipes_clean.pkl")

# Sample down from ~170,000 to 25,000 recipes
df = df.sample(n=min(25_000, len(df)), random_state=42).reset_index(drop=True)

# Build vocabulary - keep ingredients that appear in at least 1 recipe
ingredient_counts = Counter()
for recipe in df["ingredients"]:
    for i in recipe:
        ingredient_counts[i] += 1

vocab = [i for i, cnt in ingredient_counts.items() if cnt >= 1]
ing_idx = {i: idx for idx, i in enumerate(vocab)}
v = len(vocab)
print(f"Ingredients in vocab: {v}")
print(f"Recipes: {len(df)}")

# ──────────────────────────────────────────────
# 2. BUILD RECIPE x INGREDIENT MATRIX
# ──────────────────────────────────────────────
# Binary: M[recipe_i, ing_j] = 1 if ingredient j appears in recipe i

M = np.zeros((len(df), v), dtype=np.float32)
for row_idx, row in df.iterrows():
    for ingredient in row["ingredients"]:
        if ingredient in ing_idx:
            M[row_idx, ing_idx[ingredient]] = 1.0

print(f"Matrix M shape: {M.shape}  (recipes x ingredients)")

# ──────────────────────────────────────────────
# 3. Pointwise Mutual Information (PMI) SIMILARITY
# ──────────────────────────────────────────────
# PMI(i, i) = log[ P(i,i) / (P(i) * P(i)) ]
n = len(df)

# How often each ingredient appears across all recipes
freq = M.sum(axis=0)
p_i = freq / n
print(f"p_i shape (ingredient marginals vector): {p_i.shape}")

# How often each pair of ingredients appears together
# M.T @ M gives a v x v matrix
co_occ = M.T @ M
p_ii = co_occ / n
print(f"p_ii shape (co-occurrence matrix): {p_ii.shape}")

pmi = np.log((p_ii + 1e-10) / (np.outer(p_i, p_i) + 1e-10))
print(f"pmi shape: {pmi.shape}")

# Keep only positive pmi values
ppmi = np.maximum(pmi, 0)

# TF-IDF: reduce weight of columns of ppmi that correspond to common ingredients
# ppmi[i, j] * idf[j] means ingredient i's association with j is scaled by how rare j is 
idf = np.log(n / (freq + 1e-10))
ppmi = ppmi * idf
print(f"ppmi shape: {ppmi.shape}")

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
print(f"cosine similarity matrix shape: {cos_sim.shape}")

# ──────────────────────────────────────────────
# 5. TAG-INGREDIENT co-occurrence matrix
# Is not actually used because it had no effect 
# ──────────────────────────────────────────────
# Build tag vocabulary - keep tags that appear in at least 20 recipes
tag_counts = Counter()
for tags in df["tags"]:
    for tag in tags:
        tag_counts[tag] += 1

tag_vocab = [t for t, cnt in tag_counts.items() if cnt >= 20]
tag_idx = {t: idx for idx, t in enumerate(tag_vocab)}
t = len(tag_vocab)
print(f"Tag vocabulary: {t} tags")

# tag_ing[t, i] = number of recipes that have both tag t and ingredient i
tag_ing = np.zeros((t, v), dtype=np.float32)
for _, row in df.iterrows():
    for tag in row["tags"]:
        if tag in tag_idx:
            for ingredients in row["ingredients"]:
                if ingredients in ing_idx:
                    tag_ing[tag_idx[tag], ing_idx[ingredients]] += 1

# Compute tag-ingredient PPMI
p_t = tag_ing.sum(axis=1) / len(df)
print(f"p_t shape (tag marginals vector): {p_t.shape}")

p_ti = tag_ing / len(df)
print(f"p_ti shape (tag-ingredient joint probability matrix): {p_ti.shape}")

tag_ing_pmi = np.log((p_ti + 1e-10) / (np.outer(p_t, p_i) + 1e-10))
tag_ing_ppmi = np.maximum(tag_ing_pmi, 0)
print(f"tag_ing_ppmi shape: {tag_ing_ppmi.shape}")

# ──────────────────────────────────────────────
# 6. INGREDIENT SUBSTITUTION FUNCTION
# ──────────────────────────────────────────────
# min-max scale scores
def normalize(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]


def get_substitutes(target_ingredient, context_ingredients, recipe_tags=None, top_n=5):
    if target_ingredient not in ing_idx:
        print(f"'{target_ingredient}' not in vocabulary.")
        return []

    target_idx = ing_idx[target_ingredient]
    context_idxs = [ing_idx[i] for i in context_ingredients if i in ing_idx]

    # collect raw scores for every candidate in one pass
    candidates = []
    for ingredient, idx in ing_idx.items():
        # skip if ingredient is target or in context 
        if ingredient == target_ingredient or ingredient in context_ingredients:
            continue

        # how similar is the candidate to the target ingredient
        sim_score = float(cos_sim[target_idx, idx])
        # how well does the candidate fit the rest of the recipe (context)
        context_score = float(np.mean(ppmi[idx, context_idxs])) if context_idxs else 0.0
        # penalty if candidate clashes with the target
        pmi_penalty = min(float(pmi[target_idx, idx]), 0)

        candidates.append((ingredient, sim_score, context_score, pmi_penalty))

    # normalize each score to [0, 1] so weights control importance
    sim_norm = normalize([c[1] for c in candidates])
    context_norm = normalize([c[2] for c in candidates])

    scores = {}
    for i, (ingredient, _, _, pmi_penalty) in enumerate(candidates):
        scores[ingredient] = (0.6 * sim_norm[i] + 0.4 * context_norm[i] + pmi_penalty)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # filter out garbled ingredient names
    clean = [(ing, score) for ing, score in ranked if '"' not in ing and ',' not in ing]
    return clean[:top_n]

# ──────────────────────────────────────────────
# 7. EVALUATION
# ──────────────────────────────────────────────
random.seed(99)

# Collect recipes with at least 4 known ingredients
valid_recipes = []
for _, row in df.iterrows():
    known = [i for i in row["ingredients"] if i in ing_idx]
    if len(known) >= 4:
        valid_recipes.append(row)

eval_recipes = random.sample(valid_recipes, 300)
n_eval = len(eval_recipes)

# Pre-compute splits once so all three evals hide the same ingredient
eval_splits = []
for row in eval_recipes:
    recipe       = [i for i in row["ingredients"] if i in ing_idx]
    missing      = random.choice(recipe)
    partial      = [i for i in recipe if i != missing]
    partial_idxs = [ing_idx[i] for i in partial]
    eval_splits.append((missing, partial, partial_idxs))


# run hit@k and mrr given a dict of scores and the missing ingredient
def eval_metrics(scores, missing):
    ranked_names = [ing for ing, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    if missing not in ranked_names:
        return 0, 0, 0, 0.0
    rank = ranked_names.index(missing) + 1 
    hit1  = 1 if rank == 1  else 0
    hit5  = 1 if rank <= 5  else 0
    hit10 = 1 if rank <= 10 else 0
    rr    = 1.0 / rank
    return hit1, hit5, hit10, rr

# ── Evaluation 1: PPMI context score only ─────────────────────────────────
# Take out a random ingredient in recipe and evaluate if the model can predict it
# Using PPMI context scores 

print("\n Eval 1: PPMI context score")
hits_at_1 = hits_at_5 = hits_at_10 = 0
mrr = 0.0

for missing, partial, partial_idxs in eval_splits:
    scores = {}
    for ing, idx in ing_idx.items():
        if ing in partial:
            continue
        scores[ing] = float(np.mean(ppmi[idx, partial_idxs]))

    h1, h5, h10, rr = eval_metrics(scores, missing)
    hits_at_1 += h1
    hits_at_5 += h5
    hits_at_10 += h10
    mrr += rr

e1 = [hits_at_1 / n_eval, hits_at_5 / n_eval, hits_at_10 / n_eval, mrr / n_eval]

print(f"  Hit@1 : {hits_at_1  / n_eval:.3f}")
print(f"  Hit@5 : {hits_at_5  / n_eval:.3f}")
print(f"  Hit@10: {hits_at_10 / n_eval:.3f}")
print(f"  MRR   : {mrr / n_eval:.3f}")


# ── Evaluation 2: Cosine similarity score only ────────────────────────────
# Take out a random ingredient in recipe and evaluate if the model can predict it
# Using cosine similarity scores 

print("\n Eval 2: Cosine Similarity score")
hits_at_1 = hits_at_5 = hits_at_10 = 0
mrr = 0.0

for missing, partial, partial_idxs in eval_splits:
    scores = {}
    for ing, idx in ing_idx.items():
        if ing in partial:
            continue
        scores[ing] = float(np.mean(cos_sim[idx, partial_idxs]))

    h1, h5, h10, rr = eval_metrics(scores, missing)
    hits_at_1 += h1
    hits_at_5 += h5
    hits_at_10 += h10
    mrr += rr

e2 = [hits_at_1 / n_eval, hits_at_5 / n_eval, hits_at_10 / n_eval, mrr / n_eval]

print(f"  Hit@1 : {hits_at_1  / n_eval:.3f}")
print(f"  Hit@5 : {hits_at_5  / n_eval:.3f}")
print(f"  Hit@10: {hits_at_10 / n_eval:.3f}")
print(f"  MRR   : {mrr / n_eval:.3f}")


# ── Evaluation 3: Full model (PPMI + cosine sim - penalty) ────────────────
# Take out a random ingredient in recipe and evaluate if the model can predict it
# Using PPMI + cosine sim + penalty scores

print("\n Eval 3: Full model")
hits_at_1 = hits_at_5 = hits_at_10 = 0
mrr = 0.0

for missing, partial, partial_idxs in eval_splits:
    target_idx = ing_idx[missing]
    candidates = []
    for ing, idx in ing_idx.items():
        if ing in partial:
            continue
        sim_score     = float(np.mean(cos_sim[idx, partial_idxs]))
        context_score = float(np.mean(ppmi[idx, partial_idxs]))
        pmi_penalty   = min(float(pmi[target_idx, idx]), 0)
        candidates.append((ing, sim_score, context_score, pmi_penalty))

    sim_norm     = normalize([c[1] for c in candidates])
    context_norm = normalize([c[2] for c in candidates])

    scores = {}
    for i, (ing, _, _, pmi_penalty) in enumerate(candidates):
        scores[ing] = 0.6 * sim_norm[i] + 0.4 * context_norm[i] + pmi_penalty

    h1, h5, h10, rr = eval_metrics(scores, missing)
    hits_at_1 += h1
    hits_at_5 += h5
    hits_at_10 += h10
    mrr += rr

e3 = [hits_at_1 / n_eval, hits_at_5 / n_eval, hits_at_10 / n_eval, mrr / n_eval]

print(f"  Hit@1 : {hits_at_1  / n_eval:.3f}")
print(f"  Hit@5 : {hits_at_5  / n_eval:.3f}")
print(f"  Hit@10: {hits_at_10 / n_eval:.3f}")
print(f"  MRR   : {mrr / n_eval:.3f}")


# ── Evaluation chart ────────────────────────────────────────────────────────
metrics = ["Hit@1", "Hit@5", "Hit@10", "MRR"]
x = np.arange(len(metrics))

fig, ax = plt.subplots()
bars1 = ax.bar(x - 0.25, e1, 0.25, label="PPMI")
bars2 = ax.bar(x, e2, 0.25, label="Cosine")
bars3 = ax.bar(x + 0.25, e3, 0.25, label="Full Model")

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom")
        
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title("Evaluating the Model: PPMI vs Cosine vs Full Model")
ax.legend()

plt.tight_layout()
plt.savefig("evaluation_chart.png")

# ──────────────────────────────────────────────
# 8. DEMO
# ──────────────────────────────────────────────
demo_cases = [
    ("butter",      ["flour", "sugar", "eggs", "vanilla extract"], ["desserts", "cakes", "sweets"]),
    ("soy sauce",   ["garlic", "ginger", "sesame oil", "rice"],    ["asian", "chinese"]),
    ("heavy cream", ["onion", "garlic", "pasta", "parmesan cheese"], ["italian", "pasta"]),
    ("eggs",        ["flour", "sugar", "butter", "baking powder"], ["baking", "desserts"]),
    ("olive oil",   ["garlic", "tomatoes", "basil", "pasta"],      ["italian"]),
]

print("\nDEMO: Top-5 substitutes")
for target, context, tags in demo_cases:
    subs = get_substitutes(target, context, recipe_tags=tags, top_n=5)
    print(f"\nSubstitutes for '{target}' (tags: {tags}):")
    for ing, score in subs:
        print(f"  {ing} ({score:.3f})")

# ── Demo table ────────────────────────────────────────────────────────────
table_data = []
for target, context, tags in demo_cases:
    subs = get_substitutes(target, context, recipe_tags=tags, top_n=3)
    sub_names = [ing.title() for ing, _ in subs]
    table_data.append([target.title()] + sub_names)

col_labels = ["Ingredient", "Substitute 1", "Substitute 2", "Substitute 3"]

fig, ax = plt.subplots()
ax.axis("off")

table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
table.auto_set_font_size(True)
table.scale(1, 2)

for col in range(len(col_labels)):
    table[0, col].set_facecolor("#4C72B0")
    table[0, col].set_text_props(color="white", fontweight="bold")

for row in range(1, len(table_data) + 1):
    table[row, 0].set_facecolor("#4C72B0")
    table[row, 0].set_text_props(fontweight="bold")

plt.tight_layout()
plt.savefig("demo_table.png", dpi=150, bbox_inches="tight")


# ──────────────────────────────────────────────
# 9. SAVE DATA FOR STREAMLIT APP + SUMMARY
# ──────────────────────────────────────────────

top1000_ings = [ing for ing, _ in ingredient_counts.most_common(1000) if ing in ing_idx]
top1000_idxs = [ing_idx[i] for i in top1000_ings]
cos_sim_top  = cos_sim[np.ix_(top1000_idxs, top1000_idxs)]
ppmi_top     = ppmi[np.ix_(top1000_idxs, top1000_idxs)]

np.save("cos_sim_top1000.npy", cos_sim_top)
np.save("ppmi_top1000.npy",    ppmi_top)
pd.DataFrame(cos_sim_top).to_csv("cos_sim_top1000.csv", index=False, header=False)
pd.DataFrame(ppmi_top).to_csv("ppmi_top1000.csv",       index=False, header=False)
with open("vocab_top1000.txt", "w") as f:
    for ing in top1000_ings:
        f.write(ing + "\n")
print("\nSaved: cos_sim_top1000.npy/.csv, ppmi_top1000.npy/.csv, vocab_top1000.txt")

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
