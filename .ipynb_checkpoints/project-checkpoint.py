import pandas as pd
import ast
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import random

df = pd.read_csv("RAW_recipes.csv")
df["ingredients"] = df["ingredients"].apply(ast.literal_eval)
df["ingredients"] = df["ingredients"].apply(lambda x: [i.strip().lower() for i in x])
df = df.sample(n=min(20_000, len(df)), random_state=42).reset_index(drop=True)

ingredient_counts = Counter(ing for recipe in df["ingredients"] for ing in recipe)
vocab = [ing for ing, cnt in ingredient_counts.items() if cnt >= 20]
ing2idx = {ing: idx for idx, ing in enumerate(vocab)}
V = len(vocab)

co_occ = np.zeros((V, V), dtype=np.float32)
for recipe in df["ingredients"]:
    idxs = [ing2idx[i] for i in recipe if i in ing2idx]
    for a in idxs:
        for b in idxs:
            if a != b:
                co_occ[a, b] += 1

total_pairs = co_occ.sum()
marginal = co_occ.sum(axis=1)
pmi_matrix = np.zeros((V, V), dtype=np.float32)
for i in range(V):
    for j in range(V):
        if co_occ[i, j] > 0:
            p_ij = co_occ[i, j] / total_pairs
            p_i  = marginal[i] / total_pairs
            p_j  = marginal[j] / total_pairs
            pmi_matrix[i, j] = max(np.log(p_ij / (p_i * p_j)), 0)

cos_sim = cosine_similarity(pmi_matrix)


def get_substitutes(target_ingredient, context, top_n=5):
    if target_ingredient not in ing2idx:
        return []
    target_idx = ing2idx[target_ingredient]
    context_idxs = [ing2idx[i] for i in context if i in ing2idx]
    scores = {}
    for ing, idx in ing2idx.items():
        if ing == target_ingredient or ing in context:
            continue
        sim_score = cos_sim[target_idx, idx]
        context_score = np.mean([pmi_matrix[idx, c] for c in context_idxs]) if context_idxs else 0
        scores[ing] = 0.6 * sim_score + 0.4 * context_score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


hits_at_5_neighbors = 0
mrr_total = 0.0
eval_count = 0

eval_recipes = df[df["ingredients"].apply(
    lambda x: sum(1 for i in x if i in ing2idx) >= 4
)].sample(n=200, random_state=99)

for _, row in eval_recipes.iterrows():
    recipe = [i for i in row["ingredients"] if i in ing2idx]
    if len(recipe) < 4:
        continue
    missing = random.choice(recipe)
    partial = [i for i in recipe if i != missing]
    missing_idx = ing2idx[missing]
    sim_scores = cos_sim[missing_idx].copy()
    sim_scores[missing_idx] = -1
    top20_neighbors = set(vocab[i] for i in np.argsort(sim_scores)[-20:])
    preds = [p[0] for p in get_substitutes(missing, partial, top_n=5)]
    if any(p in top20_neighbors for p in preds):
        hits_at_5_neighbors += 1
    for rank, p in enumerate(preds, start=1):
        if p in top20_neighbors:
            mrr_total += 1.0 / rank
            break
    eval_count += 1

print(f"Hit@5: {hits_at_5_neighbors/eval_count:.3f}")
print(f"MRR:   {mrr_total/eval_count:.3f}")

demo_cases = [
    ("butter",      ["flour", "sugar", "eggs", "vanilla extract"]),
    ("soy sauce",   ["garlic", "ginger", "sesame oil", "rice"]),
    ("heavy cream", ["onion", "garlic", "pasta", "parmesan cheese"]),
]

for target, context in demo_cases:
    subs = get_substitutes(target, context, top_n=5)
    print(f"\nSubstitutes for '{target}':")
    for ing, score in subs:
        print(f"  {ing:<30} {score:.4f}")