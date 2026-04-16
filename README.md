# DS4420 Final Project
Tanishi Datta and Shruthi Palaniappan

## Project Overview

Cooking often requires adapting recipes when certain ingredients are unavailable or when substitutions are needed for dietary restrictions or personal preferences. This project explores whether patterns in recipe ingredient combinations can be used to identify potential food substitutions.

Using a large recipe dataset from Food.com (230,000+ recipes), we analyze how ingredients commonly appear together across thousands of recipes. By identifying patterns of ingredient co-occurrence, we aim to discover ingredients that play similar roles in recipes and could potentially serve as substitutes.

We apply two machine learning methods and package the results in an interactive Streamlit app.

---

## Models

### Model 1 — Item-Based Collaborative Filtering (Python)
**File:** `model1_collaborative_filtering.py`

A fully manual implementation (no sklearn or modeling packages).

1. **Data loading & cleaning** — Parses `RAW_recipes.csv` via regex (the file uses a non-standard quoted format), then applies a 5-step cleaning pipeline: drop null/empty lists, remove tab-garbled rows, strip/lowercase entries, require ≥3 ingredients per recipe, and deduplicate by ingredient set.
2. **Rating matrix** — Builds a binary recipe × ingredient matrix (1 if ingredient appears in recipe).
3. **PPMI embeddings** — Computes Positive PMI for all ingredient pairs from co-occurrence counts.
4. **Cosine similarity** — Derives a V × V cosine similarity matrix from the PPMI vectors.
5. **Substitution scoring** — Ranks candidates by a weighted score: `0.6 × cosine_similarity + 0.4 × mean_PPMI_context_fit`.
6. **Evaluation** — Leave-one-out proxy evaluation on 300 recipes; reports **Hit@5** and **MRR**.
7. **Exports** — Saves `cos_sim_top1000.npy`, `ppmi_top1000.npy`, and `vocab_top1000.txt` (top-1000 ingredients by frequency) for use by the Streamlit app and Model 2.

### Model 2 — Bayesian Logistic Regression (R)
**File:** `model2_bayesian.R`

Uses `brms`/Stan (pre-built package, allowed since Model 1 is the manual implementation).

1. **Loads** the pre-computed similarity matrices exported by Model 1 (`.csv` versions to avoid a `reticulate` dependency).
2. **Builds training data** — Samples ~5,250 (target, candidate) pairs. Features: z-scored cosine similarity, PPMI score, and context overlap. Label = 1 if candidate is in the top-20 PPMI neighbors of the target (same proxy ground-truth as Model 1).
3. **Fits Bayesian logistic regression** — `label ~ cos_sim_z + ppmi_z + ctx_overlap_z` with weakly-informative priors: `Normal(0, 2)` on the intercept and `Normal(0, 1)` on each slope (4 chains × 2000 iterations).
4. **Evaluation** — Reports in-sample accuracy, AUC (manual trapezoidal implementation), and LOO-CV via `loo()`.
5. **Plots** — Saves `plot_posterior_coefficients.png` (posterior densities) and `plot_predicted_probs.png` (predicted probability distributions by label).
6. **Prediction function** — `get_bayesian_substitutes()` ranks candidates by posterior mean predicted probability.
7. **Saves** the fitted model to `bayesian_model.rds`.

---

## Streamlit App (Extra Credit)
**File:** `app.py`

An interactive two-tab web app.

- **Tab 1 — Project Overview:** Problem description, approach summary, dataset stats, and references.
- **Tab 2 — Substitution Explorer:** Select a target ingredient and recipe context, then see the top substitutes ranked by the CF model's weighted score, visualized as a horizontal bar chart. Includes a score breakdown table and a PPMI context similarity heatmap.

**Run:**
```bash
pip install streamlit pandas numpy plotly
streamlit run app.py
```

> The app requires the precomputed data files. Run `model1_collaborative_filtering.py` first.

---

## Setup & Running Order

### Step 1 — Get the dataset (`RAW_recipes.csv`, ~300 MB)

**Option A — Download from Kaggle (no extra tools needed)**

1. Go to [Food.com Recipes on Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
2. Download and unzip the dataset
3. Place `RAW_recipes.csv` in the project root directory

**Option B — Git LFS (pulls automatically on clone)**

If the repo was cloned with Git LFS support, the file downloads automatically. You just need Git LFS installed:

```bash
# Install Git LFS (one-time setup)
brew install git-lfs      # Mac
# or: sudo apt install git-lfs  (Linux)

git lfs install

# If you already cloned without LFS, pull the large files:
git lfs pull
```

> If `RAW_recipes.csv` shows as a small text file (~130 bytes) after cloning, you are missing Git LFS — run the commands above.

---

### Steps 2–5

```bash
# 2. Run preprocessing (Python) — cleans RAW_recipes.csv, saves recipes_clean.pkl and recipes_clean.csv
pip install pandas numpy
python preprocess.py

# 3. Run Model 1 (Python) — requires recipes_clean.pkl, generates .npy / .csv / .txt artifacts
python model1_collaborative_filtering.py

# 4. Run Model 2 (R) — requires recipes_clean.csv and the artifacts from step 3
# In R:
# install.packages(c("brms", "dplyr", "ggplot2", "tidyr", "readr"))
# source("model2_bayesian.R")

# 5. Launch the Streamlit app
pip install streamlit plotly
streamlit run app.py
```

---

## Files

| File | Description |
|------|-------------|
| `preprocess.py` | Parses and cleans RAW_recipes.csv, outputs recipes_clean.pkl/.csv |
| `model1_collaborative_filtering.py` | Model 1: PPMI + cosine similarity CF model |
| `model2_bayesian.R` | Model 2: Bayesian logistic regression |
| `app.py` | Streamlit interactive app |
| `cos_sim_top1000.npy / .csv` | Cosine similarity matrix (top-1000 ingredients) |
| `ppmi_top1000.npy / .csv` | PPMI matrix (top-1000 ingredients) |
| `vocab_top1000.txt` | Vocabulary list (top-1000 ingredients) |

---

## Dataset

**Food.com Recipes** — [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- 230,000+ recipes
- ~1,500 unique ingredients after cleaning (≥20 occurrences)

---

## References

1. H. M., M. G., and P. S., "A Comprehensive Framework for Nutrition Analysis and Ingredient Substitution Using Machine Learning," *ICDSAAI 2025*, doi: 10.1109/ICDSAAI65575.2025.11011623.
2. E. Oz and F. Oz, "AI-Enabled Ingredient Substitution in Food Systems," *Foods*, vol. 14, 2025, doi: 10.3390/foods14223919.
3. P. Fermín Cueto et al., "Completing Partial Recipes Using Item-Based Collaborative Filtering," *arXiv:1907.11250*, 2020.
