# SwapSmart: Ingredient Substitution Using ML
**DS 4420 Final Project — Tanishi Datta & Shruthi Palaniappan**

Ever been mid-recipe and realize you're out of an ingredient? This project uses 230,000+ Food.com recipes to learn which ingredients play similar roles in cooking and can substitute for each other.

---

## Models

### Model 1 — Item-Based Collaborative Filtering (Python)
**File:** `model1_collaborative_filtering.py`

Manual implementation (no sklearn or modeling packages). Builds a binary recipe × ingredient matrix, computes PPMI embeddings, and ranks substitutes by a weighted score: `0.6 × cosine_similarity + 0.4 × context_fit`.

### Model 2 — Bayesian Logistic Regression (R)
**File:** `model2_bayesian.R`

Uses `brms`/Stan. Labels pairs using tag Jaccard similarity (independent from features). Features: z-scored cosine similarity, PPMI score, context overlap. Results: AUC 0.801, accuracy 90.0%.

---

## Streamlit App
**File:** `app.py` — deployed at [4420-project-swapsmart.streamlit.app](https://4420-project-swapsmart.streamlit.app)

Two tabs: project overview and an interactive substitute finder.

```bash
pip install streamlit pandas numpy
streamlit run app.py
```

---

## Running Order

```bash
# 1. Preprocess
pip install pandas numpy
python preprocess.py

# 2. Run Model 1 (generates .npy, .csv, and .txt files needed by app and Model 2)
python model1_collaborative_filtering.py

# 3. Run Model 2 (R)
# install.packages(c("brms", "dplyr", "ggplot2", "readr"))
# source("model2_bayesian.R")

# 4. Launch app
streamlit run app.py
```

---

## Files

| File | Description |
|------|-------------|
| `preprocess.py` | Cleans RAW_recipes.csv → recipes_clean.pkl/.csv |
| `model1_collaborative_filtering.py` | CF model, exports .npy and .csv artifacts |
| `model2_bayesian.R` | Bayesian logistic regression |
| `app.py` | Streamlit app |
| `requirements.txt` | Python dependencies for deployment |
| `cos_sim_top1000.npy` | Cosine similarity matrix — used by app |
| `cos_sim_top1000.csv` | Cosine similarity matrix — used by Model 2 (R) |
| `ppmi_top1000.npy` | PPMI matrix — used by app |
| `ppmi_top1000.csv` | PPMI matrix — used by Model 2 (R) |
| `vocab_top1000.txt` | Ingredient vocabulary |
| `recipes_clean.csv / .pkl` | Cleaned recipe data |
| `RAW_recipes.csv` | Raw Food.com dataset (~300 MB, from Kaggle) |

---

## Dataset

**Food.com Recipes** — [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- 230,000+ recipes, ~1,500 unique ingredients after cleaning

---

## References

1. H. M., M. G., and P. S., "A Comprehensive Framework for Nutrition Analysis and Ingredient Substitution Using Machine Learning," *ICDSAAI 2025*, doi: 10.1109/ICDSAAI65575.2025.11011623.
2. E. Oz and F. Oz, "AI-Enabled Ingredient Substitution in Food Systems," *Foods*, vol. 14, 2025, doi: 10.3390/foods14223919.
3. P. Fermín Cueto et al., "Completing Partial Recipes Using Item-Based Collaborative Filtering," *arXiv:1907.11250*, 2020.
