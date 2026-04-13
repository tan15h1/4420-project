 ══════════════════════════════════════════════════════════════════
# DS 4420 Final Project — Model 2: Bayesian Logistic Regression
# Tanishi Datta & Shruthi Palaniappan
#
# Goal: Estimate the probability that ingredient B is a good substitute
#       for ingredient A given the recipe context (other ingredients).
#
# Approach:
#   - For each (target, candidate) pair observed in the data, extract
#     features: co-occurrence frequency, cosine similarity (from PPMI),
#     and context overlap score.
#   - Label = 1 if candidate is in the top-20 PPMI neighbors of target
#     (same proxy ground-truth used in Model 1), else 0.
#   - Fit a Bayesian logistic regression with regularizing priors using brms.
#
# Package note: brms/rstan are pre-built modeling packages, which IS allowed
# for the R model — only ONE method must be manual (satisfied by Model 1 in Python).
# ══════════════════════════════════════════════════════════════════

library(brms)
library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)

set.seed(4420)

# ──────────────────────────────────────────────
# 1. LOAD PRE-COMPUTED SIMILARITY DATA
#    (exported by model1_collaborative_filtering.py)
# ──────────────────────────────────────────────

# Read vocabulary
vocab <- readLines("vocab_top1000.txt")
V     <- length(vocab)
ing2idx <- setNames(seq_along(vocab), vocab)

# Load cosine similarity and PPMI matrices (top-1000 ingredients).
# model1_collaborative_filtering.py exports both .npy and .csv versions;
# we use the .csv files here to avoid a reticulate dependency.
cos_mat  <- as.matrix(read.csv("cos_sim_top1000.csv",  header = FALSE))
ppmi_mat <- as.matrix(read.csv("ppmi_top1000.csv",     header = FALSE))

cat("Vocabulary size:", V, "\n")

# ──────────────────────────────────────────────
# 1b. DATA CLEANING (VERIFICATION)
#
# NOTE: The Bayesian model consumes the pre-computed similarity matrices
# exported by model1_collaborative_filtering.py, so you must run that
# Python script first. This section mirrors the Python cleaning pipeline
# on the raw CSV for completeness and to verify data quality in R.
# ──────────────────────────────────────────────

cat("Cleaning raw recipe data (verification)...\n")

# Read the raw file line-by-line.
# RAW_recipes.csv wraps each physical line in outer double-quotes; ingredient
# lists appear as ""['ing1', 'ing2', ...]"",N" near the end of each line.
raw_lines <- readLines("RAW_recipes.csv", encoding = "UTF-8", warn = FALSE)
cat("  Raw lines read:", length(raw_lines), "\n")

# Extract ingredient lists using the same pattern used in Python.
# Capture group 1: the Python list literal; group 2: n_ingredients count.
ing_pattern <- '"\\[([^\\]]+)\\]"",([0-9]+)"'
matches     <- regmatches(raw_lines, regexec(ing_pattern, raw_lines))

# Helper: parse a Python list literal string into a character vector.
# Extracts all single-quoted tokens, e.g. "['a', 'b']" -> c("a", "b").
parse_py_list <- function(s) {
  tokens <- regmatches(s, gregexpr("'[^']*'", s))[[1]]
  trimws(gsub("^'|'$", "", tokens))
}

recipes_raw <- Filter(Negate(is.null), lapply(matches, function(m) {
  if (length(m) < 3 || nchar(m[1]) == 0) return(NULL)
  ing_str <- m[2]          # e.g. "'butter', 'flour', 'sugar'"
  n_ing   <- as.integer(m[3])
  ings    <- parse_py_list(ing_str)
  if (length(ings) != n_ing || n_ing < 1 || n_ing > 100) return(NULL)
  list(ingredients = ings)
}))

recipes_df <- tibble(
  ingredients = lapply(recipes_raw, `[[`, "ingredients")
)
cat("  Records extracted:", nrow(recipes_df), "\n")

# 1. Drop rows where ingredient list is empty or NULL
recipes_df <- recipes_df %>%
  filter(!sapply(ingredients, is.null),
         sapply(ingredients, length) > 0)

# 2. Remove garbled rows: any ingredient containing a tab character.
#    Garbled entries look like "por\tk spare\tribs" or "fres\th ga\trl\tic".
has_tab <- function(ings) any(grepl("\t", ings))
recipes_df <- recipes_df %>%
  filter(!sapply(ingredients, has_tab))

# 3. Strip whitespace, lowercase, and drop empty / single-character entries
clean_ings <- function(ings) {
  ings <- trimws(tolower(ings))
  ings[nchar(ings) > 1]
}
recipes_df <- recipes_df %>%
  mutate(ingredients = lapply(ingredients, clean_ings))

# 4. Remove recipes with fewer than 3 usable ingredients after cleaning
recipes_df <- recipes_df %>%
  filter(sapply(ingredients, length) >= 3)

# 5. Deduplicate by ingredient set
recipes_df <- recipes_df %>%
  mutate(ing_key = sapply(ingredients, function(x) paste(sort(x), collapse = "|"))) %>%
  distinct(ing_key, .keep_all = TRUE) %>%
  select(-ing_key)

cat("  After cleaning:", nrow(recipes_df), "recipes\n\n")

# ──────────────────────────────────────────────
# 2. BUILD TRAINING DATASET
#    For a sample of (target, candidate) pairs, compute features
#    and assign a binary label.
# ──────────────────────────────────────────────

# Sample ingredient pairs (keep tractable: ~5000 pairs)
n_targets    <- 150
n_candidates <- 35
target_ings  <- sample(vocab, n_targets)

rows <- list()
for (t_ing in target_ings) {
  t_idx <- ing2idx[[t_ing]]
  
  # Top-20 PPMI neighbors = positive class (label = 1)
  sims     <- cos_mat[t_idx, ]
  sims[t_idx] <- -1
  top20    <- order(sims, decreasing = TRUE)[1:20]
  top20_set <- top20
  
  # Sample n_candidates candidates (mix of positives and negatives)
  cands <- sample(setdiff(seq_len(V), t_idx), n_candidates)
  
  for (c_idx in cands) {
    c_ing       <- vocab[c_idx]
    cos_score   <- cos_mat[t_idx, c_idx]
    ppmi_score  <- ppmi_mat[t_idx, c_idx]
    
    # Context overlap: mean PPMI between candidate and top-5 neighbors of target
    top5        <- order(cos_mat[t_idx, ], decreasing = TRUE)[2:6]
    ctx_overlap <- mean(ppmi_mat[c_idx, top5])
    
    label <- as.integer(c_idx %in% top20_set)
    
    rows[[length(rows) + 1]] <- data.frame(
      target     = t_ing,
      candidate  = c_ing,
      cos_sim    = cos_score,
      ppmi       = ppmi_score,
      ctx_overlap = ctx_overlap,
      label      = label,
      stringsAsFactors = FALSE
    )
  }
}

train_df <- bind_rows(rows)

cat("Training pairs:", nrow(train_df), "\n")
cat("Positive rate:", mean(train_df$label), "\n")

# ──────────────────────────────────────────────
# 2b. TRAIN_DF CLEANING
# ──────────────────────────────────────────────

n_pairs <- nrow(train_df)

# 1. Drop rows with NA in any feature column
train_df <- train_df %>%
  filter(!is.na(cos_sim), !is.na(ppmi), !is.na(ctx_overlap))
cat(sprintf("  Step 1 – drop NA feature rows:        removed %5d, %d remaining\n",
            n_pairs - nrow(train_df), nrow(train_df)))
n_pairs <- nrow(train_df)

# 2. Remove duplicate (target, candidate) pairs
train_df <- train_df %>%
  distinct(target, candidate, .keep_all = TRUE)
cat(sprintf("  Step 2 – drop duplicate pairs:        removed %5d, %d remaining\n",
            n_pairs - nrow(train_df), nrow(train_df)))

cat(sprintf("  ── After cleaning: %d training pairs (positive rate: %.3f) ──\n",
            nrow(train_df), mean(train_df$label)))

# ──────────────────────────────────────────────
# 3. FEATURE SCALING
# ──────────────────────────────────────────────

train_df <- train_df %>%
  mutate(
    cos_sim_z     = scale(cos_sim)[,1],
    ppmi_z        = scale(ppmi)[,1],
    ctx_overlap_z = scale(ctx_overlap)[,1]
  ) %>%
  select(-cos_sim, -ppmi, -ctx_overlap)  # drop raw columns; scaled versions are used for modeling

# ──────────────────────────────────────────────
# 4. BAYESIAN LOGISTIC REGRESSION WITH brms
#
# Model: label ~ cos_sim + ppmi + ctx_overlap
#
# Priors:
#   - Intercept: Normal(0, 2)   — weakly informative, centered near 0 probability
#   - Slopes:    Normal(0, 1)   — regularizing; discourages extreme coefficients
#
# These are standard weakly-informative priors for logistic regression
# (see Gelman et al., 2008; Stan Prior Choice Recommendations).
# ──────────────────────────────────────────────

priors <- c(
  prior(normal(0, 2), class = Intercept),
  prior(normal(0, 1), class = b, coef = cos_sim_z),
  prior(normal(0, 1), class = b, coef = ppmi_z),
  prior(normal(0, 1), class = b, coef = ctx_overlap_z)
)

cat("\nFitting Bayesian logistic regression...\n")

bayes_model <- brm(
  formula = label ~ cos_sim_z + ppmi_z + ctx_overlap_z,
  data    = train_df,
  family  = bernoulli(link = "logit"),
  prior   = priors,
  chains  = 4,
  iter    = 2000,
  warmup  = 1000,
  cores   = 4,
  seed    = 4420,
  refresh = 500
)

# ──────────────────────────────────────────────
# 5. RESULTS SUMMARY
# ──────────────────────────────────────────────

cat("\n=== Posterior Summary ===\n")
print(summary(bayes_model))

# Posterior draws for coefficients
post <- as_draws_df(bayes_model,
                    variable = c("b_Intercept",
                                 "b_cos_sim_z",
                                 "b_ppmi_z",
                                 "b_ctx_overlap_z"))

cat("\n=== 95% Credible Intervals (log-odds scale) ===\n")
for (param in colnames(post)) {
  q <- quantile(post[[param]], c(0.025, 0.5, 0.975))
  cat(sprintf("  %-25s  median=%.3f  95%% CI=[%.3f, %.3f]\n",
              param, q[2], q[1], q[3]))
}

# ──────────────────────────────────────────────
# 6. IN-SAMPLE POSTERIOR PREDICTIVE CHECK
# ──────────────────────────────────────────────

# Predicted probability for each observation
train_df$pred_prob <- fitted(bayes_model, type = "response")[, "Estimate"]

# Convert to binary prediction at 0.5 threshold
train_df$pred_label <- as.integer(train_df$pred_prob >= 0.5)

accuracy <- mean(train_df$pred_label == train_df$label)
cat(sprintf("\nIn-sample accuracy (threshold=0.5): %.3f\n", accuracy))

# AUC (manual implementation — no pROC package needed)
compute_auc <- function(labels, scores) {
  ord     <- order(scores, decreasing = TRUE)
  labels  <- labels[ord]
  n_pos   <- sum(labels)
  n_neg   <- length(labels) - n_pos
  tp <- cumsum(labels)
  fp <- cumsum(1 - labels)
  tpr <- tp / n_pos
  fpr <- fp / n_neg
  # Trapezoidal integration
  sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1))) / 2
}

auc_val <- compute_auc(train_df$label, train_df$pred_prob)
cat(sprintf("In-sample AUC: %.3f\n", auc_val))

# ──────────────────────────────────────────────
# 7. LEAVE-ONE-OUT CROSS-VALIDATION (LOO-CV)
#    via brms/loo — standard Bayesian model comparison metric
# ──────────────────────────────────────────────

cat("\nComputing LOO-CV...\n")
loo_result <- loo(bayes_model)
print(loo_result)

# ──────────────────────────────────────────────
# 8. PLOTS FOR REPORT
# ──────────────────────────────────────────────

# Plot 1: Posterior distributions of coefficients
post_long <- post %>%
  pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
  mutate(parameter = recode(parameter,
    "b_Intercept"     = "Intercept",
    "b_cos_sim_z"     = "Cosine Similarity",
    "b_ppmi_z"        = "PPMI Score",
    "b_ctx_overlap_z" = "Context Overlap"
  ))

p1 <- ggplot(post_long, aes(x = value, fill = parameter)) +
  geom_density(alpha = 0.7, color = NA) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  facet_wrap(~parameter, scales = "free", ncol = 2) +
  scale_fill_manual(values = c("#E07A5F", "#3D405B", "#81B29A", "#F2CC8F")) +
  labs(
    title    = "Posterior Distributions of Model Coefficients",
    subtitle = "Bayesian Logistic Regression — Ingredient Substitution",
    x        = "Log-Odds",
    y        = "Density"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold"))

ggsave("plot_posterior_coefficients.png", p1, width = 8, height = 5, dpi = 150)
cat("Saved: plot_posterior_coefficients.png\n")

# Plot 2: Predicted probability vs. label
p2 <- ggplot(train_df, aes(x = pred_prob, fill = factor(label))) +
  geom_histogram(bins = 40, alpha = 0.75, position = "identity") +
  scale_fill_manual(values = c("#E07A5F", "#81B29A"),
                    labels = c("Non-substitute", "Near-neighbor substitute"),
                    name   = "Label") +
  labs(
    title    = "Posterior Predicted Probabilities by Label",
    subtitle = "Higher scores → model believes candidate is a good substitute",
    x        = "Predicted Probability of Being a Substitute",
    y        = "Count"
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("plot_predicted_probs.png", p2, width = 8, height = 4.5, dpi = 150)
cat("Saved: plot_predicted_probs.png\n")

# ──────────────────────────────────────────────
# 9. PREDICTION FUNCTION
#    Given a target ingredient + context, rank candidates
#    by posterior mean predicted probability.
# ──────────────────────────────────────────────

get_bayesian_substitutes <- function(target_ing, context_ings, top_n = 5) {
  if (!(target_ing %in% vocab)) {
    cat("Ingredient not in vocabulary:", target_ing, "\n")
    return(NULL)
  }
  
  t_idx       <- ing2idx[[target_ing]]
  ctx_idxs    <- ing2idx[context_ings[context_ings %in% vocab]]
  top5_t      <- order(cos_mat[t_idx, ], decreasing = TRUE)[2:6]
  
  candidates <- setdiff(vocab, c(target_ing, context_ings))
  
  feat_df <- data.frame(
    candidate   = candidates,
    cos_sim_z   = scale(cos_mat[t_idx, ing2idx[candidates]])[,1],
    ppmi_z      = scale(ppmi_mat[t_idx, ing2idx[candidates]])[,1],
    ctx_overlap_z = scale(sapply(ing2idx[candidates], function(ci)
                             mean(ppmi_mat[ci, top5_t])))[,1]
  )
  
  feat_df$prob <- predict(bayes_model,
                           newdata = feat_df,
                           type    = "response")[, "Estimate"]
  
  feat_df %>%
    arrange(desc(prob)) %>%
    head(top_n) %>%
    select(candidate, prob)
}

# Demo predictions
demo_cases <- list(
  list(target = "butter",       ctx = c("flour", "sugar", "eggs", "vanilla extract")),
  list(target = "soy sauce",    ctx = c("garlic", "ginger", "sesame oil", "rice")),
  list(target = "heavy cream",  ctx = c("onion", "garlic", "pasta", "parmesan cheese"))
)

cat("\n=== Bayesian Model: Top-5 Substitutes ===\n")
for (case in demo_cases) {
  cat(sprintf("\nSubstitutes for '%s':\n", case$target))
  result <- get_bayesian_substitutes(case$target, case$ctx, top_n = 5)
  if (!is.null(result)) print(result)
}

# Save model
saveRDS(bayes_model, "bayesian_model.rds")
cat("\nModel saved to bayesian_model.rds\n")
cat("Done.\n")
