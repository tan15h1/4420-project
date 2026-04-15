# DS 4420 Final Project - Model 2: Bayesian Logistic Regression
# Tanishi Datta & Shruthi Palaniappan
#
# Fits a Bayesian logistic regression to predict whether ingredient B
# is a good substitute for ingredient A given recipe context.
# Requires outputs from model1_collaborative_filtering.py.

library(brms)
library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)

set.seed(4420)

# ──────────────────────────────────────────────
# 1. LOAD PRECOMPUTED DATA
# ──────────────────────────────────────────────

vocab   <- readLines("vocab_top1000.txt")
V       <- length(vocab)
ing2idx <- setNames(seq_along(vocab), vocab)

cos_mat  <- as.matrix(read.csv("cos_sim_top1000.csv",  header = FALSE))
ppmi_mat <- as.matrix(read.csv("ppmi_top1000.csv",     header = FALSE))

cat("Vocabulary size:", V, "\n")

# ──────────────────────────────────────────────
# 2. LOAD CLEANED DATA
# ──────────────────────────────────────────────
# preprocess.py already cleaned RAW_recipes.csv and saved recipes_clean.csv
# ingredients and tags are pipe-separated strings, split them back into lists

recipes_df <- read.csv("recipes_clean.csv", stringsAsFactors = FALSE)
recipes_df$ingredients <- strsplit(recipes_df$ingredients, "\\|")
recipes_df$tags        <- strsplit(recipes_df$tags, "\\|")

cat("Recipes loaded:", nrow(recipes_df), "\n")

# ──────────────────────────────────────────────
# 3. BUILD TRAINING PAIRS
# ──────────────────────────────────────────────
# Features per (target, candidate) pair:
#   cos_sim     - cosine similarity between target and candidate
#   ppmi        - PPMI score between target and candidate
#   ctx_overlap - mean PPMI between candidate and target's top-5 neighbors
# Label = 1 if candidate is in the top-20 PPMI neighbors of target

n_targets    <- 150
n_candidates <- 35
targets      <- sample(vocab, n_targets)

rows <- list()
for (t in targets) {
  t_idx <- ing2idx[[t]]
  sims  <- cos_mat[t_idx, ]
  sims[t_idx] <- -1
  top20 <- order(sims, decreasing = TRUE)[1:20]
  top5  <- order(cos_mat[t_idx, ], decreasing = TRUE)[2:6]
  cands <- sample(setdiff(seq_len(V), t_idx), n_candidates)

  for (c_idx in cands) {
    rows[[length(rows) + 1]] <- data.frame(
      target      = t,
      candidate   = vocab[c_idx],
      cos_sim     = cos_mat[t_idx, c_idx],
      ppmi        = ppmi_mat[t_idx, c_idx],
      ctx_overlap = mean(ppmi_mat[c_idx, top5]),
      label       = as.integer(c_idx %in% top20),
      stringsAsFactors = FALSE
    )
  }
}

train_df <- bind_rows(rows) %>%
  filter(!is.na(cos_sim), !is.na(ppmi), !is.na(ctx_overlap)) %>%
  distinct(target, candidate, .keep_all = TRUE)

cat("Training pairs:", nrow(train_df), "\n")
cat("Positive rate:", mean(train_df$label), "\n")

# ──────────────────────────────────────────────
# 4. FEATURE SCALING
# ──────────────────────────────────────────────

train_df <- train_df %>%
  mutate(
    cos_sim_z     = scale(cos_sim)[,1],
    ppmi_z        = scale(ppmi)[,1],
    ctx_overlap_z = scale(ctx_overlap)[,1]
  ) %>%
  select(-cos_sim, -ppmi, -ctx_overlap)

# ──────────────────────────────────────────────
# 5. BAYESIAN LOGISTIC REGRESSION
# ──────────────────────────────────────────────

# Check defaults first
default_prior(label ~ cos_sim_z + ppmi_z + ctx_overlap_z,
              data   = train_df,
              family = bernoulli("logit"))

manual_prior <- c(
  prior(student_t(4, 0, 10), class = "Intercept"),
  prior(normal(0, 2), class = b, coef = cos_sim_z),
  prior(normal(0, 2), class = b, coef = ppmi_z),
  prior(normal(0, 2), class = b, coef = ctx_overlap_z)
)

cat("\nFitting Bayesian logistic regression...\n")

model <- brm(
  label ~ cos_sim_z + ppmi_z + ctx_overlap_z,
  family  = bernoulli("logit"),
  data    = train_df,
  chains  = 4,
  iter    = 1000,
  warmup  = 180,
  cores   = getOption("mc.cores", 1),
  prior   = manual_prior,
  seed    = 4420
)

# ──────────────────────────────────────────────
# 6. EVALUATION
# ──────────────────────────────────────────────

summary(model)
plot(model) # posterior distributions + chains for convergence

post_draws <- as.data.frame(model)

cat("\n=== 95% Credible Intervals ===\n")
for (col in c("b_Intercept", "b_cos_sim_z", "b_ppmi_z", "b_ctx_overlap_z")) {
  q <- quantile(post_draws[[col]], c(0.025, 0.5, 0.975))
  cat(sprintf("  %-25s  median=%.3f  95%% CI=[%.3f, %.3f]\n", col, q[2], q[1], q[3]))
}

# In-sample posterior predictive check
post_preds <- posterior_predict(model)
pred_prob  <- colMeans(post_preds)
pred_label <- as.integer(pred_prob >= 0.5)
cat(sprintf("\nIn-sample accuracy (threshold=0.5): %.3f\n", mean(pred_label == train_df$label)))

# AUC via trapezoidal integration
compute_auc <- function(labels, scores) {
  ord    <- order(scores, decreasing = TRUE)
  labels <- labels[ord]
  n_pos  <- sum(labels)
  n_neg  <- length(labels) - n_pos
  tpr    <- cumsum(labels) / n_pos
  fpr    <- cumsum(1 - labels) / n_neg
  sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1))) / 2
}

cat(sprintf("In-sample AUC: %.3f\n", compute_auc(train_df$label, pred_prob)))

cat("\nComputing LOO-CV...\n")
print(loo(model))

# Visualize how each feature affects substitution probability
plot(conditional_effects(model, "cos_sim_z"))
plot(conditional_effects(model, "ppmi_z"))
plot(conditional_effects(model, "ctx_overlap_z"))

# ──────────────────────────────────────────────
# 7. PLOTS
# ──────────────────────────────────────────────

post_long <- post_draws %>%
  select(b_Intercept, b_cos_sim_z, b_ppmi_z, b_ctx_overlap_z) %>%
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
  labs(title = "Posterior Distributions of Model Coefficients",
       x = "Log-Odds", y = "Density") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none", plot.title = element_text(face = "bold"))

ggsave("plot_posterior_coefficients.png", p1, width = 8, height = 5, dpi = 150)
cat("Saved: plot_posterior_coefficients.png\n")

train_df$pred_prob <- pred_prob
p2 <- ggplot(train_df, aes(x = pred_prob, fill = factor(label))) +
  geom_histogram(bins = 40, alpha = 0.75, position = "identity") +
  scale_fill_manual(values = c("#E07A5F", "#81B29A"),
                    labels = c("Non-substitute", "Near-neighbor substitute"),
                    name = "Label") +
  labs(title = "Posterior Predicted Probabilities by Label",
       x = "Predicted Probability", y = "Count") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("plot_predicted_probs.png", p2, width = 8, height = 4.5, dpi = 150)
cat("Saved: plot_predicted_probs.png\n")

# ──────────────────────────────────────────────
# 8. SUBSTITUTION FUNCTION
# ──────────────────────────────────────────────

sigmoid <- function(w, x) {
  1 / (1 + exp(-t(w) %*% x))
}

get_substitutes <- function(target, context, top_n = 5) {
  if (!(target %in% vocab)) {
    cat("Ingredient not in vocabulary:", target, "\n")
    return(NULL)
  }

  t_idx <- ing2idx[[target]]
  top5  <- order(cos_mat[t_idx, ], decreasing = TRUE)[2:6]
  cands <- setdiff(vocab, c(target, context))

  # Scale features using training set parameters
  raw_cos  <- cos_mat[t_idx, ing2idx[cands]]
  raw_ppmi <- ppmi_mat[t_idx, ing2idx[cands]]
  raw_ctx  <- sapply(ing2idx[cands], function(i) mean(ppmi_mat[i, top5]))

  feat_df <- data.frame(
    candidate     = cands,
    cos_sim_z     = scale(raw_cos)[,1],
    ppmi_z        = scale(raw_ppmi)[,1],
    ctx_overlap_z = scale(raw_ctx)[,1]
  )

  # for each candidate, average sigmoid over all posterior draws
  w_cols <- c("b_Intercept", "b_cos_sim_z", "b_ppmi_z", "b_ctx_overlap_z")
  probs  <- c()
  for (j in seq_len(nrow(feat_df))) {
    x    <- matrix(c(1, feat_df$cos_sim_z[j], feat_df$ppmi_z[j], feat_df$ctx_overlap_z[j]), ncol = 1)
    sigs <- c()
    for (i in seq_len(nrow(post_draws))) {
      w    <- matrix(unlist(post_draws[i, w_cols]), ncol = 1)
      sigs <- c(sigs, sigmoid(w, x))
    }
    probs <- c(probs, mean(sigs))
  }

  feat_df$prob <- probs
  feat_df <- feat_df[order(feat_df$prob, decreasing = TRUE), ]
  head(feat_df[, c("candidate", "prob")], top_n)
}

demo_cases <- list(
  list(target = "butter",      ctx = c("flour", "sugar", "eggs", "vanilla extract")),
  list(target = "soy sauce",   ctx = c("garlic", "ginger", "sesame oil", "rice")),
  list(target = "heavy cream", ctx = c("onion", "garlic", "pasta", "parmesan cheese"))
)

cat("\nDEMO: Top-5 substitutes\n")
for (case in demo_cases) {
  cat(sprintf("\nSubstitutes for '%s':\n", case$target))
  result <- get_substitutes(case$target, case$ctx, top_n = 5)
  if (!is.null(result)) print(result)
}

saveRDS(model, "bayesian_model.rds")
cat("\nSaved: bayesian_model.rds\nDone.\n")
