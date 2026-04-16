# DS 4420 Final Project - Model 2: Bayesian Logistic Regression
# Tanishi Datta & Shruthi Palaniappan
#
# Uses outputs from model1_collaborative_filtering.py to fit a Bayesian
# logistic regression that predicts whether ingredient B is a good
# substitute for ingredient A given the recipe context.
# Labels are derived from real recipe swap patterns (Jaccard similarity)
# rather than PPMI neighbors, so features and labels are independent.

library(brms)
library(dplyr)
library(ggplot2)
library(tidyr)
set.seed(4420)

# ──────────────────────────────────────────────
# 1. LOAD PRECOMPUTED DATA
# ──────────────────────────────────────────────
# Run model1_collaborative_filtering.py first to generate these files

vocab   <- readLines("vocab_top1000.txt")
V       <- length(vocab)
ing2idx <- setNames(seq_along(vocab), vocab)

cos_mat  <- as.matrix(read.csv("cos_sim_top1000.csv",  header = FALSE))
ppmi_mat <- as.matrix(read.csv("ppmi_top1000.csv",     header = FALSE))

cat(sprintf("Vocabulary size: %d ingredients\n", V))

# load cleaned recipes so labels come from real recipe patterns, not PPMI
recipes_df <- read.csv("recipes_clean.csv", stringsAsFactors = FALSE)
recipes_df$ingredients <- strsplit(recipes_df$ingredients, "\\|")
recipes_df$tags        <- strsplit(recipes_df$tags,        "\\|")
cat(sprintf("Recipes loaded: %d\n", nrow(recipes_df)))

# ──────────────────────────────────────────────
# 2. BUILD TRAINING PAIRS
# ──────────────────────────────────────────────
# Sample 150 target ingredients, 35 candidates each (~5250 pairs)
# For each pair compute 3 features:
#   cos_sim     - cosine similarity between target and candidate
#   ppmi        - PPMI score between target and candidate
#   ctx_overlap - how well candidate fits the target's top-5 neighbors
#
# Label = 1 if Jaccard similarity between recipe TAG sets is high
# Tags are a curated set of ~60 categories (cuisine, diet, dish type)
# e.g. butter and margarine both in {desserts, baking, american} -> high Jaccard
# e.g. butter and soy sauce: {desserts, baking} vs {asian, chinese} -> low Jaccard
# Tag sets are independent of PPMI/cosine (ingredient co-occurrence) features

# precompute which tags each ingredient appears with across all recipes
ing_tags <- list()
for (i in seq_len(nrow(recipes_df))) {
  ings <- recipes_df$ingredients[[i]]
  ings <- ings[ings %in% vocab]
  tags <- recipes_df$tags[[i]]
  if (length(ings) == 0 || length(tags) == 0) next
  for (ing in ings) {
    ing_tags[[ing]] <- union(ing_tags[[ing]], tags)
  }
}

# Jaccard similarity between two tag sets
# high = both ingredients appear in same types of recipes = likely substitutes
jaccard <- function(a, b) {
  if (length(a) == 0 || length(b) == 0) return(0)
  length(intersect(a, b)) / length(union(a, b))
}

n_targets    <- 150
n_candidates <- 35
target_ings  <- sample(vocab, n_targets)

rows <- list()
for (target in target_ings) {
  target_idx     <- ing2idx[[target]]
  top5_neighbors <- order(cos_mat[target_idx, ], decreasing = TRUE)[2:6]

  candidates <- sample(setdiff(seq_len(V), target_idx), n_candidates)

  for (cand_idx in candidates) {
    cand        <- vocab[cand_idx]

    # Jaccard between tag sets - how similar are the recipe types they appear in
    # tags are independent of PPMI/cosine (which use ingredient co-occurrence)
    context_sim <- jaccard(ing_tags[[target]], ing_tags[[cand]])

    rows[[length(rows) + 1]] <- data.frame(
      target      = target,
      candidate   = cand,
      cos_sim     = cos_mat[target_idx, cand_idx],
      ppmi        = ppmi_mat[target_idx, cand_idx],
      ctx_overlap = mean(ppmi_mat[cand_idx, top5_neighbors]),
      context_sim = context_sim,
      stringsAsFactors = FALSE
    )
  }
}

train_df <- bind_rows(rows) %>%
  filter(!is.na(cos_sim), !is.na(ppmi), !is.na(ctx_overlap)) %>%
  distinct(target, candidate, .keep_all = TRUE)

# Use 90th percentile as threshold so positive rate is always ~10%
# Fixed threshold of 0.3 was unreliable - context set sizes vary widely across vocab
jaccard_thresh <- quantile(train_df$context_sim, 0.90)
cat(sprintf("Jaccard threshold (90th pct): %.4f\n", jaccard_thresh))
train_df$label <- as.integer(train_df$context_sim >= jaccard_thresh)
train_df        <- select(train_df, -context_sim)

cat(sprintf("Training pairs: %d\n", nrow(train_df)))
cat(sprintf("Positive rate: %.4f\n", mean(train_df$label)))

# ──────────────────────────────────────────────
# 3. FEATURE SCALING
# ──────────────────────────────────────────────
# Z-score each feature so they are on the same scale before fitting

train_df <- train_df %>%
  mutate(
    cos_sim_z     = scale(cos_sim)[,1],
    ppmi_z        = scale(ppmi)[,1],
    ctx_overlap_z = scale(ctx_overlap)[,1]
  ) %>%
  select(-cos_sim, -ppmi, -ctx_overlap)

# ──────────────────────────────────────────────
# 4. BAYESIAN LOGISTIC REGRESSION
# ──────────────────────────────────────────────
# student_t on intercept (fat tails for logistic regression baseline)
# Normal(0, 2) on slopes - weakly informative, discourages extreme values

default_prior(label ~ cos_sim_z + ppmi_z + ctx_overlap_z,
              data   = train_df,
              family = bernoulli("logit"))

priors <- c(
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
  prior   = priors,
  seed    = 4420
)

# ──────────────────────────────────────────────
# 5. EVALUATION
# ──────────────────────────────────────────────

summary(model)
plot(model) # trace plots - check chains mixed well (fuzzy caterpillar shape)

# extract all posterior samples as a dataframe (one row per MCMC draw)
posterior <- as.data.frame(model)

# check ACF plots to see if chains are mixing well - same as class
# want autocorrelation to drop off quickly, close to 0 after a few lags
acf(posterior[, "b_cos_sim_z"],     main = "ACF - Cosine Similarity")
acf(posterior[, "b_ppmi_z"],        main = "ACF - PPMI Score")
acf(posterior[, "b_ctx_overlap_z"], main = "ACF - Context Overlap")

# 95% credible intervals for each coefficient
cat("\n=== 95% Credible Intervals ===\n")
for (param in c("b_Intercept", "b_cos_sim_z", "b_ppmi_z", "b_ctx_overlap_z")) {
  q <- quantile(posterior[[param]], c(0.025, 0.5, 0.975))
  cat(sprintf("  %-25s  median=%.3f  95%% CI=[%.3f, %.3f]\n", param, q[2], q[1], q[3]))
}

# predicted probability for each training pair
post_preds   <- posterior_predict(model)
pred_probs   <- colMeans(post_preds)
pred_labels  <- as.integer(pred_probs >= 0.5)
total_correct <- sum(pred_labels == train_df$label)

cat(sprintf("\nIn-sample accuracy: %d / %d = %.3f\n",
            total_correct, nrow(train_df), total_correct / nrow(train_df)))

# AUC via trapezoidal integration (no external packages)
compute_auc <- function(labels, scores) {
  ord    <- order(scores, decreasing = TRUE)
  labels <- labels[ord]
  n_pos  <- sum(labels)
  n_neg  <- length(labels) - n_pos
  tpr    <- cumsum(labels) / n_pos
  fpr    <- cumsum(1 - labels) / n_neg
  sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1))) / 2
}

total_auc <- compute_auc(train_df$label, pred_probs)
cat(sprintf("In-sample AUC: %.3f\n", total_auc))

cat("\nComputing LOO-CV...\n")
print(loo(model))

# how each feature independently affects predicted substitution probability
plot(conditional_effects(model, "cos_sim_z"))
plot(conditional_effects(model, "ppmi_z"))
plot(conditional_effects(model, "ctx_overlap_z"))

# ──────────────────────────────────────────────
# 6. PLOTS
# ──────────────────────────────────────────────

# reshape posterior samples into long format for plotting
post_long <- posterior %>%
  select(b_Intercept, b_cos_sim_z, b_ppmi_z, b_ctx_overlap_z) %>%
  pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
  mutate(parameter = recode(parameter,
    "b_Intercept"     = "Intercept",
    "b_cos_sim_z"     = "Cosine Similarity",
    "b_ppmi_z"        = "PPMI Score",
    "b_ctx_overlap_z" = "Context Overlap"
  ))

# Plot 1: posterior distributions of each coefficient
# dashed line at 0 - if the whole distribution is to the right, that feature matters
p1 <- ggplot(post_long, aes(x = value, fill = parameter)) +
  geom_density(alpha = 0.7, color = NA) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  facet_wrap(~parameter, scales = "free", ncol = 2) +
  scale_fill_manual(values = c("#E07A5F", "#3D405B", "#81B29A", "#F2CC8F")) +
  labs(title = "Posterior Distributions of Model Coefficients",
       subtitle = "Distributions entirely right of 0 indicate a positive predictor of substitutability",
       x = "Log-Odds", y = "Density") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(size = 10, color = "gray40"))

ggsave("plot_posterior_coefficients.png", p1, width = 8, height = 5, dpi = 150)
cat("Saved: plot_posterior_coefficients.png\n")

# Plot 2: ROC curve - shows how well the model separates substitutes from non-substitutes
# area under the curve = AUC (closer to 1.0 = better)
train_df$pred_prob <- pred_probs

ord    <- order(pred_probs, decreasing = TRUE)
labels <- train_df$label[ord]
n_pos  <- sum(labels)
n_neg  <- length(labels) - n_pos
tpr    <- c(0, cumsum(labels) / n_pos)       # true positive rate
fpr    <- c(0, cumsum(1 - labels) / n_neg)   # false positive rate

roc_df <- data.frame(fpr = fpr, tpr = tpr)

p2 <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "#81B29A", linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.75, y = 0.15,
           label = sprintf("AUC = %.3f", total_auc),
           size = 5, color = "#3D405B") +
  annotate("text", x = 0.63, y = 0.57,
           label = "Random guessing", size = 3.5, color = "gray50", angle = 34) +
  labs(title = "ROC Curve — Bayesian Logistic Regression",
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("plot_roc_curve.png", p2, width = 6, height = 6, dpi = 150)
cat("Saved: plot_roc_curve.png\n")

# Plot 3: predicted probability by label (boxplot)
p3 <- ggplot(train_df, aes(x = factor(label), y = pred_prob, fill = factor(label))) +
  geom_boxplot(alpha = 0.8, outlier.alpha = 0.2) +
  scale_fill_manual(values = c("#E07A5F", "#81B29A"),
                    labels = c("Non-substitute", "Substitute"),
                    name = "Label") +
  scale_x_discrete(labels = c("Non-substitute", "Substitute")) +
  labs(title = "Predicted Substitution Probability by Label",
       x = "", y = "Predicted Probability") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "none")

ggsave("plot_predicted_probs.png", p3, width = 8, height = 5, dpi = 150)
cat("Saved: plot_predicted_probs.png\n")

# Plot 4: predicted probability histogram by label
p4 <- ggplot(train_df, aes(x = pred_prob, fill = factor(label))) +
  geom_histogram(bins = 40, alpha = 0.75, position = "identity") +
  scale_fill_manual(values = c("#E07A5F", "#81B29A"),
                    labels = c("Non-substitute", "Substitute"),
                    name = "Label") +
  labs(title = "Posterior Predicted Probabilities by Label",
       x = "Predicted Probability", y = "Count") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("plot_predicted_histogram.png", p4, width = 8, height = 5, dpi = 150)
cat("Saved: plot_predicted_histogram.png\n")

# ──────────────────────────────────────────────
# 7. SUBSTITUTION FUNCTION
# ──────────────────────────────────────────────

sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

get_substitutes <- function(target_ingredient, context_ingredients, top_n = 5) {
  if (!(target_ingredient %in% vocab)) {
    cat("Ingredient not in vocabulary:", target_ingredient, "\n")
    return(NULL)
  }

  target_idx     <- ing2idx[[target_ingredient]]
  top5_neighbors <- order(cos_mat[target_idx, ], decreasing = TRUE)[2:6]
  candidates     <- setdiff(vocab, c(target_ingredient, context_ingredients))

  # compute features for every candidate
  raw_cos_sim     <- cos_mat[target_idx, ing2idx[candidates]]
  raw_ppmi        <- ppmi_mat[target_idx, ing2idx[candidates]]
  raw_ctx_overlap <- sapply(ing2idx[candidates], function(i) mean(ppmi_mat[i, top5_neighbors]))

  cand_df <- data.frame(
    candidate     = candidates,
    cos_sim_z     = scale(raw_cos_sim)[,1],
    ppmi_z        = scale(raw_ppmi)[,1],
    ctx_overlap_z = scale(raw_ctx_overlap)[,1]
  )

  # first tried brms predict() directly but it was really slow for 1000 candidates
  # cand_df$score <- predict(model, newdata = cand_df, type = "response")[, "Estimate"]

  # also tried looping over each candidate and each posterior draw but got stuck
  # avg_probs <- c()
  # for (j in seq_len(nrow(cand_df))) {
  #   x    <- matrix(c(1, cand_df$cos_sim_z[j], cand_df$ppmi_z[j], cand_df$ctx_overlap_z[j]), ncol = 1)
  #   sigs <- c()
  #   for (i in seq_len(nrow(posterior))) {
  #     w    <- matrix(unlist(posterior[i, param_cols]), ncol = 1)
  #     sigs <- c(sigs, sigmoid(t(w) %*% x))
  #   }
  #   avg_probs <- c(avg_probs, mean(sigs))
  # }

  # switched to matrix multiply - same result, way faster
  param_cols <- c("b_Intercept", "b_cos_sim_z", "b_ppmi_z", "b_ctx_overlap_z")
  W          <- as.matrix(posterior[, param_cols])       # draws x 4
  X          <- cbind(1, cand_df$cos_sim_z, cand_df$ppmi_z, cand_df$ctx_overlap_z)  # cands x 4
  log_odds   <- X %*% t(W)                               # cands x draws
  avg_probs  <- rowMeans(sigmoid(log_odds))              # average prob per candidate

  cand_df$score <- avg_probs
  cand_df <- cand_df[order(cand_df$score, decreasing = TRUE), ]
  head(cand_df[, c("candidate", "score")], top_n)
}

# ──────────────────────────────────────────────
# 8. DEMO
# ──────────────────────────────────────────────

demo_cases <- list(
  list(target = "butter",      context = c("flour", "sugar", "eggs", "vanilla extract")),
  list(target = "soy sauce",   context = c("garlic", "ginger", "sesame oil", "rice")),
  list(target = "heavy cream", context = c("onion", "garlic", "pasta", "parmesan cheese")),
  list(target = "eggs",        context = c("flour", "sugar", "butter", "baking powder")),
  list(target = "olive oil",   context = c("garlic", "tomatoes", "basil", "pasta"))
)

cat("\nDEMO: Top-5 substitutes\n")
for (case in demo_cases) {
  cat(sprintf("\nSubstitutes for '%s':\n", case$target))
  ranked <- get_substitutes(case$target, case$context, top_n = 5)
  if (!is.null(ranked)) print(ranked)
}

cat("\nDone.\n")
