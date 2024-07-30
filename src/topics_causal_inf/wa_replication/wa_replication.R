library(grf)

# ======================================================================================
# Functions
# ======================================================================================

# Simulation
simulation <- function(
    n_sim, n_obs, dim, main_effect, treatment_effect, propensity, num_trees,
    sample_fraction, return_grid = FALSE) {
  # Run experiments
  if (return_grid == TRUE) {
    return(experiment(n_obs, dim, main_effect, treatment_effect, propensity, num_trees,
      sample_fraction,
      return_grid = return_grid
    ))
  }

  mse <- rep(0, n_sim)
  for (i in 1:n_sim) {
    mse[i] <- experiment(n_obs, dim, main_effect, treatment_effect, propensity,
      num_trees,
      sample_fraction,
      return_grid = return_grid
    )
  }

  return(mse)
}

# Single experiment
experiment <- function(
    n_obs, dim, main_effect, treatment_effect, propensity, num_trees, sample_fraction,
    return_grid = FALSE) {
  # Draw data
  data <- data(n_obs, dim, main_effect, treatment_effect, propensity)
  data_test <- data(n_obs, dim, main_effect, treatment_effect, propensity)

  # Fit causal forest
  cf <- causal_forest(data$x, data$y, data$w,
    num.trees = num_trees,
    sample.fraction = sample_fraction
  )

  if (!return_grid) {
    # Predict on test-data
    pred <- predict(cf, data_test$x)
    true <- treatment_effect(data_test$x)

    # Evaluate fit by mse
    mse <- mean((pred$predictions - true)^2)

    # Make prediction on x1, x2 from 0 to 1
    return(mse)
  }

  data_eval <- data(10000, dim, main_effect, treatment_effect, propensity)

  pred <- predict(cf, data_eval$x)

  pred_eval <- pred$predictions
  true_eval <- treatment_effect(data_eval$x)

  # Put into one dataframe
  mse <- mean((pred_eval - true_eval)^2)

  out <- cbind(data_eval$x, pred_eval, true_eval, mse)

  colnames(out) <- c(paste0("x", 1:dim), "pred", "true", "mse")

  return(out)
}

# Generate data
data <- function(n_obs, dim, main_effect, treatment_effect, propensity) {
  x <- matrix(0, n_obs, dim)
  for (i in 1:dim) {
    x[, i] <- runif(n_obs)
  }

  w <- rbinom(n_obs, 1, propensity(x))
  y <- main_effect(x) + w * treatment_effect(x) + rnorm(n_obs)

  return(list(x = x, y = y, w = w))
}

xi <- function(x, a = 20, b = 1 / 3) {
  return(1 + 1 / (1 + exp(-a * (x - b))))
}

# ======================================================================================
# Get config from pytask
# ======================================================================================

args <- commandArgs(trailingOnly = TRUE)
path_to_yaml <- args[length(args)]
config <- yaml::yaml.load_file(path_to_yaml)

n_sim <- config[["n_sim"]]
n_obs <- config[["n_obs"]]
dim <- config[["dim"]]
return_grid <- config[["return_grid"]]
num_trees <- config[["num_trees"]]
sample_fraction <- config[["sample_fraction"]]

# ======================================================================================
# Main
# ======================================================================================

m_dgp_linear <- function(x) {
  return(2 * x[, 1] - 1)
}

m_dgp_cons <- function(x) {
  return(rep(0, nrow(x)))
}

t_dgp1 <- function(x) {
  return(rep(0, nrow(x)))
}

t_dgp2 <- function(x) {
  return(xi(x[, 1]) * xi(x[, 2]))
}

t_dgp3 <- function(x) {
  return(xi(x[, 1], a = 12, b = 1 / 2) * xi(x[, 2], a = 12, b = 1 / 2))
}

t_dgp4 <- function(x) {
  return(
    xi(x[, 1], a = 12, b = 1 / 2)
    * xi(x[, 2], a = 12, b = 1 / 2)
      * xi(x[, 3], a = 12, b = 1 / 2)
      * xi(x[, 4], a = 12, b = 1 / 2)
  )
}

t_dgp5 <- function(x) {
  return(
    xi(x[, 1], a = 12, b = 1 / 2)
    * xi(x[, 2], a = 12, b = 1 / 2)
      * xi(x[, 3], a = 12, b = 1 / 2)
      * xi(x[, 4], a = 12, b = 1 / 2)
      * xi(x[, 5], a = 12, b = 1 / 2)
      * xi(x[, 6], a = 12, b = 1 / 2)
      * xi(x[, 7], a = 12, b = 1 / 2)
      * xi(x[, 8], a = 12, b = 1 / 2)
  )
}

p_dgp_beta <- function(x) {
  return(0.25 * (1 + dbeta(x[, 1], 2, 4)))
}

p_dgp_cons <- function(x) {
  return(rep(0.5, nrow(x)))
}

if (config[["dgp"]] == "dgp1") {
  res <- simulation(n_sim, n_obs, dim, m_dgp_linear, t_dgp1, p_dgp_beta,
    return_grid = return_grid, num_trees = num_trees, sample_fraction = sample_fraction
  )
}
if (config[["dgp"]] == "dgp2") {
  res <- simulation(n_sim, n_obs, dim, m_dgp_cons, t_dgp2, p_dgp_cons,
    return_grid = return_grid, num_trees = num_trees, sample_fraction = sample_fraction
  )
}
if (config[["dgp"]] == "dgp3") {
  res <- simulation(n_sim, n_obs, dim, m_dgp_cons, t_dgp3, p_dgp_cons,
    return_grid = return_grid, num_trees = num_trees, sample_fraction = sample_fraction
  )
}
if (config[["dgp"]] == "dgp4") {
  res <- simulation(n_sim, n_obs, dim, m_dgp_cons, t_dgp4, p_dgp_cons,
    return_grid = return_grid, num_trees = num_trees, sample_fraction = sample_fraction
  )
}
if (config[["dgp"]] == "dgp5") {
  res <- simulation(n_sim, n_obs, dim, m_dgp_cons, t_dgp5, p_dgp_cons,
    return_grid = return_grid, num_trees = num_trees, sample_fraction = sample_fraction
  )
}

# Save results to 3 separate files
saveRDS(res, file = config[["produces"]])
