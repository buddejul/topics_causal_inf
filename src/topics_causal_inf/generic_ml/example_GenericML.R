# install.packages("GenericML")
# install.packages("glmnet")
# install.packages("ranger")
# install.packages("e1071")

library("GenericML")

n <- 1000
p <- 3
d <- rbinom(n, 1, 0.5)
z <- matrix(runif(n * p), n, p)

colnames(z) <- paste("z", 1:p)

y0 <- as.numeric(z %*% rexp(p))

# treatment effect increases with Z1, has a level shift along Z2 (at 0.5)
# and has no pattern along Z3
cate <- 2 * z[, 1] + ifelse(z[, 2] >= 0.5, 1, -1)
ate <- mean(cate) # average treatment effect
y1 <- cate + y0 # potential outcome under treatment
y <- ifelse(d == 1, y1, y0) # observed outcome

# ======================================================================================
# 2 Prepare arguments for GenericML function
# ======================================================================================
# quantile cutoffs for the GATES grouping of the estimated CATEs
quantile_cutoffs <- c(0.25, 0.5, 0.75)

# specify the learner of the propensity score (non-penalized logistic regression here).
# Propensity scores can also directly be supplied.
learner_propensity_score <- "mlr3::lrn('glmnet', lambda = 0, alpha = 1)"

# specify the considered learners of the BCA and the CATE
# (here: lasso, random forest, and SVM)
learners_genericml <- c(
  "lasso", "mlr3::lrn('ranger', num.trees = 100)",
  "mlr3::lrn('svm')"
)

# specify the data that shall be used for the CLAN
# here, we use all variables of Z and uniformly distributed random noise
z_clan <- cbind(z, random = runif(n))

# specify the number of splits (many to rule out seed-dependence of results)
num_splits <- 1000

# specify if a HT transformation shall be used when estimating BLP and GATES
ht <- FALSE

# A list controlling the vars used in the matrix X1 for the BLP and GATES regressions.
x1_blp <- setup_X1()
x1_gates <- setup_X1()

# consider differences between group K (most affected) with groups 1, 2, and 3.
diff_gates <- setup_diff(
  subtract_from = "most",
  subtracted = 1:3
)
diff_clan <- setup_diff(
  subtract_from = "most",
  subtracted = 1:3
)

# specify the significance level
significance_level <- 0.05

# specify minimum variation of pred. before Gaussian noise with var=var(Y)/20 is added.
min_variation <- 1e-05

# specify which estimator of the error covariance matrix shall be used in BLP and GATES
# (standard OLS covariance matrix estimator here)
vcov_blp <- setup_vcov()
vcov_gates <- setup_vcov()

# specify whether assumed that group variances of the most/least affected groups equal.
equal_variances_clan <- FALSE

# specify the proportion of samples that shall be selected in the auxiliary set
prop_aux <- 0.5

# specify sampling strategy (possibly stratified). Here simple random sampling is used.
stratify <- setup_stratify()

# specify whether splits and auxiliary results of the learners shall be stored
store_splits <- TRUE
store_learners <- FALSE # to save memory

# parallelization options
parallel <- TRUE
num_cores <- 8 # 8 cores
seed <- 123456

res <- GenericML(
  Z = z, D = d, Y = y,
  learner_propensity_score = learner_propensity_score,
  learners_GenericML = learners_genericml,
  num_splits = num_splits,
  Z_CLAN = z_clan,
  HT = ht,
  X1_BLP = x1_blp,
  X1_GATES = x1_gates,
  vcov_BLP = vcov_blp,
  vcov_GATES = vcov_gates,
  quantile_cutoffs = quantile_cutoffs,
  diff_GATES = diff_gates,
  diff_CLAN = diff_clan,
  equal_variances_CLAN = equal_variances_clan,
  prop_aux = prop_aux,
  stratify = stratify,
  significance_level = significance_level,
  min_variation = min_variation,
  parallel = parallel,
  num_cores = num_cores,
  seed = seed,
  store_splits = store_splits,
  store_learners = store_learners
)

# ======================================================================================
# BLP Results
# ===========a==========================================================================
res_blp <- get_BLP(res)
