library(grf)

# ======================================================================================
# Get config from pytask
# ======================================================================================

args <- commandArgs(trailingOnly = TRUE)
path_to_yaml <- args[length(args)]
config <- yaml::yaml.load_file(path_to_yaml)

# ======================================================================================
# Main
# ======================================================================================

data <- read.csv(config[["bruhn_2016_data"]])

y <- data$outcome

w <- data$treatment

school <- data$school

x <- data[-(1:3)]

sum(!complete.cases(x)) / nrow(x)

t.test(w ~ !complete.cases(x))

cf <- causal_forest(x, y, w, W.hat = 0.5, clusters = school)

ate <- average_treatment_effect(cf)
ate

varimp <- variable_importance(cf)
ranked.vars <- order(varimp, decreasing = TRUE)

# Top 5 variables according to this measure
colnames(x)[ranked.vars[1:5]]

# Q: How does this know what to cluster by?
# --> specified clusters in the causal_forest function
best_linear_projection(cf, x[ranked.vars[1:5]])

pred <- predict(cf, x)

hist(pred$predictions)

saveRDS(pred, file = config[["produces"]])
