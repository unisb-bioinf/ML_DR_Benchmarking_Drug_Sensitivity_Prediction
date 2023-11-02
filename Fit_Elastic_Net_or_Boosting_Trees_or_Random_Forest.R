# train: data.frame of training data where rownames are cell line names/IDs and columns are features
# test: data.frame of test data where rownames are cell line names/IDs and columns are features
# method: name of ML algorithm (either "boosting_trees", "elastic_net", or "random_forest")
# method_parameters: names list of hyperparameters for the used ML algorithm 
#   elastic net: "lambda", "alpha", "standardize_values"
#   random_forest: "num_trees", "mtry"
#   boosting_trees: "num_trees", "splits_per_tree", "lambda", "bag_fraction", "distr"
# seed: random seed
# returns: model predictions for training and test data (model is only trained on training data) and runtime of training
fit_model = function(train, test, method, method_parameters, seed = 2208) {
  
  set.seed(seed)
  train_data = train[sample(nrow(train)),]
  
  if(method == "boosting_trees") {
    library(gbm)
    
    start_time = Sys.time()
    set.seed(seed)
    model = gbm(response ~ . , data = train_data, distribution = as.character(method_parameters$distr), n.trees = method_parameters$num_trees,
                shrinkage = method_parameters$lambda, interaction.depth = method_parameters$splits_per_tree, bag.fraction = method_parameters$bag_fraction,
                cv.folds = 0, n.cores = cores)
    end_time = Sys.time()

    # compute predictions
    predictions_test = data.frame(predict(model, test, n.trees = method_parameters$num_trees))
    predictions_train = data.frame(predict(model, train_data, n.trees = method_parameters$num_trees))
  }
  
  if(method == "elastic_net") {
    library(glmnet)
    
    start_time = Sys.time()
    set.seed(seed)
    model = glmnet(x = train_data[, colnames(train_data) != "response"], y = train_data$response,
           standardize = method_parameters$standardize_values, lambda = method_parameters$lambda, alpha = method_parameters$alpha,
           num.threads = cores)
    end_time = Sys.time()
    
    predictions_test = data.frame(predict(model, newx = as.matrix(test[, colnames(test) != "response"])))
    predictions_train = data.frame(predict(model, newx = as.matrix(train_data[, colnames(train_data) != "response"])))
  }
  
  if(method == "random_forest") {
    library("ranger")
    
    # ranger only allows column names without '-'
    colnames(train_data) = make.names(colnames(train_data))
    colnames(test) = make.names(colnames(test))
    
    mtry = method_parameters$mtry
    num_trees = method_parameters$num_trees
    
    start_time = Sys.time()
    model = ranger(response ~ . , data = train_data, num.trees = num_trees, respect.unordered.factors = 'order', seed = seed, mtry = mtry, num.threads = cores)
    end_time = Sys.time()
    
    # compute test MSE
    predictions_test = data.frame(predict(model,data=test)$prediction)
    predictions_train = data.frame(predict(model,data=train_data)$prediction)
  }
  
  colnames(predictions_test) = colnames(predictions_train) = "predicted_value"
  rownames(predictions_test) = rownames(test)
  rownames(predictions_train) = rownames(train_data)
  
  duration = difftime(end_time, start_time, units = 'sec')  
  
  return(list("predictions_test" = predictions_test, "predictions_train" = predictions_train, "runtime" = duration))
}

# based on the book "An Introduction to Statistical Learning: With Applications in R", page 234
# actual: vector of actual ln(IC50) values for training or test cell lines
# predicted: vector of predicted ln(IC50) values for training or test cell lines
# num_predictiors: number of input features the model was trained on
# returns: adjusted R squared value
R_squared_adjusted = function(actual, predicted, num_predictors) {
  
  num_samples = length(actual)
  
  if(num_samples - num_predictors - 1 < 1) {
    return(NA)
  } else {
    RSS = sum((actual - predicted) ^ 2) 
    TSS = sum((actual - mean(actual)) ^ 2) 
    
    R2_adj = 1 - ((RSS/(num_samples - num_predictors - 1))/(TSS/(num_samples - 1)))
    
    return(R2_adj)
  }
}

# data: data.frame (either training or test samples) with columns "sample" (cell line name/ID), "response" (true ln(IC50)), "predicted_response" (predicted ln(IC50))
# num_features (optional): number of input features the model was trained on
# mean_train_IC50 (optional): mean ln(IC50) of training samples
# range_train_IC50 (optional): size of the range of ln(IC50) values in training data
# returns: various error measures
compute_errors = function(data, num_features = NA, mean_train_IC50 = NA, range_train_IC50 = NA) {
  library(ggpubr)
  MSE = mean((data$response - data$predicted_response)^2)
  Median_SE = median((data$response - data$predicted_response)^2)
  MAE = mean(abs(data$response - data$predicted_response))
  PCC = cor.test(data$response, data$predicted_response)
  PCC_pvalue = PCC$p.value
  PCC = PCC$estimate
  
  # measures that require additional information on number of features + train data
  Adjusted_R2 = Baseline_MSE = Baseline_normalized_MSE = Range_normalized_MSE = NA
  if(!is.na(num_features)) {
    Adjusted_R2 = R_squared_adjusted(data$response, data$predicted_response, num_features)
  }
  if(!is.na(mean_train_IC50)) {
    Baseline_MSE = mean((data$response - mean_train_IC50)^2)
    Baseline_normalized_MSE = MSE / Baseline_MSE
  }
  if(!is.na(range_train_IC50)) {
    Range_normalized_MSE = MSE / range_train_IC50
  }
  
  return(list(
    "MSE" = MSE,
    "Median_SE" = Median_SE,
    "MAE" = MAE,
    "PCC" = PCC,
    "PCC_pvalue" = PCC_pvalue,
    "Adjusted_R2" = Adjusted_R2,
    "Baseline_MSE" = Baseline_MSE,
    "Baseline_normalized_MSE" = Baseline_normalized_MSE,
    "Range_normalized_MSE" = Range_normalized_MSE
  ))
}

# data: data.frame (either training or test samples) with columns "sample" (cell line name/ID), "response" (true ln(IC50)), "predicted_response" (predicted ln(IC50))
# drug_threshold: drug-specific ln(IC50) threshold to divide cell lines in sensitive (ln(IC50) < threshold) and resistant ones
# num_features (optional): number of input features the model was trained on
# mean_train_IC50 (optional): mean ln(IC50) of training samples
# range_train_IC50 (optional): size of the range of ln(IC50) values in training data
# returns: various error measures that require an ln(IC50) threshold
compute_errors_threshold_based = function(data, drug_threshold, num_features = NA, mean_train_IC50 = NA, range_train_IC50 = NA) {
  library(ggpubr)
  
  data$class = ifelse(data$response < threshold, 1, 0)
  data$predicted_class = ifelse(data$predicted_response < threshold, 1, 0)
  
  data_sens = data[data$class == 1,]
  data_res = data[data$class == 0,]
  stopifnot(nrow(data) == (nrow(data_sens) + nrow(data_res)))
  
  data_correct = data[data$class == data$predicted_class,]
  data_incorrect = data[data$class != data$predicted_class,]
  data_TP = data[(data$predicted_class == 1) & (data$class == 1),]
  data_TN = data[(data$predicted_class == 0) & (data$class == 0),]
  data_FP = data[(data$predicted_class == 1) & (data$class == 0),]
  data_FN = data[(data$predicted_class == 0) & (data$class == 1),]
  
  stopifnot(nrow(data_correct) + nrow(data_incorrect) == nrow(data))
  stopifnot(nrow(data_TP) + nrow(data_TN) == nrow(data_correct))
  stopifnot(nrow(data_FP) + nrow(data_FN) == nrow(data_incorrect))
  stopifnot(all.equal(data_correct$predicted_class, data_correct$class))
  stopifnot(unique(c(data_TP$predicted_class, data_TP$class)) == 1)
  stopifnot(unique(c(data_TN$predicted_class, data_TN$class)) == 0)
  stopifnot(unique(data_FP$predicted_class) == 1)
  stopifnot(unique(data_FN$predicted_class) == 0)
  stopifnot(unique(data_FP$class) == 0)
  stopifnot(unique(data_FN$class) == 1)
  
  TP = nrow(data_TP)
  TN = nrow(data_TN)
  FP = nrow(data_FP)
  FN = nrow(data_FN)
  stopifnot((TP + TN + FP + FN) == nrow(data))
  
  Sensitivity = TP / (TP + FN)
  Specificity = TN / (TN + FP)
  
  MSE_sensitive_CLs = mean((data_sens$response - data_sens$predicted_response)^2)
  MSE_resistant_CLs = mean((data_res$response - data_res$predicted_response)^2)
  MSE_correctly_classified_CLs = mean((data_correct$response - data_correct$predicted_response)^2)
  MSE_incorrectly_classified_CLs = mean((data_incorrect$response - data_incorrect$predicted_response)^2)
  MSE_TPs = mean((data_TP$response - data_TP$predicted_response)^2)
  MSE_TNs = mean((data_TN$response - data_TN$predicted_response)^2)
  MSE_FPs = mean((data_FP$response - data_FP$predicted_response)^2)
  MSE_FNs = mean((data_FN$response - data_FN$predicted_response)^2)
  
  Median_SE_sensitive_CLs = median((data_sens$response - data_sens$predicted_response)^2)
  Median_SE_resistant_CLs = median((data_res$response - data_res$predicted_response)^2)
  Median_SE_correctly_classified_CLs = median((data_correct$response - data_correct$predicted_response)^2)
  Median_SE_incorrectly_classified_CLs = median((data_incorrect$response - data_incorrect$predicted_response)^2)
  Median_SE_TPs = median((data_TP$response - data_TP$predicted_response)^2)
  Median_SE_TNs = median((data_TN$response - data_TN$predicted_response)^2)
  Median_SE_FPs = median((data_FP$response - data_FP$predicted_response)^2)
  Median_SE_FNs = median((data_FN$response - data_FN$predicted_response)^2)
  
  MAE_sensitive_CLs = mean(abs(data_sens$response - data_sens$predicted_response))
  MAE_resistant_CLs = mean(abs(data_res$response - data_res$predicted_response))
  MAE_correctly_classified_CLs = mean((data_correct$response - data_correct$predicted_response))
  MAE_incorrectly_classified_CLs = mean((data_incorrect$response - data_incorrect$predicted_response))
  MAE_TPs = mean(abs(data_TP$response - data_TP$predicted_response))
  MAE_TNs = mean(abs(data_TN$response - data_TN$predicted_response))
  MAE_FPs = mean(abs(data_FP$response - data_FP$predicted_response))
  MAE_FNs = mean(abs(data_FN$response - data_FN$predicted_response))
  
  if(nrow(data_sens) > 2) {
    PCC_sensitive_CLs = cor.test(data_sens$response, data_sens$predicted_response)
    PCC_pvalue_sensitive_CLs = PCC_sensitive_CLs$p.value
    PCC_sensitive_CLs = PCC_sensitive_CLs$estimate
  } else {
    PCC_sensitive_CLs = PCC_pvalue_sensitive_CLs = NA
  }
  if(nrow(data_res) > 2) {
    PCC_resistant_CLs = cor.test(data_res$response, data_res$predicted_response)
    PCC_pvalue_resistant_CLs = PCC_resistant_CLs$p.value
    PCC_resistant_CLs = PCC_resistant_CLs$estimate
  } else {
    PCC_resistant_CLs = PCC_pvalue_resistant_CLs = NA
  }
  if(nrow(data_correct) > 2) {
    PCC_correctly_classified_CLs = cor.test(data_correct$response, data_correct$predicted_response)
    PCC_pvalue_correctly_classified_CLs = PCC_correctly_classified_CLs$p.value
    PCC_correctly_classified_CLs = PCC_correctly_classified_CLs$estimate
  } else {
    PCC_correctly_classified_CLs = PCC_pvalue_correctly_classified_CLs = NA
  }
  if(nrow(data_incorrect) > 2) {
    PCC_incorrectly_classified_CLs = cor.test(data_incorrect$response, data_incorrect$predicted_response)
    PCC_pvalue_incorrectly_classified_CLs = PCC_incorrectly_classified_CLs$p.value
    PCC_incorrectly_classified_CLs = PCC_incorrectly_classified_CLs$estimate
  } else {
    PCC_incorrectly_classified_CLs = PCC_pvalue_incorrectly_classified_CLs = NA
  }
  if(TP > 2) {
    PCC_TPs = cor.test(data_TP$response, data_TP$predicted_response)
    PCC_pvalue_TPs = PCC_TPs$p.value
    PCC_TPs = PCC_TPs$estimate
  } else {
    PCC_TPs = PCC_pvalue_TPs = NA
  }
  if(TN > 2) {
    PCC_TNs = cor.test(data_TN$response, data_TN$predicted_response)
    PCC_pvalue_TNs = PCC_TNs$p.value
    PCC_TNs = PCC_TNs$estimate
  } else {
    PCC_TNs = PCC_pvalue_TNs = NA
  }
  if(FP > 2) {
    PCC_FPs = cor.test(data_FP$response, data_FP$predicted_response)
    PCC_pvalue_FPs = PCC_FPs$p.value
    PCC_FPs = PCC_FPs$estimate
  } else {
    PCC_FPs = PCC_pvalue_FPs = NA
  }
  if(FN > 2) {
    PCC_FNs = cor.test(data_FN$response, data_FN$predicted_response)
    PCC_pvalue_FNs = PCC_FNs$p.value
    PCC_FNs = PCC_FNs$estimate
  } else {
    PCC_FNs = PCC_pvalue_FNs = NA
  }
  
  # measures that require additional information on number of features + train data
  Adjusted_R2_sensitive_CLs = Adjusted_R2_resistant_CLs = Adjusted_R2_correctly_classified_CLs = Adjusted_R2_incorrectly_classified_CLs = NA
  Adjusted_R2_TPs = Adjusted_R2_TNs = Adjusted_R2_FPs = Adjusted_R2_FNs = NA
  Baseline_normalized_MSE_sensitive_CLs = Baseline_normalized_MSE_resistant_CLs = Baseline_normalized_MSE_correctly_classified_CLs = Baseline_normalized_MSE_incorrectly_classified_CLs = NA
  Baseline_normalized_MSE_TPs = Baseline_normalized_MSE_TNs = Baseline_normalized_MSE_FPs = Baseline_normalized_MSE_FNs = NA
  Range_normalized_MSE_sensitive_CLs = Range_normalized_MSE_resistant_CLs = Range_normalized_MSE_correctly_classified_CLs = Range_normalized_MSE_incorrectly_classified_CLs = NA
  Range_normalized_MSE_TPs = Range_normalized_MSE_TNs = Range_normalized_MSE_FPs = Range_normalized_MSE_FNs = NA
  
  if(!is.na(num_features)) {
    Adjusted_R2_sensitive_CLs = R_squared_adjusted(data_sens$response, data_sens$predicted_response, num_features)
    Adjusted_R2_resistant_CLs = R_squared_adjusted(data_res$response, data_res$predicted_response, num_features)
    Adjusted_R2_correctly_classified_CLs = R_squared_adjusted(data_correct$response, data_correct$predicted_response, num_features)
    Adjusted_R2_incorrectly_classified_CLs = R_squared_adjusted(data_incorrect$response, data_incorrect$predicted_response, num_features)
    Adjusted_R2_TPs = R_squared_adjusted(data_TP$response, data_TP$predicted_response, num_features)
    Adjusted_R2_TNs = R_squared_adjusted(data_TN$response, data_TN$predicted_response, num_features)
    Adjusted_R2_FPs = R_squared_adjusted(data_FP$response, data_FP$predicted_response, num_features)
    Adjusted_R2_FNs = R_squared_adjusted(data_FN$response, data_FN$predicted_response, num_features)  
  }
  if(!is.na(mean_train_IC50)) {
    Baseline_MSE_sensitive_CLs = mean((data_sens$response - mean_train_IC50)^2)
    Baseline_MSE_resistant_CLs = mean((data_res$response - mean_train_IC50)^2)
    Baseline_MSE_correctly_classified_CLs = mean((data_correct$response - mean_train_IC50)^2)
    Baseline_MSE_incorrectly_classified_CLs = mean((data_incorrect$response - mean_train_IC50)^2)
    Baseline_MSE_TPs = mean((data_TP$response - mean_train_IC50)^2)
    Baseline_MSE_TNs = mean((data_TN$response - mean_train_IC50)^2)
    Baseline_MSE_FPs = mean((data_FP$response - mean_train_IC50)^2)
    Baseline_MSE_FNs = mean((data_FN$response - mean_train_IC50)^2)
    
    Baseline_normalized_MSE_sensitive_CLs = MSE_sensitive_CLs / Baseline_MSE_sensitive_CLs
    Baseline_normalized_MSE_resistant_CLs = MSE_resistant_CLs / Baseline_MSE_resistant_CLs
    Baseline_normalized_MSE_correctly_classified_CLs = MSE_correctly_classified_CLs / Baseline_MSE_correctly_classified_CLs
    Baseline_normalized_MSE_incorrectly_classified_CLs = MSE_incorrectly_classified_CLs / Baseline_MSE_incorrectly_classified_CLs
    Baseline_normalized_MSE_TPs = MSE_TPs / Baseline_MSE_TPs
    Baseline_normalized_MSE_TNs = MSE_TNs / Baseline_MSE_TNs
    Baseline_normalized_MSE_FPs = MSE_FPs / Baseline_MSE_FPs
    Baseline_normalized_MSE_FNs = MSE_FNs / Baseline_MSE_FNs
  }
  if(!is.na(range_train_IC50)) {
    Range_normalized_MSE_sensitive_CLs = MSE_sensitive_CLs / range_train_IC50
    Range_normalized_MSE_resistant_CLs = MSE_resistant_CLs / range_train_IC50
    Range_normalized_MSE_correctly_classified_CLs = MSE_correctly_classified_CLs / range_train_IC50
    Range_normalized_MSE_incorrectly_classified_CLs = MSE_incorrectly_classified_CLs / range_train_IC50
    Range_normalized_MSE_TPs = MSE_TPs / range_train_IC50
    Range_normalized_MSE_TNs = MSE_TNs / range_train_IC50
    Range_normalized_MSE_FPs = MSE_FPs / range_train_IC50
    Range_normalized_MSE_FNs = MSE_FNs / range_train_IC50
  }
  
  return(list(
    "TP" = TP,
    "TN" = TN,
    "FP" = FP,
    "FN" = FN,
    "Sensitivity" = Sensitivity,
    "Specificity" = Specificity,
    "MSE_sensitive_CLs" = MSE_sensitive_CLs,
    "MSE_resistant_CLs" = MSE_resistant_CLs,
    "MSE_correctly_classified_CLs" = MSE_correctly_classified_CLs,
    "MSE_incorrectly_classified_CLs" = MSE_incorrectly_classified_CLs,
    "MSE_TPs" = MSE_TPs,
    "MSE_TNs" = MSE_TNs,
    "MSE_FPs" = MSE_FPs,
    "MSE_FNs" = MSE_FNs,
    "Median_SE_sensitive_CLs" = Median_SE_sensitive_CLs,
    "Median_SE_resistant_CLs" = Median_SE_resistant_CLs,
    "Median_SE_correctly_classified_CLs" = Median_SE_correctly_classified_CLs,
    "Median_SE_incorrectly_classified_CLs" = Median_SE_incorrectly_classified_CLs,
    "Median_SE_TPs" = Median_SE_TPs,
    "Median_SE_TNs" = Median_SE_TNs,
    "Median_SE_FPs" = Median_SE_FPs,
    "Median_SE_FNs" = Median_SE_FNs,
    "MAE_sensitive_CLs" = MAE_sensitive_CLs,
    "MAE_resistant_CLs" = MAE_resistant_CLs,
    "MAE_correctly_classified_CLs" = MAE_correctly_classified_CLs,
    "MAE_incorrectly_classified_CLs" = MAE_incorrectly_classified_CLs,
    "MAE_TPs" = MAE_TPs,
    "MAE_TNs" = MAE_TNs,
    "MAE_FPs" = MAE_FPs,
    "MAE_FNs" = MAE_FNs,
    "PCC_sensitive_CLs" = PCC_sensitive_CLs,
    "PCC_pvalue_sensitive_CLs" = PCC_pvalue_sensitive_CLs,
    "PCC_resistant_CLs" = PCC_resistant_CLs,
    "PCC_pvalue_resistant_CLs" = PCC_pvalue_resistant_CLs,
    "PCC_correctly_classified_CLs" = PCC_correctly_classified_CLs,
    "PCC_pvalue_correctly_classified_CLs" = PCC_pvalue_correctly_classified_CLs,
    "PCC_incorrectly_classified_CLs" = PCC_incorrectly_classified_CLs,
    "PCC_pvalue_incorrectly_classified_CLs" = PCC_pvalue_incorrectly_classified_CLs,
    "PCC_TPs" = PCC_TPs,
    "PCC_pvalue_TPs" = PCC_pvalue_TPs,
    "PCC_TNs" = PCC_TNs,
    "PCC_pvalue_TNs" = PCC_pvalue_TNs,
    "PCC_FPs" = PCC_FPs,
    "PCC_pvalue_FPs" = PCC_pvalue_FPs,
    "PCC_FNs" = PCC_FNs,
    "PCC_pvalue_FNs" = PCC_pvalue_FNs,
    "Adjusted_R2_sensitive_CLs" = Adjusted_R2_sensitive_CLs,
    "Adjusted_R2_resistant_CLs" = Adjusted_R2_resistant_CLs,
    "Adjusted_R2_correctly_classified_CLs" = Adjusted_R2_correctly_classified_CLs,
    "Adjusted_R2_incorrectly_classified_CLs" = Adjusted_R2_incorrectly_classified_CLs,
    "Adjusted_R2_TPs" = Adjusted_R2_TPs,
    "Adjusted_R2_TNs" = Adjusted_R2_TNs,
    "Adjusted_R2_FPs" = Adjusted_R2_FPs,
    "Adjusted_R2_FNs" = Adjusted_R2_FNs,
    "Baseline_normalized_MSE_sensitive_CLs" = Baseline_normalized_MSE_sensitive_CLs,
    "Baseline_normalized_MSE_resistant_CLs" = Baseline_normalized_MSE_resistant_CLs,
    "Baseline_normalized_MSE_correctly_classified_CLs" = Baseline_normalized_MSE_correctly_classified_CLs,
    "Baseline_normalized_MSE_incorrectly_classified_CLs" = Baseline_normalized_MSE_incorrectly_classified_CLs,
    "Baseline_normalized_MSE_TPs" = Baseline_normalized_MSE_TPs,
    "Baseline_normalized_MSE_TNs" = Baseline_normalized_MSE_TNs,
    "Baseline_normalized_MSE_FPs" = Baseline_normalized_MSE_FPs,
    "Baseline_normalized_MSE_FNs" = Baseline_normalized_MSE_FNs,
    "Range_normalized_MSE_sensitive_CLs" = Range_normalized_MSE_sensitive_CLs,
    "Range_normalized_MSE_resistant_CLs" = Range_normalized_MSE_resistant_CLs,
    "Range_normalized_MSE_correctly_classified_CLs" = Range_normalized_MSE_correctly_classified_CLs,
    "Range_normalized_MSE_incorrectly_classified_CLs" = Range_normalized_MSE_incorrectly_classified_CLs,
    "Range_normalized_MSE_TPs" = Range_normalized_MSE_TPs,
    "Range_normalized_MSE_TNs" = Range_normalized_MSE_TNs,
    "Range_normalized_MSE_FPs" = Range_normalized_MSE_FPs,
    "Range_normalized_MSE_FNs" = Range_normalized_MSE_FNs
  ))
}
