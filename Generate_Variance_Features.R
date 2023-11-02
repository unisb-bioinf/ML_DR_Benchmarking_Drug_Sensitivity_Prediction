# train_cls: vector of names/IDs of training cell lines
# exp_matrix: data.frame containing gene expression values, rownames are cell line IDs/names, columnnames are gene names
# returns: data.frame with single column containing variance of gene expression, ordered descendingly by magnitude, rownames are gene names
compute_variance_for_given_samples = function(train_cls, exp_matrix) {

  train_data = exp_matrix[train_cls,]
  
  variances = data.frame(sapply(train_data, function(x) var(x)))
  colnames(variances) = "variance"
  variances = variances[order(-variances$variance), , drop = F]
  
  return(variances)
}
