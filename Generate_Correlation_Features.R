# train_cls: data.frame with single column containing ln(IC50) values of cell lines for one drug, rownames are cell line IDs/names
# exp_matrix: data.frame containing gene expression values, rownames are cell line IDs/names, columnnames are gene names
# returns: data.frame with single column containing Pearson correlation coeffients between gene expression and ln(IC50) values ordered descendingly by absolute magnitude, rownames are gene names
compute_correlations_for_given_samples = function(train_cls, exp_matrix) {
  colnames(train_cls) = "IC50"
  
  train_data = merge(train_cls, exp_matrix, by = 0)
  rownames(train_data) = train_data$Row.names
  train_data$Row.names = NULL
  
  correlations = data.frame(sapply(train_data, function(x) cor(train_data$IC50, x)))
  colnames(correlations) = "correlation"
  correlations = correlations[rownames(correlations) != "IC50", , drop = F]
  correlations = correlations[order(-abs(correlations$correlation)), , drop = F]
  
  return(correlations)
}
