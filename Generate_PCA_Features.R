# train_cls: vector of names/IDs of training cell lines
# test_cls: vector of names/IDs of test cell lines
# exp_matrix: data.frame containing gene expression values, rownames are cell line names/IDs, columnnames are gene names
# returns: data.frames containing first 500 principal components for training/test cell lines computed on expression data of training cell lines, rownames are cell line names/IDs, columnnames are principal components
compute_PCA_for_given_samples = function(train_cls, test_cls, exp_matrix) {
  
  train_data = exp_matrix[train_cls,]
  test_data = exp_matrix[test_cls,]
  
  pca = prcomp(train_data, center = TRUE,scale. = TRUE, rank. = 500)
  
  features_train = data.frame(predict(pca, newdata = train_data))
  features_test = data.frame(predict(pca, newdata = test_data))
  
  return(list("train_matrix" = features_train, "test_matrix" = features_test))
}
