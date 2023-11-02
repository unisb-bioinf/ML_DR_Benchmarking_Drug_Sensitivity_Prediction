# path_to_matlab: path to bin folder of matlab installation
library(matlabr)
options(matlab.path = path_to_matlab)
stopifnot(have_matlab())

# train_cls: vector of names/IDs of training cell lines
# test_cls: vector of names/IDs of test cell lines
# exp_matrix: data.frame containing gene expression values, rownames are cell line names/IDs, columnnames are gene names
# NOTE: the following paramaters need to be set directly in the matlab code below
# path_to_pasl: path to folder containing code of PASL
# pathway_matrix: path to binary matrix where rows are pathways, columns are genes (1: gene is part of pathway, 0: gene is not part of pathway)
# pathway_names: path to list with pathway names (number of names here must correspond to number of rows in pathway_matrix and order must be identical)
# returns: data.frames containing 500 ordered PASL features training/test cell lines, rownames are cell line names/IDs, columnnames are PASL features
# returns: duration of PASL computations, number of unique pathways in the computed features
compute_PASL_for_given_samples = function(train_cls, test_cls, exp_matrix) {
  
  train_data = exp_matrix[train_cls,]
  test_data = exp_matrix[test_cls,]
  
  write.table(train_data, 'temp_train_data.txt', row.names = T, col.names = NA, quote = F)
  write.table(test_data, 'temp_test_data.txt', row.names = T, col.names = NA, quote = F)
  
  write.table(train_data, 'temp_train_data_no_names.txt', row.names = F, col.names = F, quote = F)
  write.table(test_data, 'temp_test_data_no_names.txt', row.names = F, col.names = F, quote = F)
  
  # a1: Number of atoms (inference phase)
  # a2: Number of atoms (discovery phase)
  # t: threshold for reordering the genesets (0.9)
  # lambda: Box - Cox normalization parameter (1/3)
  # m: Number of non-zeros per atom of discovery phase (2000)
  code = c(
    "clear; close all; clc;",
    
    "diary diary.txt;",
    
    "addpath(path_to_pasl)",
    
    "X_train = readmatrix('temp_train_data_no_names.txt');",
    "G = readmatrix(pathway_matrix);",
    "geneset_names = readlines(pathway_names);",
    "geneset_names(end)=[];",
    
    "a1      = 500;",
    "a2      = 0;",
    "t       = 0.9;",
    "lambda  = 1/3;",
    "m       = 2000;",
    "verbose = 0;",
    
    "tic;",
    
    "[D, L, selected_genesets, mu, sigma] = ...
PASL(X_train, G, geneset_names, a1, a2, t, lambda, m, verbose);",
    
    "duration = toc;",
    
    "writematrix(D, 'temp_D.txt');",
    "writematrix(L, 'temp_L.txt');",
    "writetable(struct2table(selected_genesets), 'temp_selected_genesets.txt');",
    "sigma_mu = [array2table(transpose(mu), 'VariableNames', {'mu'}), array2table(transpose(sigma), 'VariableNames', {'sigma'})];",
    "writetable(sigma_mu, 'temp_mu_and_sigma.txt');",
    "fileID = fopen('temp_duration_seconds.txt','w');",
    "fprintf(fileID,'%f',duration);",
    "fclose(fileID);",
    
    "X_test = readmatrix('temp_test_data_no_names.txt');",
    "X_test = X_test - repmat(mu, size(X_test, 1), 1);",
    "X_test = X_test ./ repmat(sigma, size(X_test, 1), 1);",
    "L_test = X_test * pinv(D);",
    
    "writematrix(L_test, 'temp_L_test.txt');",
    
    "diary off;"
  )
  
  start_time = Sys.time()
  res = run_matlab_code(code)
  end_time = Sys.time()
  stopifnot(res == 0)
  duration = difftime(end_time, start_time, units = 'sec')  
  cat("duration: ", duration, "  ")
  
  train_features = data.frame(fread('temp_L.txt'), stringsAsFactors = F)
  test_features = data.frame(fread('temp_L_test.txt'), stringsAsFactors = F)
  genesets = read.csv('temp_selected_genesets.txt', stringsAsFactors = F)
  duration = readChar('temp_duration_seconds.txt', file.info('temp_selected_genesets.txt')$size)
  
  rownames(train_features) = rownames(train_data)
  rownames(test_features) = rownames(test_data)
  colnames(train_features) = colnames(test_features) = paste0("F", c(1:length(genesets$geneset_names)), "_", genesets$geneset_names)
  
  return(list("train_matrix" = train_features, "test_matrix" = test_features, "duration" = duration, "unique_features" = length(unique(genesets$geneset_names))))
}
