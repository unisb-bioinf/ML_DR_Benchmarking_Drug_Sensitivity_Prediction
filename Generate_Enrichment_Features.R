# input files can be obtained through enrichment analysis performed using GeneTrail webservice (https://genetrail.bioinf.uni-sb.de/)
# required columns: X.Name (gene name), Regulation_direction (1 for enriched, 0 for depleted), P.value (adjusted enrichment p-value)
# up_file: categories correspond to all cell lines in which investigated genes are upregulated
# down_file: categories correspond to all cell lines in which investigated genes are downregulated
# returns: data.frame with single ordered column containing p-values, rownames are gene names
sort_genes_by_enrichment_pvalue = function(up_file, down_file) {
  
  # split data in four conditions (up-/downregulated and enriched/depleted)
  up_enriched = up_file[up_file$Regulation_direction == 1,]
  up_depleted = up_file[up_file$Regulation_direction == 0,]
  down_enriched = down_file[down_file$Regulation_direction == 1,]
  down_depleted = down_file[down_file$Regulation_direction == 0,]
  
  # combine all data into one data frame
  up_enriched$Note = "up_enriched"
  down_enriched$Note = "down_enriched"
  up_depleted$Note = "up_depleted"
  down_depleted$Note = "down_depleted"
  all_enrichments = rbind(up_enriched, down_enriched, up_depleted, down_depleted)
  
  # rank genes for each of the 4 cases separately by p-value
  library(dplyr)
  all_enrichments =  all_enrichments %>% group_by(Note) %>% mutate(rank = order(order(P.value, decreasing=F)))
  
  # for each gene: only keep entries with smallest rank to avoid duplicate selection
  library(data.table)
  all_enrichments = setDT(all_enrichments)[ , .SD[which.min(rank)], by = X.Name]
  
  # rank genes again
  all_enrichments =  all_enrichments %>% group_by(Note) %>% mutate(rank = order(order(P.value, decreasing=F)))
  
  # order by rank, break ties by p-value
  all_enrichments = all_enrichments[order(all_enrichments$rank, all_enrichments$P.value),]
  
  # convert back to data frame and format correctly
  all_enrichments = data.frame(all_enrichments, row.names = all_enrichments$X.Name)
  all_enrichments = all_enrichments[, "P.value", drop = F]
  
  return(all_enrichments)
}
