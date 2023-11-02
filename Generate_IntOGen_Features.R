# intogen_data: data.frame with cancer drivers obtained from IntOGen website (https://www.intogen.org/download; file Unfiltered_driver_results_05.tsv)
# exp_genes: vector of gene names (only genes that are listed in this vector will be contained in the final list)
# returns: sorted list of IntOGen cancer driver genes

intogen_data = intogen_data[intogen_data$FILTER == "PASS",]

intogen = data.frame(SYMBOL = unique(intogen_data$SYMBOL), NUM_COHORTS = NA, TIER = NA)

# assign number of cohorts and best tier to each gene
for(g in 1:nrow(intogen)) {
  gene = intogen$SYMBOL[g]
  cohorts = unique(intogen_data$NUM_COHORTS[intogen_data$SYMBOL == gene])
  tier = min(intogen_data$TIER[intogen_data$SYMBOL == gene])
  intogen$NUM_COHORTS[g] = cohorts
  intogen$TIER[g] = tier
}

# order by tier and number of cohorts
intogen = intogen[order(intogen$TIER, -intogen$NUM_COHORTS),]
intogen$SYMBOL = as.character(intogen$SYMBOL)

# remove genes that are not present in expression matrix
intogen = intogen[intogen$SYMBOL %in% exp_genes,]

