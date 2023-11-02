# genes: vector of gene names to sample from
# folder: path to folder where results will be saved

set.seed(2208)
list_lengths = c(1:25, 50, 100, 200, 300, 400, 500) # number of genes in random lists
number_of_lists = 10 # number of random lists generated for each length

dir.create(folder)

for(list_length in list_lengths) {
  cat(list_length, " ")
  list_folder = paste0(folder, "Length_", list_length, "/")
  dir.create(list_folder)
  for(n in 1:number_of_lists) {
    list = sample(x = genes, size = list_length, replace = F)
    write.table(list, paste0(list_folder, "Length_", list_length, "_List_", n, ".txt"), quote = F, col.names = F, row.names = F)
  }
}
