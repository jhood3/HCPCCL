setwd("/Users/johnhood/Research/Schein/HCPCCL/FARMM")
load("df_samples.RData")
#load("df_subject.RData")
#load("mat_otu_genus.RData")
load("X_array.RData")
subjects <- read.table("20200619_farmm_metadata.tsv", header=TRUE, sep="\t")
#reduce to one line per SubjectID
subjects <- subjects[!duplicated(subjects$SubjectID), ]
#Y <- read.table("table-matched.txt", header=TRUE)
X <- X_array
dim(X)
#sparse_X = as(X, "sparseArray")
sparse_X <- matrix(0, nrow=sum(X > 0, na.rm=TRUE) + sum(is.na(X)), ncol = 4)
#save X as csv file
counter <- 1
for (i in 1:(dim(X)[1])){
  for (j in 1:(dim(X)[2])){
    for (k in 1:(dim(X)[3])){
      if (sum(X[i, j, k] > 0, na.rm = TRUE) > 0){
    sparse_X[counter, ] <- t(c(i, j, k, X[i, j, k]))
    counter <- counter + 1
      }
      if (is.na(X[i, j, k])){
      sparse_X[counter, ] <- t(c(i, j, k, -1))
      counter <- counter + 1
    }
}
  }
}
groups <- subjects$study_group[1:30]
#save sparse_X as csv file
write.csv(sparse_X, "sparse_XFARMM.csv", row.names=FALSE)
write.csv(groups, "groups.csv", row.names=FALSE)
