library(stats)
library(lme4)
library(PRROC)
library(umap)
library(randomForest)
#setwd("/Users/johnhood/Research/Schein/Allocation/AllocaDA/FARMM")
setwd("/Users/johnhood/Research/Schein/Allocation/FARMM")
Cs <- seq(3, 23, by=4)
c <- 15
d <- 3
k <- 3
nums = 0:4
seeds = 380:389
prs <- matrix(0, length(seeds), length(nums)) 
prs <- array(0, c(length(seeds), length(Cs), length(nums)))
for (s in 1:length(seeds)){
  for (n in 1:length(nums)){
  for (m in 1:length(Cs)){
    num <- nums[n]
    seed <- seeds[s]
    c <- Cs[m]
    #try opening file, if doesn't work, then continue
    try({
A <- read.table(paste0("A", num, "_C", c, "_D", d, "_K", k,"_", seed, ".csv"), sep = ",")
})
#A <- read.table(paste0("A", num, "_C", c, "_D", d, "_K", k,".csv"), sep = ",")

A <- as.data.frame(A)
vegan <- as.numeric(groups == "EEN")
#fit <- glm(vegan ~ V1 + V2 + V3, family = binomial, data=A)
true <- vegan
#pred <- fit$fitted.values
#print(roc(true, pred))
#ci <- ci.auc(roc(true, pred), method="bootstrap", ci=0.95)[c(1,3)]
#leave-one-out fit
loo <- rep(NA, nrow(A))
#loo_r <- rep(NA, nrow(A))
for(i in 1:nrow(A)){
  #fitr <- randomForest(factor(vegan[-i]) ~ V1 + V2 + V3, data = as.data.frame(A[-i,]))
  fitl <- glm(vegan[-i] ~ V1 + V2 + V3, family = binomial, data = as.data.frame(A[-i,]))
  loo[i] <- predict(fitl, newdata = A[i,])
  #loo_r[i] <- predict(fitr, newdata = A[i,])
}
pr <- pr.curve(1 - loo, weights.class0 = 1 - true)
#pr_r <- pr.curve(1-loo_r, weights.class0 = 1-true)
prs[s,m, n] <- pr$auc.integral
}
  }
}

#get median by column of prs, only including nonzero values
mean_prs <- apply(1 - prs, c(2,3), median)
plot(Cs, mean_prs[,1], type='l', lwd=2, ylim=c(0.0,0.25), xlab="Number of gene-specific components", ylab="AUC-PR error", col='blue')
points(Cs, mean_prs[,1], pch=16, col='blue')
for (m in 2:length(nums)){
lines(Cs, mean_prs[,m], type='l', lwd=2, lty=m, col=m)
points(Cs, mean_prs[,m], pch=16, col=m)
}

#confidence interval for columns
#lower quantile
#iqr_prs <- apply(prs, 2, IQR)
#upper quantile
iqr_prs <- apply(prs, 2, IQR)
lines(Cs, mean_prs + iqr_prs/2, lty=2, col='blue')
lines(Cs, mean_prs - iqr_prs/2, lty=2, col='blue')

T0 <- read.table(paste0("T0_C", c, "_D", d, "_K", k, ".csv"), sep = ",")
par(mfrow=c(1,1))
#want to plot time series, where lines to be dotted and lined, and shade under the curve
#shade under curve

plot(T0[,1]/max(T0[,1]), col='red', type='l', lty=1, lwd=2, ylim=c(0,1), xlab="Time", ylab="Proportion of loading")
points(T0[,1]/max(T0[,1]), col='red', pch=16)
lines(T0[,2]/max(T0[,2]), col='blue', type='l', lty=1, lwd=2)
points(T0[,2]/max(T0[,2]), col='blue', pch=16)
lines(T0[,3]/max(T0[,3]), col='darkgreen', type='l', lty=1, lwd=2)
points(T0[,3]/max(T0[,3]), col='darkgreen', pch=16)

par(mfrow=c(1,3))
plot(A[,1], A[,2], col=factor(groups), pch=19, xlab="V1", ylab="V2")
plot(A[,1], A[,3], col=factor(groups), pch=19, xlab="V1", ylab="V3")
plot(A[,2], A[,3], col=factor(groups), pch=19, xlab="V2", ylab="V3")
uA <- umap(A)
par(mfrow=c(1,1))
plot(uA$layout[,1], uA$layout[,2], pch=19, xlab="UMAP1", ylab="UMAP2", col=factor(groups))
#load("df_samples.RData")
par(mfrow=c(1,1))
G0 <- read.table(paste0("G0_C", c, "_D", d, "_K", k, ".csv"), sep = ",")
umapG <- umap(G0)
names <- dimnames(X)[[1]]
s_names <- substr(names, 1, 18)
plot(umapG$layout[,1], umapG$layout[,2], pch=19, xlab="UMAP1", ylab="UMAP2", col=factor(s_names))

meta_data <- read.table("20200619_farmm_metadata.tsv", sep = "\t", header = TRUE)

#unique names, up to first 10 characters
c <- 15

core_values <- read.table(paste0("core0_C", c, "_D", d, "_K", k, ".csv"), sep = ",")
indices_QM <- read.table(paste0("indices0_C", c, "_D", d, "_K", k, ".csv"), sep = ",")
core <- array(0, c(c,d,k))
for (i in 1:(dim(indices_QM)[1])){
  core[indices_QM[i,1], indices_QM[i,2], indices_QM[i,3]] <- core_values$V1[i]
}

#time slices
par(mfrow=c(1,k))
for (i in 1:k){
  image(core[,,i], xlab="gene", ylab="subject", main=paste("Time PC", i))
}
#subject slices
par(mfrow=c(1,d))
for (i in 1:d){
  image(core[,i,], xlab="gene", ylab="time", main=paste("Subject PC", i))
}

#gene slices
par(mfrow=c(3,c/3))
for (i in 1:c){
  image(core[i,,], xlab="", ylab="", main="", xaxt='n', yaxt='n')#, xlim = c(0,1), ylim = c(0,1))
  #want no tick marks or axis numbering
  #axis(1, at = c(0,1), labels = FALSE)
  #axis(2, at = c(0,1), labels = FALSE)
  }

#time by subject, summed over gene
par(mfrow=c(1,1))
image(apply(core, c(2,3), sum), xlab="subject", ylab="time", main="Time by Subject")

#l1 <- read.table("l1.csv", sep = ",")
#l2 <- read.table("l2.csv", sep = ",")
#l3 <- read.table("l3.csv", sep = ",")
#image(as.matrix(l1))
#image(as.matrix(l2))
#image(as.matrix(l3))

plot(mean_prs, type='l')

