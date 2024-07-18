setwd("/Users/johnhood/Research/Schein/HCPCCL")
library(kernlab)
set.seed(123)
num <- 4
c <- 15 
d <- 3
k <- 3
heldouts <- seq(0.2, 0.5, by=0.1)
f_groups <- factor(groups)
#Ours <- read.table(paste0("results/thresholding/thresholding/A", num, "_C", c, "_D", d, "_K", k,"_", seed, ".csv"), sep = ",")

prOurs <- matrix(0, 4, 10)
prCTF <- matrix(0, 4, 10)
prMT <- matrix(0, 4, 10)
prPCA <- matrix(0, 4, 10)

for (j in 1:10){
  seed <- j + 379
  for (num in 1:4){
    hp <- heldouts[num]
    load(paste0("results/FARMM/", hp, "fit_CTF.RData"))
    load(paste0("results/FARMM/", hp, "fit_microTensor.RData"))
    load(paste0("results/FARMM/", hp, "fit_pca.RData"))
    Ours <- read.table(paste0("results/thresholding/thresholding/A", num, "_C", c, "_D", d, "_K", k,"_", seed, ".csv"), sep = ",")
    CTF <- l_fit_ctf[[j]]$s
    MT <- l_fit_microTensor[[j]]$s
    PCA <- l_fit_pca[[j]]$s
    Ours <- scale(Ours)
    CTF <- scale(CTF)
    MT <- scale(MT)
    PCA <- scale(PCA)
    Our_spec <- specc(as.matrix(Ours), centers=3)@.Data
    CTF_spec <- specc(as.matrix(CTF), centers=3)@.Data
    MT_spec <- specc(as.matrix(MT), centers=3)@.Data
    PCA_spec <- specc(as.matrix(PCA), centers=3)@.Data
    
    labels <- Our_spec
    labels_CTF <- CTF_spec
    labels_MT <- MT_spec
    labels_PCA <- PCA_spec
    
    prOurs[num,j] <- mutinformation(labels, f_groups)
    prCTF[num,j] <- mutinformation(labels_CTF, f_groups)
    prMT[num,j] <- mutinformation(labels_MT, f_groups)
    prPCA[num,j] <- mutinformation(labels_PCA, f_groups)
  }
}

prMTs <- apply(prMT, 1, median)
prPCAs <- apply(prPCA, 1, median)
prCTFs <- apply(prCTF, 1, median)
prOursF <- apply(prOurs, 1, median)
#standard errors
prMTse <- apply(prMT, 1, sd)/sqrt(10)
prPCAse <- apply(prPCA, 1, sd)/sqrt(10)
prCTFse <- apply(prCTF, 1, sd)/sqrt(10)
prOursse <- apply(prOurs, 1, sd)/sqrt(10)

plot(seq(0.2, 0.5, by=0.1), prOursF, col="black", ylim=c(0, 1), xlab="Proportion Data Missing", ylab="Mutual Information", main="", pch=16)
points(seq(0.2, 0.5, by=0.1), prMTs, pch=16, col="gray")
points(seq(0.2, 0.5, by=0.1), prPCAs, pch=16, col="lightgreen")
points(seq(0.2, 0.5, by=0.1), prCTFs, pch=16, col="lightblue")
#plot standard errors as error bars
arrows(seq(0.2, 0.5, by=0.1), prOursF - prOursse, seq(0.2, 0.5, by=0.1), prOursF + prOursse, col="black", angle=90, code=3)
arrows(seq(0.2, 0.5, by=0.1), prMTs - prMTse, seq(0.2, 0.5, by=0.1), prMTs + prMTse, col="gray", angle=90, code=3)
arrows(seq(0.2, 0.5, by=0.1), prPCAs - prPCAse, seq(0.2, 0.5, by=0.1), prPCAs + prPCAse, col="lightgreen", angle=90, code=3)
arrows(seq(0.2, 0.5, by=0.1), prCTFs - prCTFse, seq(0.2, 0.5, by=0.1), prCTFs + prCTFse, col="lightblue", angle=90, code=3)



legend("topright", legend=c("Ours", "microTensor", "PCA", "CTF"), col=c("black", "gray", "lightgreen", "lightblue"), lty=1:1, cex=0.8)






