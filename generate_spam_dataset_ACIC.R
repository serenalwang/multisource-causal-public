# This script was adapted from ACIC 2019 for the paper
# Multi-Source Causal Inference Using Control Variates (authors redacted).
# ORIGINAL ACIC 2019 COMMENTS:
# Simulate low dimensional data based on spam email dataset
# (binary outcome)
# Author and date of original ACIC 2019 data generating process: 
# Susan Gruber, sgruber@putnamds.com
# July 5, 2019
# Sources of spam email dataset:
   # (a) Creators: Mark Hopkins, Erik Reeber, George Forman, Jaap Suermondt
        # Hewlett-Packard Labs, 1501 Page Mill Rd., Palo Alto, CA 94304
   # (b) Donor: George Forman (gforman at nospam hpl.hp.com)  650-857-7835
   # (c) Generated: June-July 1999
# Step 1: modify the covariates to make them better suited for the Data challenge
# Step 2:  Develop four DGPs based on the modified covariate set, W.
#          and generate 100 datasets for each, with sample size = 500
#          (Consider the 30,000 original observations to be the source population)
#     Mod 1. Simple, straightforward, parametric main terms models  
#     Mod 2. Treatment and outcome are complex functions of measured covariates
#     Mod 3.  Poor overlap
#     Mod 4: Treatment effect heterogeneity, complex models.
#------------------------------------------------------------------------------
#  R 3.6.0 changed the "sample" function
# This re-sets it to the old behavior, same as when the ACIC 2019 datasets
# were generated
if (getRversion() < '3.6.0') {
    RNGkind(sample.kind="Rounding")
}
# 
##------------------------------------------------------------------
# Step 1. Read in the data and create modified covariate set
#------------------------------------------------------------------
d <- read.csv("spambase.csv", header =  FALSE)
# columns 1- 54 are continuous word frequencies (percentages between 0 and 100) 
# columns 55-57 count capital letters in various ways.
# discard them, and  replace with  a binary indicator of more than the mean number of capital letters
# column 58 is the indicator of whether message is spam or not
# Create datasets of size 500 x 25 and let people analyze them.
# Keep 20 from 1- 54, and also 55
d$A  <- as.integer(d[,57] > mean(d[,57]))  # 24% are above the mean - think of this as treatment
keep <- c(58,59,  5:25, 55)
d <- d[,keep]
colnames(d) <- c("Y", "A", paste0("V", 1:(ncol(d)-2)))
d$V22 <- log(d$V22)
#------------------------------------------------------------------
# Step 2:  Develop four DGPs 
#------------------------------------------------------------------
	n <- nrow(d)
	m.or <- glm(Y ~ ., data = d, family = "binomial")
#------------------------------------------------------------------
# Modification 1 : very close to the logistic regression models fitted to the actual data
# parametric regression as starting point
#------------------------------------------------------------------
d.mod1 <- d
set.seed(10)
# pscore model
beta.mod1 <- c(-3, .33*colMeans(d.mod1[,-(1:2)])/ apply(d.mod1[,-(1:2)], 2, sd))
g1W.mod1 <- plogis(cbind(1, as.matrix(d.mod1[,-(1:2)])) %*% beta.mod1)
d.mod1$A <- rbinom(n, 1, g1W.mod1)
# Use glm model as truth for this simulation
beta.Q.mod1 <- -coef(m.or)
beta.Q.mod1[1] <- .25
beta.Q.mod1[2] <- 1  # coef on A
logit.drs.mod1 <-  cbind(1, as.matrix(d[,-(1:2)])) %*% beta.Q.mod1[-2]
d.mod1$Y <- rbinom(n, 1, plogis(d.mod1$A + logit.drs.mod1 ))
# True population ATE
psi0.mod1 <- mean(plogis(1 + logit.drs.mod1)  - plogis(logit.drs.mod1))
# generate datasets
set.seed(10)
niter <- 100
n.b <- 10000
mod1_files = 1 # Only generate one file
for (i in mod1_files){
	b <- sample(1:n, n.b, replace = TRUE)
	d.mod1$A[b] <- rbinom(n.b, 1, g1W.mod1[b])
	d.mod1$Y[b] <- rbinom(n.b, 1, plogis(d.mod1$A[b] + logit.drs.mod1[b] ))
	write.csv(d.mod1[b,], file = paste0("spam_binMod1", i , ".csv"), row.names = FALSE)
}
# Generate a larger version to apply selection bias.
n.b <- 100000
for (i in mod1_files){
	b <- sample(1:n, n.b, replace = TRUE)
	d.mod1$A[b] <- rbinom(n.b, 1, g1W.mod1[b])
	d.mod1$Y[b] <- rbinom(n.b, 1, plogis(d.mod1$A[b] + logit.drs.mod1[b] ))
	write.csv(d.mod1[b,], file = paste0("spam_binMod1_large", i , ".csv"), row.names = FALSE)
}