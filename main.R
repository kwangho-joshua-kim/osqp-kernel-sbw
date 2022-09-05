#####################################################################################  
#        R function for OSQP-based kernel SBW with Nystrom approximation            #                
#        Kwhangho Kim (kkim@hcp.med.harvard.edu)                                    #
#                                                                                   #
#####################################################################################

library(Matrix)
library(MASS)
library(matrixStats)
library(osqp)
library(kbal)
library(RSpectra)
library(Rcpp)
library(RcppArmadillo)

source("utils.R")
# sourceCpp('RBF_kernel_C.cpp')
sourceCpp('RBF_kernel_C_parallel.cpp')



# --------------------------------------------------------------------#
#                    example (Hainmueller, 2012)                      #
# --------------------------------------------------------------------#
n=1e+6
sig.123 <- diag(c(2,1,1))
sig.123[1,2] <- 1; sig.123[1,3] <- -1; sig.123[2,3] <- -0.5;
sig.123 <- forceSymmetric(sig.123)
beta_coef <- c(1,2,-2,-1,-0.5,1)

X.123 <- as.matrix(mvrnorm(n, mu = rep(0,3), Sigma = sig.123))
X.4 <- runif(n,-3,3)
X.5 <- rchisq(n,1)
X.6 <- rbinom(n,1,0.5)
X <- cbind(X.123, X.4, X.5, X.6)
A <- ifelse(X %*% matrix(beta_coef, ncol = 1) + rnorm(n,0,30) > 0,1,0)
Y <- (X.123[,1] + X.123[,2] + X.5)^2 + rnorm(n,0,1)

# Step 1) Construct the approximated kernel basis
# + the parameter c shouldn't be too large; typically recommend using 50~500 
#   (c=100 works well enough in the above example)
# + When you're using many covariates consider setting dim.reduction=TRUE
ptm <- proc.time()
X_ <- kernel.basis(X,A,Y,
                   kernel.approximation=TRUE,
                   c = 100)
et <- proc.time() - ptm

# Step 2) Compute the sbw via osqp
# + Use smaller tolerance levels for higher accuracy
# + Polishing is an additional algorithm step where OSQP tries to compute a high-accuracy solution
high.acc.setting <- osqpSettings(alpha = 1.5, verbose = FALSE, 
                                 warm_start = FALSE, # use warm start when you iterate over multiple deltas
                                 eps_abs = 1e-3, # use more strict absolute tolerance if needed
                                 eps_rel = 1e-3, # use more strict relative tolerance if needed
                                 polish=FALSE) # solution polishing
ptm <- proc.time()
res <- osqp_kernel_sbw(X,A,Y, 
                       delta.v=1e-4, X_=X_,
                       osqp.setting=high.acc.setting)
et <- proc.time() - ptm
res[[1]]$t


# Step 1 & 2 together:
ptm <- proc.time()
res <- osqp_kernel_sbw(X,A,Y, 
                       delta.v=1e-4, 
                       c = 100,
                       osqp.setting=high.acc.setting)
et <- proc.time() - ptm
res[[1]]$t