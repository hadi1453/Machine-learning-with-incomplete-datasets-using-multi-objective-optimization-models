library(e1071)
library(ISLR)
library(caret)
library(fclust)
library(cluster)
library(erer)
library(mlbench)
library(DescTools)
library(aricode)

memory.limit(size = 56000)
options(scipen=10000)

Path_Imp_SVM <- "C:/Users/hkhorshidi/Documents/CIS Unimelb/Research topics/Robust Data Mining/Imputation-SVM"

data(iris)
#iris_N <- iris
iris_N <- apply(iris[,-5], 2, FUN = function(x) (x-min(x))/(max(x)-min(x)))

data(Sonar)



zoo <- read.csv(file.path(Path_Imp_SVM,"zoo.csv"))


# Fit SVM classification model
SVM_fit = function(df, n){
  z <- unlist(lapply(df, is.numeric))
  df[,z] <- apply(df[,z], 2, FUN = function(x) {(x-min(x))/(max(x)-min(x))})
  set.seed(1)
  train <- sample(nrow(df), nrow(df)/2)
  df_tr <- df[train,]
  df_te <- df[-train,]
  svm_df_tr <- svm(as.factor(df_tr[,n])~., data = df_tr, kernel = "radial")
  #svm_df_tr <- train(as.factor(df_tr[,n])~., data = df_tr, method = "svmRadial")
  svm_pred_tr <- predict(svm_df_tr, df_tr)
  Accu_tr <- mean(df_tr[,n]==svm_pred_tr)
  svm_pred_te <- predict(svm_df_tr, df_te)
  Accu_te <- mean(df_te[,n]==svm_pred_te)
  return(c(Accu_tr, Accu_te))
}
tune(svm, Purchase ~ ., data = OJ.train, kernel = "linear")

SVM_fit(wine, n)

# Randomly remove p% missing values
Random_missing <- function(df, n, p){
  D <- dim(df[,-n])
  MS_lenegth <- round(p*D[1]*D[2])
  MS_list <- matrix(NA,MS_lenegth, 3)
  for (a in 1:MS_lenegth) {
    i <- sample(1:D[1],1)
    j <- sample(1:D[2],1)
    MS_list[a,] <- c(i,j,df[i,j])
    df[i,j] <- NA
  }
  df1 <- df[complete.cases(df),]
  return(df1)
}

# Generate random missing value and classify using SVM
z <- unlist(lapply(df, is.numeric))
df[,z] <- apply(df[,z], 2, FUN = function(x) {(x-min(x))/(max(x)-min(x))})
set.seed(1)
train <- sample(nrow(df), nrow(df)/2)
df_tr <- df[train,]
df_te <- df[-train,]

df_tr1 <- Random_missing(df_tr, n, 0.15)

svm_df_tr <- svm(as.factor(df_tr1[,n])~., data = df_tr1, kernel = "radial")
svm_pred_tr <- predict(svm_df_tr, df_tr1)
Accu_tr <- mean(df_tr1[,n]==svm_pred_tr)
svm_pred_te <- predict(svm_df_tr, df_te)
Accu_te <- mean(df_te[,n]==svm_pred_te)

############################################################################################
### Generate missing values
# Simple overall 
set.seed(1)
for(d in c("iris_N")){
  for(r in c(0.01, 0.05, 0.1, 0.2, 0.35, 0.5)){
    for(n in 1:1){
      # simple
      DS <- get(d)
      DS_row <- round(r*nrow(DS)*ncol(DS))
      DS_row <- if(DS_row > nrow(DS)){round(0.06*nrow(DS)*ncol(DS))}else{DS_row}
      val_list <- matrix(NA, DS_row, 3)
      val_row <- sample(1:nrow(DS), DS_row)
      for(i in 1:DS_row){
        val_col <- sample(1:ncol(DS), 1)
        val_list[i, 1] <- val_row[i]
        val_list[i, 2] <- val_col
        val_list[i, 3] <- DS[val_row[i],val_col]
        DS[val_row[i],val_col] <- NA
      }
      assign(paste("Simple_",d,"_",r,"_",n,sep = ""),DS)
      assign(paste("Values_Simple_",d,"_",r,"_",n,sep = ""),val_list)
    }
  }
}

############################################################################################
### Imputation using fuzzy clustering using Silhuette optimization + centroid genetic coding

# Auxiliary function for soft allocation of data points into clusters using fuzzy clustering
Membership_Distance <- function(Z, Xr, m){
  Dis <- matrix(0,nrow=nrow(Z), ncol=nrow(Xr))
  Umat <-  matrix(0,nrow=nrow(Z), ncol=nrow(Xr))
  for (j in 1:nrow(Xr)) {
    Dis[,j] <-  matrix(apply(Z, 1, FUN = function(x) {sum((x-Xr[j,])^2)}),,1)
  }
  for (i in 1:nrow(Dis)) {
    for (j in 1:ncol(Dis)) {
      U <- 0
      if(!0 %in% Dis[i,]){
        for (h in 1:ncol(Dis)) {
          U <- U + if(Dis[i,h]>0){(Dis[i,j]/Dis[i,h])^(1/(m-1))}else if(Dis[i,j]){1}
        }
        U <- 1/U
      }else if(Dis[i,j]==0){U <- 1}
      Umat[i,j] <- U
    }
  }
  return(list(Distance=Dis, Membership=Umat))
}

# Auxiliary function for objective function for imputation (ASW)
ASW_func <- function (Z_imp, U){
  #a <- SIL(Z_imp, U)
  #A <- a$sil
  #A <- SIL.F(Z_imp, U)
  c <- cl.memb(U)[,-2]
  if(length(table(c)) == 1){c[1] <- 11} # To avoid having just 1 cluster
  a <- silhouette(c, dist(Z_imp))
  A <- (mean(a[,3])+1)/2
  return(A)
}

# Auxiliary function for objective function for imputation (CCA)
CCA_func <- function (X, Y){
  Rx <- nrow(X)
  Ry <- nrow(Y)
  if(Rx > Ry){X <- X[sample(1:Rx, Ry),]}else if(Ry > Rx){Y <- Y[sample(1:Ry, Rx),]}
  D <- ncol(X)
  VarX <- abs(matrix(1, 1, D)%*%(var(X)%*%matrix(1, D, 1)))
  VarY <- abs(matrix(1, 1, D)%*%(var(Y)%*%matrix(1, D, 1)))
  CovXY <- abs(matrix(1, 1, D)%*%(cov(X, Y)%*%matrix(1, D, 1)))
  CorXY <- CovXY/sqrt(VarX*VarY)/0.1
  return(CorXY)
}

# Auxiliary function for objective function for imputation (Variance ratio)
VR_func <- function (X, Y){
  D <- ncol(X)
  VarX <- abs(matrix(1, 1, D)%*%(var(X)%*%matrix(1, D, 1)))
  VarY <- abs(matrix(1, 1, D)%*%(var(Y)%*%matrix(1, D, 1)))
  VR <- if(VarX < VarY){VarX/VarY}else{VarY/VarX}
  #VR1 <- (VR - 0.5)/0.5
  return(VR)
}

# Auxiliary function for objective function for classification (CV error for SVM model)
error_func <- function (X, Y, C, K, g, d, r){
  df <- data.frame(X=X,Y=Y)
  set.seed(1)
  if(K=="linear"){
    CV_fit <- tune(svm, as.factor(Y) ~ ., data = df, kernel = K, cost = C)
  }else if(K=="radial"){
    CV_fit <- tune(svm, as.factor(Y) ~ ., data = df, kernel = K, cost = C, gamma = g)
  }else if(K=="sigmoid"){
    CV_fit <- tune(svm, as.factor(Y) ~ ., data = df, kernel = K, cost = C, gamma = g, coef0 = r)
  }else{
    CV_fit <- tune(svm, as.factor(Y) ~ ., data = df, kernel = K, cost = C, gamma = g, degree = d, coef0 = r)
    }
  return(list(error=CV_fit$best.performance, model=CV_fit$best.model))
}

# Auxiliary function for generating missing values (Simple, Overall) (Simple, UD)
Missing_func <- function(df, r, P, M, a){
  set.seed(1)
  DS <- df # data set having missing values 
  DS_row <- round(r*nrow(DS)*ncol(DS)/100) # number of missing values
  DS_row1 <- if(DS_row > nrow(DS)){nrow(DS)}else{DS_row}
  
  p <- 1
  if(P == "simple" & M == "overall"){
    val_list <- matrix(NA, DS_row1, 3)
    val_row <- sample(1:nrow(DS), DS_row1)
    for(i in 1:DS_row1){
      val_col <- sample(1:ncol(DS), 1)
      val_list[i, 1] <- val_row[i] + a
      val_list[i, 2] <- val_col
      val_list[i, 3] <- DS[val_row[i],val_col]
      DS[val_row[i],val_col] <- NA # data set with missing values
    }
    }else if(P == "simple" & M == "UD"){
      val_list <- matrix(NA, DS_row1, 3)
      val_row <- sample(1:nrow(DS), DS_row1)
      for(i in 1:DS_row1){
        val_col <- i-ncol(DS)*floor(i/(ncol(DS)+0.0001))
        val_list[i, 1] <- val_row[i] + a
        val_list[i, 2] <- val_col
        val_list[i, 3] <- DS[val_row[i],val_col]
        DS[val_row[i],val_col] <- NA
      }
    }else if(P == "medium" & M == "overall"){
      val_list <- matrix(NA, 0, 3)
      val_row <- sample(1:nrow(DS), DS_row1)
      j <- 1
      while(DS_row>1){
        if(DS_row>floor(ncol(DS)/2)){m <- sample(2:floor(ncol(DS)/2),1)}else{m <- DS_row}
        DS_col <- sample(1:ncol(DS),m)
        val_list <- rbind(val_list, matrix(NA, m, 3))
        for(i in DS_col){
          val_list[p, 1] <- val_row[j] + a
          val_list[p, 2] <- i
          val_list[p, 3] <- DS[val_row[j],i]
          DS[val_row[j],i] <- NA
          p <- p + 1
        }
        DS_row <- DS_row - m
        j <- j+1
      }
    }else if(P == "medium" & M == "UD"){
      val_list <- matrix(NA, 0, 3)
      val_row <- sample(1:nrow(DS), DS_row1)
      j <- 1
      while(DS_row>1){
        if(DS_row>floor(ncol(DS)/2)){m <- sample(2:floor(ncol(DS)/2),1)}else{m <- DS_row}
        if(length(names(table(val_list[,2])))!=ncol(DS)){DS_col <- sample(1:ncol(DS),m)}else{DS_col <- sample(order(table(val_list[,2]))[1:floor(ncol(DS)/2)],m)}
        val_list <- rbind(val_list, matrix(NA, m, 3))
        for(i in DS_col){
          val_list[p, 1] <- val_row[j] + a
          val_list[p, 2] <- i
          val_list[p, 3] <- DS[val_row[j],i]
          DS[val_row[j],i] <- NA
          p <- p + 1
        }
        DS_row <- DS_row - m
        j <- j+1
      }
    }else if(P == "complex" & M == "overall"){
      val_list <- matrix(NA, 0, 3)
      val_row <- sample(1:nrow(DS), DS_row1)
      j <- 1
      while(DS_row>=floor(ncol(DS)/2)){
        if(DS_row>floor(0.8*ncol(DS))){m <- sample(floor(ncol(DS)/2):floor(0.8*ncol(DS)),1)}else{m <- DS_row}
        DS_col <- sample(1:ncol(DS),m)
        val_list <- rbind(val_list, matrix(NA, m, 3))
        for(i in DS_col){
          val_list[p, 1] <- val_row[j] + a
          val_list[p, 2] <- i
          val_list[p, 3] <- DS[val_row[j],i]
          DS[val_row[j],i] <- NA
          p <- p + 1
        }
        DS_row <- DS_row - m
        j <- j+1
      }
    }else if(P == "complex" & M == "UD"){
      val_list <- matrix(NA, 0, 3)
      val_row <- sample(1:nrow(DS), DS_row1)
      j <- 1
      while(DS_row>=floor(ncol(DS)/2)){
        if(DS_row>floor(0.8*ncol(DS))){m <- sample(floor(ncol(DS)/2):floor(0.8*ncol(DS)),1)}else{m <- DS_row}
        if(length(names(table(val_list[,2])))!=ncol(DS)){DS_col <- sample(1:ncol(DS),m)}else{DS_col <- sample(order(table(val_list[,2]))[1:floor(0.8*ncol(DS))],m)}
        val_list <- rbind(val_list, matrix(NA, m, 3))
        for(i in DS_col){
          val_list[p, 1] <- val_row[j] + a
          val_list[p, 2] <- i
          val_list[p, 3] <- DS[val_row[j],i]
          DS[val_row[j],i] <- NA
          p <- p + 1
        }
        DS_row <- DS_row - m
        j <- j+1
      }
    }
  return(list(df_missed=DS, missed_values=val_list))
}

# Auxiliary function for Mean Imputation
Mean_Imp <- function(data){
  for (i in 1:ncol(data)) {
    data[is.na(data[,i]),i] <- mean(data[,i], na.rm = TRUE)
  }
  return(data)
}

# Auxiliary function to measure imputation performance (RMSE, MAE)
Imp_perform <- function(Imp, val_list){
  RMSE1 <- 0
  MAE1 <- 0
  for (k in 1:nrow(val_list)){
    if(is.numeric(Imp[,val_list[k,2]])){
      RMSE1 <- RMSE1 + (Imp[val_list[k,1], val_list[k,2]]-val_list[k,3])^2
      MAE1 <- MAE1 + abs(Imp[val_list[k,1], val_list[k,2]]-val_list[k,3])
    }else{
      RMSE1 <- RMSE1 + if(Imp[val_list[k,1], val_list[k,2]]==val_list[val_list[k,1], val_list[k,2]]){0}else{1}
      MAE1 <- MAE1 + if(Imp[val_list[k,1], val_list[k,2]]==val_list[val_list[k,1], val_list[k,2]]){0}else{1}
    }
  }
  RMSE1 <- sqrt(RMSE1/nrow(val_list))
  MAE1 <- MAE1/nrow(val_list)
  return(list(RMSE=RMSE1, MAE=MAE1))
}

Z <- zoo[,-c(1,17)] # Experiment on Zoo dataset
Xr <- rbind(zoo[1,-c(1,17)], zoo[50,-c(1,17)], zoo[100,-c(1,17)])
a <- Membership_Distance(Z, Xr, 2)
c <- ASW_func(Z, a$Membership)
## Recording measures
Imp_SVM_Results_te <- data.frame('Data'=rep(c('iris', 'zoo', 'third'),each=90), 'Obj'=rep(rep(c('ASW', 'CCA', 'VR'), each=30),3), 'pattern'=rep(rep(c('simple', 'medium', 'complex'), each=10),9), 'model'=rep(rep(c('overall', 'UD'), each=5), 27), 'ratio'=rep(c(1, 5, 10, 25, 50), 54), 'hypervolume'=rep(NA, 270), 'time'=rep(NA, 270), 'Obj1'=rep(NA, 270), 'Obj2'=rep(NA, 270), 'MAE'=rep(NA, 270), 'RMSE'=rep(NA, 270), 'test_imputed'=rep(NA, 270), 'test'=rep(NA, 270), 'CorObj1MAE'=rep(NA, 270), 'CorObj1RMSE'=rep(NA, 270), 'CorObj2Test'=rep(NA, 270), 'pareto'=rep(NA, 270))

## INITIALIZATION
set.seed(1)
df <- iris
n <- 5
train <- sample(nrow(df), nrow(df)/2) # training and testing
df_tr <- df[train,]
df_te <- df[-train,]
X_tr <- apply(df_tr[,-n], 2, FUN = function(x) (x-min(x))/(max(x)-min(x)))
Y_tr <- df_tr[,n]
X_te <- apply(df_te[,-n], 2, FUN = function(x) (x-min(x))/(max(x)-min(x)))
Y_te <- df_te[,n]

for (xi in 49:90) {
  print(xi)
  obj <- Imp_SVM_Results_te[xi, 'Obj']
  P <- Imp_SVM_Results_te[xi, 'pattern']
  M <- Imp_SVM_Results_te[xi, 'model']
  ro <- Imp_SVM_Results_te[xi, 'ratio']
  
  X_te_mis <- Missing_func(X_te, ro, P, M, nrow(X_tr)) # ro percent missing values for testing data set
  X_tr_mis <- Missing_func(X_tr, ro, P, M, 0) # ro percent missing values for train data set
  
  X_te_mis_MI <- Mean_Imp(X_te_mis$df_missed) # Mean Imputation test data
  X_tr_mis_MI <- Mean_Imp(X_tr_mis$df_missed) # Mean Imputation train data
  
  val_list <- rbind(X_tr_mis$missed_values, X_te_mis$missed_values)
  X <- rbind(X_tr_mis_MI, X_te_mis_MI) # Reconstruct data for imputation part
  
  # Set the parameters:
  N <- nrow(X) # number of samples
  K <- 10    # Maximum number of clusters
  D <- ncol(X)    # number of dimensions
  Cr <- 54      # number of initial population
  Cr_S <- 8   # Number of Chromosums selected for Cross-over
  
  tau.max <- 100 # maximum number of iterations
  #eta <- 5000 # learning rate
  epsilon <- 0.0005 # Threshold on the variation of Objective values (to terminate the process
  
  set.seed(10)
  for (c in 1:(Cr/(K-1))) {
    for (k in 2:K) {
      assign(paste("Xr",(c-1)*(K-1)+(k-1),sep = ""), X[sample(N,k),]) # randomly choose K samples as cluster centers
    }
  }
  m <- sample(seq(1.5,5,0.1),Cr, replace = T)
  Kr <- sample(c("linear", "radial", "sigmoid", "polynomial"), Cr, replace = T)
  C <- sample(seq(0.01,100,0.01),Cr, replace = T)
  #eps <- sample(seq(0.002,0.2,0.002),Cr, replace = T)
  g <- sample(seq(0.005,5,0.005),Cr, replace = T)
  d <- sample(2:5,Cr, replace = T)
  r <- sample(0:20,Cr, replace = T)
  
  Pr_Fr1 <- NULL
  Pr_Fr1_indx <- list()
  error <- data.frame('tau'=rep(1:tau.max,each=Cr), 'center'=rep(1:Cr, tau.max))  # to be used to trace the test and training errors in each iteration
  
  tau <- 1 # iteration counter
  terminate <- FALSE
  
  Z <- X
  #YZ <- Y
  
  start_time <- Sys.time() # Record the starting time of the algorithm
  # Main loop
  while(!terminate){
    for (n in 1:Cr) {
      Xr <- get(paste("Xr",n,sep = ""))
      a <- Membership_Distance(Z, Xr, m[n])
      var <- 0
      for (mm in 1:nrow(val_list)){
        PV = Z[val_list[mm,1],val_list[mm,2]] #Previous value
        NV = sum(a$Membership[val_list[mm,1],]*Xr[,val_list[mm,2]]) #New value 
        var = var + abs(NV - PV)/PV
        Z[val_list[mm,1],val_list[mm,2]] <- NV
      }
      Imp_error <- Imp_perform(Z, val_list)
      Classification <- error_func(Z[1:nrow(X_tr),], Y_tr, C[n], Kr[n], g[n], d[n], r[n])
      preds <- predict(Classification$model, data.frame(X=Z[(nrow(X_tr)+1):nrow(Z),],Y=Y_te))
      preds1 <- predict(Classification$model, data.frame(X=X_te,Y=Y_te))
      
      if(obj == "ASW"){
        error[error$tau==tau & error$center==n, 'Obj1'] <- ASW_func(Z, a$Membership)
      }else if(obj == "CCA"){
        error[error$tau==tau & error$center==n, 'Obj1'] <- CCA_func(Z[1:nrow(X_tr),], Z[(nrow(X_tr)+1):nrow(Z),])
      }else{error[error$tau==tau & error$center==n, 'Obj1'] <- VR_func(Z[1:nrow(X_tr),], Z[(nrow(X_tr)+1):nrow(Z),])}
      error[error$tau==tau & error$center==n, 'Obj2'] <- Classification$error
      error[error$tau==tau & error$center==n, 'Var'] <- var/nrow(val_list)
      error[error$tau==tau & error$center==n, 'RMSE'] <- Imp_error$RMSE
      error[error$tau==tau & error$center==n, 'MAE'] <- Imp_error$MAE
      error[error$tau==tau & error$center==n, 'Imputed_Test'] <- mean(preds!=Y_te)
      error[error$tau==tau & error$center==n, 'Test'] <- mean(preds1!=Y_te)
    }
    # Finding Pareto fronts
    ND=0
    Pareto <- list()
    indx <- 1:Cr
    Pr_Fr <- 0
    while (ND < Cr) {
      indx_Pr <- NULL
      indx_NPr <- NULL
      Pr <- 1
      NPr <- 1
      for (cr in indx) {
        Dom=0
        for (mcr in setdiff(indx, cr)) {
          if(error[error$tau==tau & error$center==cr,'Obj1'] < error[error$tau==tau & error$center==mcr,'Obj1'] & error[error$tau==tau & error$center==cr,'Obj2'] > error[error$tau==tau & error$center==mcr,'Obj2']){Dom = Dom + 1}}
        if(Dom > 0){
          indx_NPr[NPr] <- cr
          NPr <- NPr + 1
        }else{
          ND = ND + 1
          indx_Pr[Pr] <- cr
          Pr <- Pr + 1
        }
      }
      indx <- indx_NPr
      Pr_Fr <- Pr_Fr + 1
      Pareto[[Pr_Fr]] <- indx_Pr
    }
    
    # Selection
    Ncr <- (Cr - length(Pareto[[1]]))/2
    RcrN <- 0
    Rcr <- NULL
    RcrC <- 0
    while (RcrN < Ncr) {
      Rcr <- union(Rcr, Pareto[[(Pr_Fr - RcrC)]])
      RcrN <- RcrN + length(Pareto[[(Pr_Fr - RcrC)]])
      RcrC <- RcrC + 1
    }
    
    # Crossover
    Spring <- NULL
    while (RcrN >= 3) {
      P_Parent <- if(length(setdiff(1:Cr, Spring)) > Cr_S){sort(sample(setdiff(1:Cr, Spring), Cr_S))}else{setdiff(1:Cr, Spring)}
      RSp <- sample(setdiff(Rcr, Spring), 3)
      Max1 <- max(error[error$tau==tau & error$center %in% P_Parent,'Obj1'])
      Min1 <- min(error[error$tau==tau & error$center %in% P_Parent,'Obj1'])
      Min2 <- min(error[error$tau==tau & error$center %in% P_Parent,'Obj2'])
      Max2 <- max(error[error$tau==tau & error$center %in% P_Parent,'Obj2'])
      P1 <- sample(P_Parent[which(error[error$tau==tau & error$center %in% P_Parent,'Obj1']==Max1)],1)
      P2 <- sample(P_Parent[which(error[error$tau==tau & error$center %in% P_Parent,'Obj2']==Min2)],1)
      P3 <- sample(P_Parent[which(error[error$tau==tau & error$center %in% P_Parent,'Obj1']==Min1)],1)
      P4 <- sample(P_Parent[which(error[error$tau==tau & error$center %in% P_Parent,'Obj2']==Max2)],1)
      XrP1 <- get(paste("Xr", P1, sep = ""))
      XrP2 <- get(paste("Xr", P2, sep = ""))
      XrP3 <- get(paste("Xr", P3, sep = ""))
      XrP4 <- get(paste("Xr", P4, sep = ""))
      
      # First Offspring from P1 and P2
      RV <- round(runif(8))
      Xr_1 <- if(RV[1]==0){XrP1}else{XrP2}
      RV_L <- min(length(XrP1), length(XrP2))
      Cross <- matrix(round(runif(RV_L)), nco=D)
      for (ii in 1:nrow(Cross)) {
        for (jj in 1:ncol(Cross)) {
          if(Cross[ii,jj]==1){Xr_1[ii,jj] <- XrP2[ii,jj]}else{Xr_1[ii,jj] <- XrP1[ii,jj]}
        }
      }
      assign(paste("Xr", RSp[1], sep = ""), Xr_1)
      m[RSp[1]] <- if(RV[2]==0){m[P1]}else{m[P2]}
      C[RSp[1]] <- if(RV[3]==0){C[P1]}else{C[P2]}
      Kr[RSp[1]] <- if(RV[5]==0){Kr[P1]}else{Kr[P2]}
      g[RSp[1]] <- if(RV[6]==0){g[P1]}else{g[P2]}
      d[RSp[1]] <- if(RV[7]==0){d[P1]}else{d[P2]}
      r[RSp[1]] <- if(RV[8]==0){r[P1]}else{r[P2]}
      
      # Second Offspring from P1 and P4
      RV <- round(runif(8))
      Xr_1 <- if(RV[1]==0){XrP1}else{XrP4}
      RV_L <- min(length(XrP1), length(XrP4))
      Cross <- matrix(round(runif(RV_L)), nco=D)
      for (ii in 1:nrow(Cross)) {
        for (jj in 1:ncol(Cross)) {
          if(Cross[ii,jj]==1){Xr_1[ii,jj] <- XrP4[ii,jj]}else{Xr_1[ii,jj] <- XrP1[ii,jj]}
        }
      }
      assign(paste("Xr", RSp[2], sep = ""), Xr_1)
      m[RSp[2]] <- if(RV[2]==0){m[P1]}else{m[P4]}
      C[RSp[2]] <- if(RV[3]==0){C[P1]}else{C[P4]}
      Kr[RSp[2]] <- if(RV[5]==0){Kr[P1]}else{Kr[P4]}
      g[RSp[2]] <- if(RV[6]==0){g[P1]}else{g[P4]}
      d[RSp[2]] <- if(RV[7]==0){d[P1]}else{d[P4]}
      r[RSp[2]] <- if(RV[8]==0){r[P1]}else{r[P4]}
      
      # Third Offspring from P3 and P2
      RV <- round(runif(8))
      Xr_1 <- if(RV[1]==0){XrP3}else{XrP2}
      RV_L <- min(length(XrP3), length(XrP2))
      Cross <- matrix(round(runif(RV_L)), nco=D)
      for (ii in 1:nrow(Cross)) {
        for (jj in 1:ncol(Cross)) {
          if(Cross[ii,jj]==1){Xr_1[ii,jj] <- XrP2[ii,jj]}else{Xr_1[ii,jj] <- XrP3[ii,jj]}
        }
      }
      assign(paste("Xr", RSp[3], sep = ""), Xr_1)
      m[RSp[3]] <- if(RV[2]==0){m[P3]}else{m[P2]}
      C[RSp[3]] <- if(RV[3]==0){C[P3]}else{C[P2]}
      Kr[RSp[3]] <- if(RV[5]==0){Kr[P3]}else{Kr[P2]}
      g[RSp[3]] <- if(RV[6]==0){g[P3]}else{g[P2]}
      d[RSp[3]] <- if(RV[7]==0){d[P3]}else{d[P2]}
      r[RSp[3]] <- if(RV[8]==0){r[P3]}else{r[P2]}
      
      Spring <- union(Spring, RSp)
      RcrN <- RcrN - 3
    }
    
    # Mutation
    for (h in setdiff(setdiff(1:Cr, Pareto[[1]]),Spring)) {
      Xr_1 <- get(paste("Xr", h, sep = ""))
      Cr_L <- length(Xr_1) + 7
      Gene_N <- sample(1:Cr_L, 1)
      Gene_S <- sort(sample(1:Cr_L, Gene_N))
      for (hi in 1:Gene_N) {
        if(Gene_S[hi] <= length(Xr_1)){
          hx <- (Gene_S[hi] %/% D)
          hy <- (Gene_S[hi] %% D)
          if(hy != 0){hx <- hx + 1}else{hy <- D}
          Xr_1[hx,hy] <- runif(1)
        }else if(Gene_S[hi] == length(Xr_1) + 1){m[h] <- sample(seq(1.5,5,0.1), 1)}else if(Gene_S[hi] == length(Xr_1) + 2){C[h] <- sample(seq(0.01,100,0.01), 1)}else if(Gene_S[hi] == length(Xr_1) + 3){Kr[h] <- sample(c("linear", "radial", "sigmoid", "polynomial"), 1)}else if(Gene_S[hi] == length(Xr_1) + 4){g[h] <- sample(seq(0.005,5,0.005), 1)}else if(Gene_S[hi] == length(Xr_1) + 5){d[h] <- sample(2:5, 1)}else{r[h] <- sample(0:20, 1)}
      }
      assign(paste("Xr", h, sep = ""), Xr_1)
    }
    
    # Record number of non-dominated solutions in each population
    Pr_Fr1[tau] <- length(Pareto[[1]])
    Pr_Fr1_indx[[tau]] <- Pareto[[1]]
    
    # check termination criteria:
    if(tau > 1){terminate <- tau >= tau.max | abs(sum(error[error$tau==tau, 'Obj1']) - sum(error[error$tau==(tau-1), 'Obj1'])) <= epsilon | abs(sum(error[error$tau==tau, 'Obj2']) - sum(error[error$tau==(tau-1), 'Obj2'])) <= epsilon | Pr_Fr1[tau] == Cr}
    
    # update the counter:
    tau <- tau + 1
  }
  end_time <- Sys.time() # Record the ending time of the algorithm
  Imp_SVM_Results_te[xi,'time'] <- end_time - start_time # Record the running time of the algorithm in minutes
  
  #write.csv(error[error$tau < tau,], file.path(Path_Imp_SVM, "error_iris_25_simple_UD_CCA.csv"))
  #write.list(Pr_Fr1_indx, file.path(Path_Imp_SVM, "Pareto_iris_25_simple_UD_CCA.csv"))
  
  ## Correlation Analysis
  Imp_SVM_Results_te[xi,'CorObj1MAE'] <- cor(error[error$tau < tau, "Obj1"], error[error$tau < tau, "MAE"])
  #cor.test(error[error$tau < tau, "Obj1"], error[error$tau < tau, "MAE"], conf.level = 0.95)
  
  Imp_SVM_Results_te[xi,'CorObj1RMSE'] <- cor(error[error$tau < tau, "Obj1"], error[error$tau < tau, "RMSE"])
  #cor.test(error[error$tau < tau, "Obj1"], error[error$tau < tau, "RMSE"], conf.level = 0.95)
  
  Imp_SVM_Results_te[xi,'CorObj2Test'] <- cor(error[error$tau < tau, "Obj2"], error[error$tau < tau, "Test"])
  #cor.test(error[error$tau < tau, "Obj2"], error[error$tau < tau, "Test"], conf.level = 0.95)
  
  ## ND solutions averages
  Imp_SVM_Results_te[xi,'Obj1'] <- mean(error[error$tau == (tau-1) & error$center %in% Pr_Fr1_indx[[(tau-1)]], "Obj1"])
  Imp_SVM_Results_te[xi,'Obj2'] <- mean(error[error$tau == (tau-1) & error$center %in% Pr_Fr1_indx[[(tau-1)]], "Obj2"])
  Imp_SVM_Results_te[xi,'RMSE'] <- mean(error[error$tau == (tau-1) & error$center %in% Pr_Fr1_indx[[(tau-1)]], "RMSE"])
  Imp_SVM_Results_te[xi,'MAE'] <- mean(error[error$tau == (tau-1) & error$center %in% Pr_Fr1_indx[[(tau-1)]], "MAE"])
  Imp_SVM_Results_te[xi,'test_imputed'] <- mean(error[error$tau == (tau-1) & error$center %in% Pr_Fr1_indx[[(tau-1)]], "Imputed_Test"])
  Imp_SVM_Results_te[xi,'test'] <- mean(error[error$tau == (tau-1) & error$center %in% Pr_Fr1_indx[[(tau-1)]], "Test"])
  
  Rhv <- range(error[error$tau==(tau-1),][Pr_Fr1_indx[[(tau-1)]],4])
  if(Rhv[2] - Rhv[1] > 0){
    hv = hypervolume(data=error[error$tau==(tau-1),][Pr_Fr1_indx[[(tau-1)]],c(3:4)], method = "box")
    #plot(hv)
    Imp_SVM_Results_te[xi,'hypervolume'] <- get_volume(hv)
  }else{Imp_SVM_Results_te[xi,'hypervolume'] <- Rhv[2]}
  
  Imp_SVM_Results_te[xi,'pareto'] <- Pr_Fr1[tau-1]
}

  
for (i in 1:length(Pareto)) {
  print(ggplot(data=error[error$tau==tau,], aes(x=Obj1, y=Obj2)) + geom_point(col="blue") +
          geom_point(data=error[error$tau==tau,][Pareto[[i]],], colour=i) + ggtitle(paste('Population pool (tau=', tau,')')) + theme_minimal())
}
print(ggplot(data=error[error$tau==(tau-1),], aes(x=Obj1, y=Obj2)) + geom_point(col="blue") +
        geom_point(data=error[error$tau==(tau-1),][Pr_Fr1_indx[[(tau-1)]],], colour="red") + ggtitle(paste('Population pool (tau=', tau-1,')')) + xlim(-0.15,1) + ylim(0,1) + theme_minimal())

# Normalised
print(ggplot(data=as.data.frame(apply(error[error$tau==(tau-1),], 2, FUN = function(x) (x-min(x))/(max(x)-min(x)))), aes(x=Obj1, y=Obj2)) + geom_point(col="blue") +
                 geom_point(data=as.data.frame(apply(error[error$tau==(tau-1),],2,FUN = function(x) (x-min(x))/(max(x)-min(x))))[Pr_Fr1_indx[[(tau-1)]],], colour="red") + ggtitle(paste('Population pool (tau=', tau-1,')')) + xlim(-0.15,1) + ylim(0,1) + theme_minimal())

  Xc <- matrix(0,nrow(get(paste("Xr",cr,sep = ""))),D)
  # for each center:
  for (j in 1:nrow(get(paste("Xr",cr,sep = "")))){
    # update the coefficient:
    Zq <- Z[get(paste("Cl",m,sep = ""))[,j]==1,][sample(nrow(Z[get(paste("Cl",m,sep = ""))[,j]==1,]),1),]
    Xc[j,] <- if(colSums(get(paste("Cl",m,sep = "")))[j]>0){get(paste("Xr",m,sep = ""))[j,] + eta/((1+colSums(get(paste("Cl",m,sep = "")))[j])^0.75) * (Zq-get(paste("Xr",m,sep = ""))[j,]) / sum(abs(Zq-get(paste("Xr",m,sep = ""))[j,]))}else{get(paste("Xr",m,sep = ""))[j,]}
    if(any(is.na(Xc[j,]))){
      print(paste("m=",m,sep = ""))
      print(colSums(get(paste("Cl",m,sep = "")))[j])
      print(get(paste("Dis",m,sep = ""))[get(paste("Cl",m,sep = ""))[,j]==1,][1,j])
      Xc[j,] <- get(paste("Xr",m,sep = ""))[j,]
    }
  }
  assign(paste("Xr",m,sep = ""), Xc)
  
## summary of output for each data, objective function, missing type
# Time
Imp_SVM_Results_te[Imp_SVM_Results_te$Data=='third','time'] <- Imp_SVM_Results_te[Imp_SVM_Results_te$Data=='third','time']*60
Imp_SVM_Results_te[Imp_SVM_Results_te$Data=='iris' & Imp_SVM_Results_te$time <= 20,'time'] <- Imp_SVM_Results_te[Imp_SVM_Results_te$Data=='iris' & Imp_SVM_Results_te$time <= 20,'time']*60
Imp_SVM_Results_te[Imp_SVM_Results_te$Data=='zoo' & Imp_SVM_Results_te$time <= 10,'time'] <- Imp_SVM_Results_te[Imp_SVM_Results_te$Data=='zoo' & Imp_SVM_Results_te$time <= 10,'time']*60

mean(Imp_SVM_Results_te[Imp_SVM_Results_te$Data=='zoo','time'], na.rm = T)
(sum(Imp_SVM_Results_te[Imp_SVM_Results_te$Data=='iris' & Imp_SVM_Results_te$time <= 20,'time']*60) + sum(Imp_SVM_Results_te[Imp_SVM_Results_te$Data=='iris' & Imp_SVM_Results_te$time > 20,'time']))/length(Imp_SVM_Results_te[Imp_SVM_Results_te$Data=='iris','time'])
kruskal.test(time ~ Data, data = Imp_SVM_Results_te)
kruskal.test(time ~ Data, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

mean(Imp_SVM_Results_te[Imp_SVM_Results_te$Obj=='VR','time'], na.rm = T)
kruskal.test(time ~ Obj, data = Imp_SVM_Results_te)
kruskal.test(time ~ Obj, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

mean(Imp_SVM_Results_te[Imp_SVM_Results_te$ratio==1,'time'], na.rm = T)
kruskal.test(time ~ as.factor(ratio), data = Imp_SVM_Results_te)
kruskal.test(time ~ ratio, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

mean(Imp_SVM_Results_te[Imp_SVM_Results_te$pattern=='simple','time'], na.rm = T)
kruskal.test(time ~ pattern, data = Imp_SVM_Results_te)
kruskal.test(time ~ pattern, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

mean(Imp_SVM_Results_te[Imp_SVM_Results_te$model=='overall','time'], na.rm = T)
kruskal.test(time ~ model, data = Imp_SVM_Results_te)
kruskal.test(time ~ model, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

# Pareto
mean(Imp_SVM_Results_te[Imp_SVM_Results_te$Data=='third','pareto'], na.rm = T)
kruskal.test(pareto ~ Data, data = Imp_SVM_Results_te)
kruskal.test(pareto ~ Data, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

mean(Imp_SVM_Results_te[Imp_SVM_Results_te$Obj=='VR','pareto'], na.rm = T)
kruskal.test(pareto ~ Obj, data = Imp_SVM_Results_te)
kruskal.test(pareto ~ Obj, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

mean(Imp_SVM_Results_te[Imp_SVM_Results_te$ratio==10,'pareto'], na.rm = T)
kruskal.test(pareto ~ ratio, data = Imp_SVM_Results_te)
kruskal.test(pareto ~ ratio, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

mean(Imp_SVM_Results_te[Imp_SVM_Results_te$pattern=='complex','pareto'], na.rm = T)
kruskal.test(pareto ~ pattern, data = Imp_SVM_Results_te)
kruskal.test(pareto ~ pattern, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

mean(Imp_SVM_Results_te[Imp_SVM_Results_te$model=='UD','pareto'], na.rm = T)
kruskal.test(pareto ~ model, data = Imp_SVM_Results_te)
kruskal.test(pareto ~ model, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

  # Obj1
  mean(Imp_SVM_Results_te[Imp_SVM_Results_te$Data=='third','Obj1'], na.rm = T)
  mean(Imp_SVM_Results_te[Imp_SVM_Results_te$Obj=='VR','Obj1'], na.rm = T)
  mean(Imp_SVM_Results_te[Imp_SVM_Results_te$ratio==10,'Obj1'], na.rm = T)
  mean(Imp_SVM_Results_te[Imp_SVM_Results_te$pattern=='complex','Obj1'], na.rm = T)
  mean(Imp_SVM_Results_te[Imp_SVM_Results_te$model=='UD','Obj1'], na.rm = T)
  
  # Obj2
mean(Imp_SVM_Results_te[Imp_SVM_Results_te$Data=='third','Obj2'], na.rm = T)
kruskal.test(Obj2 ~ Data, data = Imp_SVM_Results_te)
kruskal.test(Obj2 ~ Data, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

mean(Imp_SVM_Results_te[Imp_SVM_Results_te$Obj=='VR','Obj2'], na.rm = T)
kruskal.test(Obj2 ~ Obj, data = Imp_SVM_Results_te)
kruskal.test(Obj2 ~ Obj, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

mean(Imp_SVM_Results_te[Imp_SVM_Results_te$ratio==1,'Obj2'], na.rm = T)
kruskal.test(Obj2 ~ ratio, data = Imp_SVM_Results_te)
kruskal.test(Obj2 ~ ratio, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

mean(Imp_SVM_Results_te[Imp_SVM_Results_te$pattern=='complex','Obj2'], na.rm = T)
kruskal.test(Obj2 ~ pattern, data = Imp_SVM_Results_te)
kruskal.test(Obj2 ~ pattern, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

mean(Imp_SVM_Results_te[Imp_SVM_Results_te$model=='UD','Obj2'], na.rm = T)
kruskal.test(Obj2 ~ model, data = Imp_SVM_Results_te)
kruskal.test(Obj2 ~ model, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

# Imputed_test difference with test
Imp_SVM_Results_te[,'test_diff'] <- Imp_SVM_Results_te[,'test_imputed'] - Imp_SVM_Results_te[, 'test']
ks.test(Imp_SVM_Results_te$test_diff, rnorm(250, mean(Imp_SVM_Results_te$test_diff), sd(Imp_SVM_Results_te$test_diff)))

mean(Imp_SVM_Results_te[, 'test_imputed'])
mean(Imp_SVM_Results_te[, 'test'])
wilcox.test(Imp_SVM_Results_te[,'test_imputed'], Imp_SVM_Results_te[, 'test'], paired = TRUE, alternative = "two.sided", conf.level = 0.95)
wilcox.test(rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[,'test_imputed'], rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[, 'test'], paired = TRUE, alternative = "two.sided", conf.level = 0.95)

mean((rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$Obj=='ASW', 'test_imputed']-rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$Obj=='ASW', 'test']))
mean((rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$Obj=='CCA', 'test_imputed']-rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$Obj=='CCA', 'test']))
mean((rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$Obj=='VR', 'test_imputed']-rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$Obj=='VR', 'test']))

kruskal.test((test_imputed - test) ~ Data, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))
kruskal.test((test_imputed - test) ~ Obj, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))
kruskal.test((test_imputed - test) ~ ratio, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))
kruskal.test((test_imputed - test) ~ pattern, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))
kruskal.test((test_imputed - test) ~ model, data = rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))

mean((rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$ratio==1, 'test_imputed']-rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$ratio==1, 'test']))
mean((rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$ratio==5, 'test_imputed']-rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$ratio==5, 'test']))
mean((rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$ratio==10, 'test_imputed']-rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$ratio==10, 'test']))
mean((rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$ratio==25, 'test_imputed']-rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$ratio==25, 'test']))
mean((rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$ratio==50, 'test_imputed']-rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$ratio==50, 'test']))

mean((rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$pattern=='simple', 'test_imputed']-rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$pattern=='simple', 'test']))
mean((rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$pattern=='medium', 'test_imputed']-rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$pattern=='medium', 'test']))
mean((rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$pattern=='complex', 'test_imputed']-rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr)[Imp_SVM_Results_te$pattern=='complex', 'test']))

# ASW
Imp_SVM_Results_all <- data.frame(rbind(Imp_SVM_Results_te_only, Imp_SVM_Results_te_tr))
median(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW','CorObj1MAE'], na.rm = T)
median(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW','CorObj1RMSE'], na.rm = T)

mean(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW','MAE'], na.rm = T)
mean(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW','RMSE'], na.rm = T)

MutInf(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW' & !is.na(Imp_SVM_Results_all$MAE),'MAE'], Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW' & !is.na(Imp_SVM_Results_all$MAE),'Obj1'])/min(Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW' & !is.na(Imp_SVM_Results_all$MAE),'Obj1']), Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW' & !is.na(Imp_SVM_Results_all$MAE),'MAE']))
MutInf(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW' & !is.na(Imp_SVM_Results_all$MAE),'RMSE'], Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW' & !is.na(Imp_SVM_Results_all$MAE),'Obj1'])/min(Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW' & !is.na(Imp_SVM_Results_all$MAE),'Obj1']), Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW' & !is.na(Imp_SVM_Results_all$MAE),'RMSE']))

# CCA
median(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA','CorObj1MAE'], na.rm = T)
median(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA','CorObj1RMSE'], na.rm = T)

mean(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA','MAE'], na.rm = T)
mean(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA','RMSE'], na.rm = T)

MutInf(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA' & !is.na(Imp_SVM_Results_all$MAE),'MAE'], Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA' & !is.na(Imp_SVM_Results_all$MAE),'Obj1'])/min(Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA' & !is.na(Imp_SVM_Results_all$MAE),'Obj1']), Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA' & !is.na(Imp_SVM_Results_all$MAE),'MAE']))
MutInf(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA' & !is.na(Imp_SVM_Results_all$MAE),'RMSE'], Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA' & !is.na(Imp_SVM_Results_all$MAE),'Obj1'])/min(Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA' & !is.na(Imp_SVM_Results_all$MAE),'Obj1']), Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA' & !is.na(Imp_SVM_Results_all$MAE),'RMSE']))

# VR
median(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR','CorObj1MAE'], na.rm = T)
median(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR','CorObj1RMSE'], na.rm = T)

mean(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR','MAE'], na.rm = T)
mean(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR','RMSE'], na.rm = T)

MutInf(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR' & !is.na(Imp_SVM_Results_all$MAE),'MAE'], Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR' & !is.na(Imp_SVM_Results_all$MAE),'Obj1'])/min(Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR' & !is.na(Imp_SVM_Results_all$MAE),'Obj1']), Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR' & !is.na(Imp_SVM_Results_all$MAE),'MAE']))
MutInf(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR' & !is.na(Imp_SVM_Results_all$MAE),'RMSE'], Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR' & !is.na(Imp_SVM_Results_all$MAE),'Obj1'])/min(Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR' & !is.na(Imp_SVM_Results_all$MAE),'Obj1']), Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR' & !is.na(Imp_SVM_Results_all$MAE),'RMSE']))

# CV Obj2
median(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW','CorObj2Test'], na.rm = T)
median(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA','CorObj2Test'], na.rm = T)
median(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR','CorObj2Test'], na.rm = T)
median(Imp_SVM_Results_all[,'CorObj2Test'], na.rm = T)

mean(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW','test'], na.rm = T)
mean(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA','test'], na.rm = T)
mean(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR','test'], na.rm = T)

MutInf(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW' & !is.na(Imp_SVM_Results_all$MAE),'test'], Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW' & !is.na(Imp_SVM_Results_all$MAE),'Obj2'])/min(Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW' & !is.na(Imp_SVM_Results_all$MAE),'Obj2']), Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='ASW' & !is.na(Imp_SVM_Results_all$MAE),'test']))
MutInf(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA' & !is.na(Imp_SVM_Results_all$MAE),'test'], Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA' & !is.na(Imp_SVM_Results_all$MAE),'Obj2'])/min(Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA' & !is.na(Imp_SVM_Results_all$MAE),'Obj2']), Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='CCA' & !is.na(Imp_SVM_Results_all$MAE),'test']))
MutInf(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR' & !is.na(Imp_SVM_Results_all$MAE),'test'], Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR' & !is.na(Imp_SVM_Results_all$MAE),'Obj2'])/min(Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR' & !is.na(Imp_SVM_Results_all$MAE),'Obj2']), Entropy(Imp_SVM_Results_all[Imp_SVM_Results_all$Obj=='VR' & !is.na(Imp_SVM_Results_all$MAE),'test']))
MutInf(Imp_SVM_Results_all[,'test'], Imp_SVM_Results_all[,'Obj2'])/min(Entropy(Imp_SVM_Results_all[,'Obj2']), Entropy(Imp_SVM_Results_all[,'test']))


kruskal.test(MAE ~ Obj, data = Imp_SVM_Results_all)
kruskal.test(RMSE ~ Obj, data = Imp_SVM_Results_all)

## comparison
mean(Imp_SVM_Results_all[Imp_SVM_Results_all$Data=='iris' & Imp_SVM_Results_all$Obj=='VR', 'test'])
mean(Imp_SVM_Results_all[Imp_SVM_Results_all$Data=='iris' & Imp_SVM_Results_all$Obj=='VR', 'test_imputed'])

mean(Imp_SVM_Results_all[Imp_SVM_Results_all$Data=='zoo' & Imp_SVM_Results_all$Obj=='VR' & Imp_SVM_Results_all$ratio==1 & Imp_SVM_Results_all$pattern=='simple', 'test'])
mean(Imp_SVM_Results_all[Imp_SVM_Results_all$Data=='iris' & Imp_SVM_Results_all$Obj=='ASW' & Imp_SVM_Results_all$ratio==1 & Imp_SVM_Results_all$pattern=='simple', 'test_imputed'])
