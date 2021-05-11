# Cervical Cancer Detection from Behavior Factors with Machine Learning Algorithm

## Introduction
This project aims on early detection of Ca Cervix based on behavior determinants including behavior, perception, intention, motivation, subjective norm, attitude, social support and empowerment, and to identify important predictors for the disease. 

## Method
The analytical plan for this study includes descriptive statistics in order to understand the distribution of the sample and the measures. Feature selection will be conducted with Principal Component Analysis (PCA), and selected features will be used for prediction. The classification methods include Logistic regression and KNN algorithm in order to evaluate whether the outcome variable of having Cervical Cancer can be predicted by the given set of features, 60% of percentage split will be used to create training and test datasets randomly to build the model and estimate the predictive accuracy.

## Data source
http://archive.ics.uci.edu/ml/datasets/Cervical+Cancer+Behavior+Risk#

## R code
### Import the dataset
library(readr)

cervix <- read_csv("/Users/Desktop/ML/sobar-72.csv")

### PCA feature selection approach
library(factoextra)

pca.cervix <- prcomp(cervix, center = TRUE,scale. = TRUE)

summary(pca.cervix)

(eig.val <- get_eigenvalue(pca.cervix))

print(pca.cervix$rotation)

library(haven)

fviz_eig(pca.cervix, addlabels = TRUE)

library(dplyr)

pca.cervix %>% biplot(cex = .5)

fviz_pca_ind(pca.cervix,mean.point=F, addEllipses = T, legend.title="ca_cervix")

### split train and test data
set.seed(103)

n = dim(cervix)[1]

cervix$ca_cervix = as.factor(cervix$ca_cervix)

train_ind <- sample(n[1], 0.6 * n)

train <- cervix[train_ind, ]

test <- cervix[-train_ind, ]

### Logistic regression with PCA
components <- cbind(ca_cervix = cervix[, "ca_cervix"], pca.cervix$x[, 1:5]) %>% as.data.frame()

glm.fit1 = glm(ca_cervix ~ ., data=components[train_ind,], family = 'binomial')

glm.pred.prob1 = predict(glm.fit1, components[-train_ind,], type = 'response')

glm.pred.class1 = ifelse(glm.pred.prob1 > 0.5, 1, 0)

library("ROCR") 

library(pROC)

pred <- prediction(glm.pred.prob1, test$ca_cervix)    

perf <- performance(pred, measure = "tpr", x.measure = "fpr")

plot(perf, main="AUC-ROC of LR", xlab="Specificity", ylab="Sensitivity")    
abline(0, 1)

auc(test$ca_cervix, glm.pred.prob1)

true = test$ca_cervix==1

#Test Confusion Matrix 

table(glm.pred.class1, true) 

#Test error

mean(glm.pred.class1 != true)

glm.pred.prob1 = predict(glm.fit1, components[train_ind,], type = 'response')

glm.pred.class1 = ifelse(glm.pred.prob1 > 0.5, 1, 0)

true = train$ca_cervix==1

#Traning Confusion Matrix 

table(glm.pred.class1, true) 

#Traning error

mean(glm.pred.class1 != true)

### K-NN with PCA
library(class)

set.seed(100)

train.error1 = rep(0,10)

test.error1 = rep(0,10)

X.pca = data.frame(pca.cervix$x[,1:5])

ca_cervix = cervix$ca_cervix

for(k in 1:10){

model.knn.train <- knn(train=X.pca[train_ind,], test=X.pca[train_ind,], ca_cervix[train_ind], k=k)

train.error1[k] <- sum(model.knn.train!= ca_cervix[train_ind])/length(ca_cervix[train_ind])

model.knn.test <- knn(train=X.pca[train_ind,], test=X.pca[-train_ind,], ca_cervix[train_ind], k=k)

test.error1[k] <- sum(model.knn.test!=ca_cervix[-train_ind])/length(ca_cervix[-train_ind])

}

plot(1:10, train.error1, col='red', type = 'b',ylim=c(0,0.15),xlab="Number of neighbors",ylab="Misclassification errors")

points(1:10, test.error1, col='blue', type = 'b')

legend("topright",legend=c("traning error","test error"), col=c("red","blue"), pch=1)

test.pred.k5 = knn(train=X.pca[train_ind,], test=X.pca[-train_ind,], ca_cervix[train_ind], k=5,prob=T)

plot(roc(ca_cervix[-train_ind], attributes(test.pred.k5)$prob),
     print.thres = T,
     print.auc = T,
     print.auc.y = 0.2, main="AUC-ROC of KNN")
