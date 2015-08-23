# Practical Machine learning
Firdhoush K  
Sunday, August 23, 2015  

##Background


Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement  a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

##Data 


The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

##Data Loading


```r
require(dplyr) 
pmlTrain <- read.csv("practical_machine_learning\\pml-training.csv",header=T, na.strings=c("NA", "#DIV/0!"))

pmlTest <- read.csv("practical_machine_learning\\pml-testing.csv",header=T, na.strings=c("NA", "#DIV/0!"))

## NA exclusion for all available variables
noNApmlTrain<-pmlTrain[, apply(pmlTrain, 2, function(x) !any(is.na(x)))] 
dim(noNApmlTrain)
```

```
## [1] 19622    60
```

```r
## variables with user information, time and undefined
cleanpmlTrain<-noNApmlTrain[,-c(1:8)]
dim(cleanpmlTrain)
```

```
## [1] 19622    52
```

```r
## 20 test cases provided clean info - Validation data set
cleanpmltest<-pmlTest[,names(cleanpmlTrain[,-52])]
dim(cleanpmltest)
```

```
## [1] 20 51
```

##Data Partitioning and Prediction 

The cleaned downloaded data set was subset in order to generate a test set independent from the 20 cases provided set. Partitioning was performed to obtain a 60% training set and a 40% test set.


```r
library(caret)
inTrain<-createDataPartition(y=cleanpmlTrain$classe, p=0.75,list=F)
training<-cleanpmlTrain[inTrain,] 
test<-cleanpmlTrain[-inTrain,] 

dim(training)
```

```
## [1] 14718    52
```

```r
dim(test)
```

```
## [1] 4904   52
```

##Results and Conclusions

Random forest trees were generated for the training dataset using cross-validation. Then the generated algorithm was examnined under the partitioned training set to examine the accuracy and estimated error of prediction. By using 51 predictors for five classes using cross-validation at a 5-fold an accuracy of 99.2% with a 95% CI [0.989-0.994] was achieved accompanied by a Kappa value of 0.99.


```r
library(randomForest)
set.seed(13333)
fitControl2<-trainControl(method="cv", number=5, allowParallel=T, verbose=T)
rffit<-train(classe~.,data=training, method="rf", trControl=fitControl2, verbose=F)
```

```
## + Fold1: mtry= 2 
## - Fold1: mtry= 2 
## + Fold1: mtry=26 
## - Fold1: mtry=26 
## + Fold1: mtry=51 
## - Fold1: mtry=51 
## + Fold2: mtry= 2 
## - Fold2: mtry= 2 
## + Fold2: mtry=26 
## - Fold2: mtry=26 
## + Fold2: mtry=51 
## - Fold2: mtry=51 
## + Fold3: mtry= 2 
## - Fold3: mtry= 2 
## + Fold3: mtry=26 
## - Fold3: mtry=26 
## + Fold3: mtry=51 
## - Fold3: mtry=51 
## + Fold4: mtry= 2 
## - Fold4: mtry= 2 
## + Fold4: mtry=26 
## - Fold4: mtry=26 
## + Fold4: mtry=51 
## - Fold4: mtry=51 
## + Fold5: mtry= 2 
## - Fold5: mtry= 2 
## + Fold5: mtry=26 
## - Fold5: mtry=26 
## + Fold5: mtry=51 
## - Fold5: mtry=51 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 26 on full training set
```

```r
predrf<-predict(rffit, newdata=test)
confusionMatrix(predrf, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    4    0    0    0
##          B    0  942    7    0    0
##          C    0    3  845   11    0
##          D    0    0    3  791    0
##          E    0    0    0    2  901
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9939          
##                  95% CI : (0.9913, 0.9959)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9923          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9926   0.9883   0.9838   1.0000
## Specificity            0.9989   0.9982   0.9965   0.9993   0.9995
## Pos Pred Value         0.9971   0.9926   0.9837   0.9962   0.9978
## Neg Pred Value         1.0000   0.9982   0.9975   0.9968   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1921   0.1723   0.1613   0.1837
## Detection Prevalence   0.2853   0.1935   0.1752   0.1619   0.1841
## Balanced Accuracy      0.9994   0.9954   0.9924   0.9915   0.9998
```

```r
pred20<-predict(rffit, newdata=cleanpmltest)
# Output for the prediction of the 20 cases provided
pred20
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
Once, the predictions were obtained for the 20 test cases provided, the below shown script was used to obtain single text files to be uploaded to the courses web site to comply with the submission assigment. 20 out of 20 hits also confirmed the accuracy of the obtained models.


```r
getwd()
```

```
## [1] "C:/Users/Firdhoush/Desktop/R"
```

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(pred20)
```
