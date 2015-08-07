# Machine learning with random forest
Coursera student 75055  
Thursday, August 06, 2015  

.

.

## Introduction

This paper documents a model that predicts if a weightlifting exercise, the Unilateral Dumbbell Biceps Curl, has been done correctly or not; if the latter the model predicts what was wrong. A correct exercise scores an A.There are five mutually exclusive incorrect ways to do the exercises (scoring B, C, D, or E). The outcome is predicted on the basis of measurements of movement sensors on a dumbbell, the waste, the arm and the forearm [1].

.

## Set up


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.1.3
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
# use parallel processing to speed things up
library(doParallel)
```

```
## Warning: package 'doParallel' was built under R version 3.1.3
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
# use 2 threads for each active core
registerDoParallel(cores=2)
```

.

## Data

The used train and test data are available on the internet [2] 

#### --- Download the train data from the internet --------------

```r
if (!file.exists("datTrain.csv")) {
    url  = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    dest = "datTrain.csv"
    meth = "internal"
    quit = TRUE
    mode = "wb"
    download.file(url, dest, meth, quit, mode)
    # NOTE this works under windows 7, modify if nessesairy
} 
train0 <- read.csv("datTrain.csv",na.strings=c("NA",""))
```

#### --- Download the test data from the internet --------------

```r
if (!file.exists("datTest.csv")) {
    url  = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    dest = "datTest.csv"
    meth = "internal"
    quit = TRUE
    mode = "wb"
    download.file(url, dest, meth, quit, mode)
    # NOTE this works under windows 7, modify if nessesairy
} 
test0 <- read.csv("datTest.csv",na.strings=c("NA",""))
```

#### --- data exploration ---

The train data consists of 19622 observation of 160 variables. 
Only 406 of the 19622 observations have valid values for all the variables. 
The other 19216 observation all have have invalid values for exactly the same 100 variables. 

The distribution of the A,B,C,D,E scores of the six subjects are:

```r
table(train0$classe,train0$user_name)
```

```
##    
##     adelmo carlitos charles eurico jeremy pedro
##   A   1165      834     899    865   1177   640
##   B    776      690     745    592    489   505
##   C    750      493     539    489    652   499
##   D    515      486     642    582    522   469
##   E    686      609     711    542    562   497
```

#### --- data selection ---

The 100 variables for which the data are mostly missing were omitted. 
From the remaining 60 variables, the first seven are purely administrative (are not movement measurements or the outcome registration), these where omitted also. 


```r
# remove mostly NA columns
NA_in_col_cnt <- colSums(is.na(train0))
keep_col      <- NA_in_col_cnt < 500
train1        <- train0[,keep_col]
# remove first seven admin columns
train2 <- train1[,-c(1:7)]
dim(train2)
```

```
## [1] 19622    53
```

```r
sum(is.na(train2))
```

```
## [1] 0
```

So the data that are useful consists of 19622 observations of 53 variables. There are no NA's in this data, all but the outcome registration (A,B,C,D,E) are numeric or integer. 

#### --- build data sets to use ---

Here we will only use 10% of the data to train the model. This is an unusual low percentage, normally we would use 60% or more. However 60% requires 40 minutes to train the model on my computer; 10% takes only 4 minutes. (See the discussion section for more information on the effects of the percentage.)   
The remaining 90% of the train2 data will be set aside to validate the model (i.e. produce an out of sample error prediction)


```r
set.seed(75055)
idxValidate <- createDataPartition( y=train2$classe , p=0.9, list=FALSE ) 
# NOTE use 0.9 to reduce cpu time
valSet <- train2[ idxValidate,]
dim(valSet)
```

```
## [1] 17662    53
```

```r
trnSet <- train2[-idxValidate,]
dim(trnSet)
```

```
## [1] 1960   53
```

The last of the 53 variables is the outcome variable classe (with the values A,B,C,D and E). The outcome data were written to a vector named y -the y in y=f(x1,x2...,x53)-. 


```r
y <- trnSet[,53]
# str(y)
```

The remaining data were written to a data-frame named predict -the data which can be used to predict the outcome y-. All the predict data are numeric or integer.


```r
predict <- trnSet[,-53]
# str(predict)
```

.

## Model building

The trnSet can be used to build a model.


```r
if (!file.exists("model.rds")) {
    # NOTE it takes up to 60 minutes to train the model
    #      so save it (so you have to do it only once)
    model <- train(classe ~ ., method="rf", data=trnSet)
    saveRDS(model, file="model.rds")
    # takes  4 min at p=0.9 (use 10% as traindata), accuracy 17 out of 20
    # takes 10 min at p=0.8 (use 20% as traindata), accuracy 19 out of 20
    # takes 20 min at p=0.6 (use 40% as traindata), accuracy 20 out of 20
    # takes 35 min at p=0.4 (use 60% as traindata), accuracy 20 out of 20
} else {
    # use the already existing model (do not train it again)
    model <- readRDS("model.rds")
}
```

To test the model, we predict what the outcomes according to model are and compare the predictions with the real values.



```r
pred <- predict(model,predict)
confusionMatrix(pred,y)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 558   0   0   0   0
##          B   0 379   0   0   0
##          C   0   0 342   0   0
##          D   0   0   0 321   0
##          E   0   0   0   0 360
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9981, 1)
##     No Information Rate : 0.2847     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2847   0.1934   0.1745   0.1638   0.1837
## Detection Rate         0.2847   0.1934   0.1745   0.1638   0.1837
## Detection Prevalence   0.2847   0.1934   0.1745   0.1638   0.1837
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

As the confusion matrix shows, the model  perfectly predicts the outcomes of the data on which it was trained. Lets see if the model also can predict cases that it has not seen before (that were not used in training the model). For that we use the data in the validation set.

.

## Validation 

First build the y and predict for the validation data.


```r
yVal       <- valSet[, 53]
predictVal <- valSet[,-53]
```

Now use our model to predict on the out of sample data, and construct a confusion matrix to see how good the prediction is.


```r
predVal <- predict(model, predictVal)
confusionMatrix(predVal, yVal)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4933  232    3   29    3
##          B   33 2893  102   13   17
##          C   29  238 2874   87   51
##          D   24   47   89 2764   81
##          E    3    8   12    2 3095
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9375          
##                  95% CI : (0.9339, 0.9411)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.921           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9823   0.8464   0.9331   0.9547   0.9532
## Specificity            0.9789   0.9884   0.9722   0.9837   0.9983
## Pos Pred Value         0.9487   0.9460   0.8765   0.9198   0.9920
## Neg Pred Value         0.9929   0.9641   0.9857   0.9911   0.9895
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2793   0.1638   0.1627   0.1565   0.1752
## Detection Prevalence   0.2944   0.1731   0.1857   0.1701   0.1767
## Balanced Accuracy      0.9806   0.9174   0.9527   0.9692   0.9757
```

The model performance on the validation set is not perfect (accuracy about 94%).   
The confusion matrix shows where the mistakes are made.

Note that we can read the confusion matrix as:  
Of the 5201 A predictions, 4933 (95%) were correct, 233 (4.5%) were really B and 29 (0.5%) were D.   
Of the 3056 B predictions, 33 (1%) were A, 2892 (95%) were correct, 102 (3%) were C, etc.   
This means that an incorrect prediction A is probably a B and an incorrect B is probably a C.   
(we will use these most probable alternatives later in the interpretation of our test results)

So with this model we can hope to get about 94% of predictions on unseen data correct.   
In terms of our test data about 19 or 18 of the 20 cases should be correct.

.

## predicting the results of the test data

The model can now be used to predict the outcomes of our test data.

The first step is to remove the unwanted columns from the test set. To do that we must use the keep_col vector we constructed earlier. 

The next step is to construct the y and the predict data for the test data.


```r
# remove unwanted colums
test1  <- test0[,keep_col]
tstSet <- test1[,-c(1:7)]
# construct yTst and predictTst
yTst    <- tstSet[, 53]  
predTst <- tstSet[,-53]
```

Then use our model to predict the outcomes of our test data,


```r
# produce the tstVals
tstVal <- predict(model, predTst)
# show the result
tstVal
```

```
##  [1] B A A A A E D D A A C C B A E E A B B B
## Levels: A B C D E
```

Coursera reports that our model got 17 out of 20 correct (slightly less than the 18 or 19 we expected).

Using the confusion matrix of our validation data, we can say that an incorrect A probably is a B, an incorrect C is probably a B and an incorrect D probably a C. If we use these probable alternatives, Coursera reports that 19 out of 20 are correct.

. . .

## Discussion 

Random forest models work amazingly well . . . but performance is an issue, these models take a long time to train.

Possible solutions to the performance problem are:

- use library(doParallel) + registerDoParallel (cores=2),   
cores=3 or 4 would be better if your computer can handle it,   
installing more working memory will make a larger number of cores possible.   
- do not make the train data set larger than needed (small is beautiful).   
- save the trained model and reuse it (do not train the model over and over again)   
- Buy a new computer with higher speed and 32 or 64 Gigs of RAM.   
- Hire for a few dollars time on an Amazon cloud computer (with lots of cores and even more RAM)   

For pure pragmatic reasons we kept the training set small, only 10% of the data was used to train the model.   

Actually I also did use larger training sets. The results were:   
p=0.9 (use 10% as traindata), takes 04 minutes, test set accuracy 17 out of 20   
p=0.8 (use 20% as traindata), takes 10 minutes, test set accuracy 19 out of 20   
p=0.6 (use 40% as traindata), takes 20 minutes, test set accuracy 20 out of 20   
p=0.4 (use 60% as traindata), takes 35 minutes, test set accuracy 20 out of 20   
The time it took is specific for my computer, newer computers might work faster.  

. 

The confusion matrix of the validation set can be used to estimate probabilities (a prediction A is 95% of the time really an A, is 4.5% a B, is 0% a C etc.). Random forest produce more information than just the prediction. In our case, where we had a second chance, we could use this information to boost our results from 17 out of 20 correct to 19 correct. 

.

## References

[1] Ugulino, W.; Cardador, D.; Vega, K.; Velloso,E.; Milidiu,R.; Fuks,H. Wearable Computing: Accelerometers' Data
Classification of Body Postures and Movements [(www->)](http://groupware.les.inf.puc-rio.br/har)  
[2] train data [(www->)](https://d396qusza40orc.cloudfront.net/predmachlearn/pml?training.csv); test data [(www->)](https://d396qusza40orc.cloudfront.net/predmachlearn/pml?testing.csv)  
[3] An Introduction to Statistical Learning with Applications in R, Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani [(www->)](http://www-bcf.usc.edu/~gareth/ISL/data.html)  
[4] Model Training and Tuning [(www->)](http://topepo.github.io/caret/training.html#custom)  
[5] Coursera, Practical Machine Learning [(www->)](https://www.coursera.org/course/predmachlearn)


