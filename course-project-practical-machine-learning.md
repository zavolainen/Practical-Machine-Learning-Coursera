Summary
-------

This report is part of the Coursera Practical Machine Learning course.
My goal is to develope a model based on the Human Activity Recognition
Weight Lifting data\* to predict the type of exercise. I ended up using
the Random Forest model.

Extracting the data
===================

    # download data file if it does not exist
    trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

    trainFile <- "data/pml-training.csv"
    testFile <- "data/pml-testing.csv"

    if (!file.exists(trainFile)) {
            download.file(trainUrl, trainFile, mode = "wb")
    }
    if (!file.exists(testFile)) {
            download.file(testUrl, testFile, mode = "wb")
    }

    trainingData <- read.csv("data/pml-training.csv")
    testingData <- read.csv("data/pml-testing.csv")

    # load the needed libraries
    suppressMessages(library(caret))
    suppressMessages(library(rattle))
    library(rpart)

    # set seed to be able to reproduce
    set.seed(100)

Exploring the data
------------------

    dim(trainingData)

    ## [1] 19622   160

    dim(testingData)

    ## [1]  20 160

    str(trainingData)

    ## 'data.frame':    19622 obs. of  160 variables:
    ##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
    ##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
    ##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
    ##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
    ##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
    ##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
    ##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
    ##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
    ##  $ kurtosis_roll_belt      : Factor w/ 397 levels "","-0.016850",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ kurtosis_picth_belt     : Factor w/ 317 levels "","-0.021887",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ kurtosis_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_roll_belt      : Factor w/ 395 levels "","-0.003095",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_roll_belt.1    : Factor w/ 338 levels "","-0.005928",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_yaw_belt      : Factor w/ 4 levels "","#DIV/0!","0.00",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
    ##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
    ##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
    ##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
    ##  $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
    ##  $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
    ##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
    ##  $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
    ##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
    ##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
    ##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
    ##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
    ##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
    ##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
    ##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
    ##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
    ##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
    ##  $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
    ##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
    ##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
    ##  $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
    ##  $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
    ##  $ kurtosis_roll_arm       : Factor w/ 330 levels "","-0.02438",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ kurtosis_picth_arm      : Factor w/ 328 levels "","-0.00484",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ kurtosis_yaw_arm        : Factor w/ 395 levels "","-0.01548",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_roll_arm       : Factor w/ 331 levels "","-0.00051",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_pitch_arm      : Factor w/ 328 levels "","-0.00184",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_yaw_arm        : Factor w/ 395 levels "","-0.00311",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
    ##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
    ##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
    ##  $ kurtosis_roll_dumbbell  : Factor w/ 398 levels "","-0.0035","-0.0073",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ kurtosis_picth_dumbbell : Factor w/ 401 levels "","-0.0163","-0.0233",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ kurtosis_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_roll_dumbbell  : Factor w/ 401 levels "","-0.0082","-0.0096",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_pitch_dumbbell : Factor w/ 402 levels "","-0.0053","-0.0084",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_yaw_dumbbell        : Factor w/ 73 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_yaw_dumbbell        : Factor w/ 73 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
    ##   [list output truncated]

There are 19622 observations and 160 columns on the data set. Data have
a lot of NA's which is not valuable information, therefore we need to
clean the data. Also we can get rid of the user-related information like
user names and time stamps.

    rowsWithNA <- which(colSums(is.na(trainingData) |trainingData=="")>0.9*dim(trainingData)[1]) 

    cleanedTrainingData <- trainingData[,-rowsWithNA] # get rid of columns that have 90% or more NA's
    cleanedTrainingData <- cleanedTrainingData[,-c(1:7)] # get rid of user-related info

    dim(cleanedTrainingData)

    ## [1] 19622    53

    # data cleaning for the test set also

    rowsWithNAtest <- which(colSums(is.na(testingData) |testingData=="")>0.9*dim(testingData)[1]) 
    cleanedTestData <- testingData[,-rowsWithNAtest]

    dim(cleanedTestData)

    ## [1] 20 60

Training data slicing
=====================

We need to split the training data to be able to examine out-of-sample
errors.

    inTrain <- createDataPartition(cleanedTrainingData$classe,
                                   p=0.75, list=FALSE)
    useTrain <- cleanedTrainingData[inTrain, ]
    useTest <- cleanedTrainingData[-inTrain, ]

Predicting model
================

    trControl <- trainControl(method="cv", number=5)

Random forest
-------------

    fitRF <- train(classe ~ ., data = useTrain, method = "rf", trControl=trControl, verbose=FALSE)
    fitRF

    ## Random Forest 
    ## 
    ## 14718 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 11774, 11774, 11774, 11774, 11776 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9913031  0.9889986
    ##   27    0.9908272  0.9883966
    ##   52    0.9838965  0.9796285
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 2.

    predictRF <- predict(fitRF, useTest) # predict using test set sliced from training data
    cMatrixRF <- confusionMatrix(useTest$classe, predictRF) #use confusion matrix to see prediction result
    cMatrixRF

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1394    1    0    0    0
    ##          B    3  940    6    0    0
    ##          C    0    5  848    2    0
    ##          D    0    0   15  789    0
    ##          E    0    0    0    2  899
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9931          
    ##                  95% CI : (0.9903, 0.9952)
    ##     No Information Rate : 0.2849          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9912          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9979   0.9937   0.9758   0.9950   1.0000
    ## Specificity            0.9997   0.9977   0.9983   0.9964   0.9995
    ## Pos Pred Value         0.9993   0.9905   0.9918   0.9813   0.9978
    ## Neg Pred Value         0.9991   0.9985   0.9948   0.9990   1.0000
    ## Prevalence             0.2849   0.1929   0.1772   0.1617   0.1833
    ## Detection Rate         0.2843   0.1917   0.1729   0.1609   0.1833
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
    ## Balanced Accuracy      0.9988   0.9957   0.9870   0.9957   0.9998

As we can see from confusion matrix, the prediction accuracy of Random
Forest is very high, 0.9931. Therefore the out-of-sample error rate is
0.0069.

Just a note for reprodusing that Random Forest model is heavy. I got
stuck for a while because of the computation. Also, I tried to visualize
the Random Forest with reprtree package but unfortunately the package
installation does not work.

Classification tree
-------------------

    fitDT <- rpart(classe ~ ., data=useTrain, method="class")
    fancyRpartPlot(fitDT)

    ## Warning: labs do not fit even at cex 0.15, there may be some overplotting

![](course-project-practical-machine-learning_files/figure-markdown_strict/unnamed-chunk-7-1.png)

    predictDT <- predict(fitDT, useTest, type = "class")
    cMatrixDT <- confusionMatrix(predictDT, useTest$classe)
    cMatrixDT

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1286  182   19   79   33
    ##          B   42  552   79   32   70
    ##          C   30  110  681  123  120
    ##          D   16   68   62  526   60
    ##          E   21   37   14   44  618
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7469          
    ##                  95% CI : (0.7345, 0.7591)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6784          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9219   0.5817   0.7965   0.6542   0.6859
    ## Specificity            0.9108   0.9436   0.9054   0.9498   0.9710
    ## Pos Pred Value         0.8043   0.7123   0.6400   0.7186   0.8420
    ## Neg Pred Value         0.9670   0.9039   0.9547   0.9334   0.9321
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2622   0.1126   0.1389   0.1073   0.1260
    ## Detection Prevalence   0.3261   0.1580   0.2170   0.1493   0.1497
    ## Balanced Accuracy      0.9163   0.7626   0.8509   0.8020   0.8285

As we can see from confusion matrix, the prediction accuracy in
Classification Tree is 0.7494, out-of-sample error 0.2506. Accuracy is
lower than in Random Forest.

Model selection
===============

It seems that the Random Forest produces better model to predict the
type of exercise than Classification tree. Accuracy of Random Forest is
very high, 0.9931. To predict the type of exercise in the Human Activity
Recognition Weight Lifting data.

Prediction
==========

Now we predict the cleaned original test data set with the selected
model.

    testingDataPrediction <- predict(fitRF, newdata=cleanedTestData)
    testingDataPrediction

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

-   *Data source: Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.;
    Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data
    Classification of Body Postures and Movements. Proceedings of 21st
    Brazilian Symposium on Artificial Intelligence. Advances in
    Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer
    Science. , pp. 52-61. Curitiba, PR: Springer Berlin /
    Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI:
    10.1007/978-3-642-34459-6\_6. Cited by 2 (Google Scholar)*
