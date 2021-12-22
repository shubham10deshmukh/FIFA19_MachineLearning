
# Q1. Summary

### A
Using a sports dataset (regression) with a lot of text, money, date data-types and required a good amount of analysis which included data processing, data cleaning, data exploration, and feature engineering.

### B
Using a feature selection technique to eliminate irrelevant or less helpful features.

### C
Using an ensemble regressor model (XgBoot Regressor) to predict for both training and testing data and calculate the different scores for validation. Using a single pipeline to include a data scaler, polynomial model, and the regressor.

### D
Using a Kfold Cross-Validation to validate the robustness of the model (calculated score and RMSE).

### E
Used Hyperparameter tuning for the regressor model (RandomizedSearchCV). however, this part ran only once and commented in the code to increase the runtime of the application.

### F
Done dataset analysis using visualization and EDA, some of the figures are in the figs folder, some were done just for understanding the dataset and not inc in the project to avoid confusion.

 
 

# Q2. Dataset Description


### Dataset Description

Contains players data of Fifa 19 video game : [LINK](https://www.kaggle.com/karangadiya/fifa19) <br>
The Dataset contains links, string, float, money, date values and need heavy cleaning and data exploration before being put to calculate VALUE (regression) using the other features.<br>

### Total Samples
18207 Rows<br>

### Total Measurments
89 Columns<br>

### Measurement Description
Contains different attributes & monetary aspects of football players like SHOT, PACE, PHYSICAL,REF, KICK, SM, DRIBBLE, PASS, SPEED , AWR, DWR, VALUE, WAGE, RELEASE CLAUSE, PLAYER FACE etc. (Total 89).<br>

### Group of Interests (Predicting Player Value)
We are predicting the value of players(euro money) in the transfer market. <br>


## AUC Values
Top - 10 AUC Values
| Feature | AUC |
|:--------|:-----:|
| Overall|  0.990|
|Reactions|  0.926|
|Potential | 0.915|
|    Wage | 0.907|
|Release Clause | 0.902|
|    Composure | 0.876|
|         LCM | 0.864|
| BallControl | 0.843|
|    Special | 0.840|
|ShortPassing | 0.838|

The AUC values are calulated – furthest from 0.5.

# Question 3

### A

The application uses a dataset that comprises of stats of soccer players from the video game FIFA 19 and contained highly irregular and textual data. The data required a lot of preprocessing. I removed some of the features initially having links or irrelevant data. The data set contained features with weight(lbs), money($, Million, K), and height data. I wrote a logic respectively for all of the features like that to convert them to float data-type, the data had a lot of null values so some of them were replaced with 0, mean and some with modes accordingly. I also encoded the data in multiple columns to make it more understandable for the model. In the feature engineering part I created a few columns out of the existing one by analyzing their impact on the target, it required me to merge or split columns and also created synthetic data using other related columns (did it for the target as well). I did Exploration data analysis to get better insights into the data and also understand the meaning and relation of several features, some of the example images are in the fig/ folder. All these codes and outputs are in data_processing_cleaning(), clean_label(), and some figures.

### B

The dataset had around 89 columns and out of which some of them were irrelevant so, to get a better understanding of which exact feature to keep for prediction I used a SelectFromModel feature selection method with LASSO as its model. In this feature selection, the lasso model recalibrates the data points by penalizing them and shirking the coefficient to 0 or less. Selectfrommodel does nothing but using the result of the regressor given for selecting the features of X which has a higher coefficient. I have normalized the data using a min-max scaler before feature selection so that the data can be scaled down to a standard range of values so that the model can better understand the data removing any outlier.<br>
I used a correlation matrix to determine which all features are inter-related and can be removed, the correlation matrix figure is also exported in the figs folder.<br>
I used recursive feature elimination during my experiment as well but due to increased running time because of its recursive nature and vast amount of data and an insignificant difference at the end, I didn't use it however, I have kept the rfecv() in the code for a better understanding of the technique

hnique. 

### C

I have used an ensemble decision tree based model (XGboost regressor) it is an extension of boosting algorithm, ensemble model trains and predicts on individual models(base learner) and then combining them. The combining process removes the bad prediction at the end summing up only the good prediction to generate good results. The booster parameter sets the type of the base learner I have used a tree type base learner instead of linear because of the non-linear dataset and the model has greater flexibility with the tree base learners(supports greater non-linearity). The estimators are the number of no of trees the model uses at splits and the learning rate is the amount of time in which the model learns both these params are inversely proportional to each other and finding a right fit is required to get the best accuracy. Overdoing these parameters can overfit the data. I tried so many combinations and also hyperparameter tuning to get the best parameters for my model for the specific data and features I am using. I have calculated different scores for validating my dataset, I calculated the scores both on the training and testing data because of highly irregular data and to know if there are any possible overfitting or underfitting. the Average, R square, RMSE, Standard Error, Standard Deviation, and Explained variance scores are exported to the scores.csv in outputs and also printed at the terminal. My model gave an RMSE and std deviation of around 52725 for train and 464790 for the test which isn't bad considering the average value of players are around 50 Million or more, A relative r square error of around 0.99 is superb considering it should be near to 1. I have also calculated a standard error and exported it to the score file.<br>
During my experiment with the dataset I used multiple other models like and Random Forest, Decision Tree etc but I ended up with XGBoost because of its boosting and regularization behavior, The trees in this are parallelized implemented in the sequential tree building process unlike RF where it randomly selects a sub-component, the XGB also penalizes model to avoid overfitting and it uses max_depth to prune tree instead of mandate backward tree pruning. Since it is an efficient version of gradient boosting, I didn't experimented that model. Based on all the learnings and experiments I felt that XGboost was the best model for my regression dataset for finding the value of the players.
Lastly, I have created a pipeline with a minmixscaler to standardize the data, a polynomial feature, and the xgboost regressor. the pipeline automates workflow, they are easier to understand and fix and also increase readability. The polynomial feature here generates a new feature matrix consisting of all polynomial combinations of the features with degrees less than or equal to the specified degree (2) and it adds feature to my dataset from existing ones and adds an entire range of non-linearity and options to learn from. The predictions of train and test data can be viewed in outputs/.. along with the actual value and name of the players, the score sheet is exported as well along with the split sets (Xtrain and Xtest) in the data folder.

### D

I have used Kfold cross-validation with the same pipeline as the predicting one to calculate the score and rmse of the train and test set, I have used StratifiedKFold to validate the robustness of the model by using 5 splits over the datasets. I got a score of around 0.98 for the test set which is only 0.01 lesser than without SKF, showing the robustness also I got an RMSE of 612758.4. The output of the CV score is printed in the terminal.

### E
I have used hyperparameter tuning to fine my xgboostregressor, I used randomizedsearchcv to randomly find combinations from the given bunch of parameters values, I have run the code once in the code initially and then used the same set in the application to avoid running it over and over again and saving on running time.

### F

I have done multiple data visualization and Exploratory data analysis, some of them are exported in the figs/ folder. I removed outliers in the label and their before and after, graphs are exported as Y_log_values ( used np.log for better understanding of values). I have created multiple columns and I have analyzed their impact and trend with the target and exported them. (columns: nationalities, contract year left joined, club). Some of my analysis involved knowing the dataset as well since its related to sports so many things are interrelated and had a significant impact on the players value like loaned_from, release clause and the overall rating, etc, the tabular analysis I did during my experiment are not included in the code or figure to avoid confusion/unreadability.



