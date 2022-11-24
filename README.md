# Mushroom classification

Eating of mushrooms has regained popularity in recent times. Mushroom hunting (called "shrooming") is prominent recently. But the dangers in mushroom eating is that there are thousands of varieties of mushrooms and many of them are poisonous. Eating a poisonous mushroom can cause side-effects such as nausea, vomitting, cramps, and diarrhea. 

This project's aim is to help mushroom lovers quickly analyze a mushroom collected in the garden or the wild to understand whether the species is harmless or poisonous. The data is more than 30 years old and was collected from [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification).

## Exploratory Data Analysis (EDA)

All the predictors are categorical variables. On closer examination of the train and test data set, there were no null values in either of the two. 

On building the distribution of selected predictors, we noticed that there were a lot of young mushroom. Mushrooms with white gill colors which are poisonous were also noticed to have a significant number in the data set. There were also a good distribution of mature mushrooms based on their spore color. 

## Feature selection

Mutual information score was used for feature seleection. This was used because all the predictors were categorical variables. After calculating mutual information score for all the predictors, the number of predictors to use was narrowed down from 23 to 14. 

These 14 predictors had significant relationship to the target variable, class. The threshold was a MI score of 0.06. 