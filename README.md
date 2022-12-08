# Mushroom classification

!["Mushrooms in the wild"](./image/mushrooms_pix.jpg)

Eating of mushrooms has regained popularity in recent times. Mushroom hunting (called "shrooming") is prominent recently. But the dangers in mushroom eating is that there are thousands of varieties of mushrooms and many of them are poisonous. Eating a poisonous mushroom can cause side-effects such as nausea, vomitting, cramps, and diarrhea. 

This project's aim is to help mushroom lovers quickly analyze a mushroom collected in the garden or the wild to understand whether the species is harmless or poisonous. The data is more than 30 years old and was collected from [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification).

## Exploratory Data Analysis (EDA)

All the predictors are categorical variables. On closer examination of the train and test data set, there were no null values in either of the two. 

On building the distribution of selected predictors, we noticed that there were a lot of young mushroom. Mushrooms with white gill colors which are poisonous were also noticed to have a significant number in the data set. There were also a good distribution of mature mushrooms based on their spore color. 

## Feature selection

Mutual information score was used for feature seleection. This was used because all the predictors were categorical variables. After calculating mutual information score for all the predictors, the number of predictors to use was narrowed down from 23 to 14. 

These 14 predictors had significant relationship to the target variable, class. The threshold was a MI score of 0.06. 

## Feature Engineering

All the features were categorical data and they were transformed to numerical features using one-hot encoding (OHE). The DictVectorizer module in SKLearn was used for this. 

## Model Building

Using the lazypredict python library, it was discovered that boosting models were top of the list in giving accurate predictions, so I went for XGBoost Classifier. 

## Exporting notebook to script

The notebook used for the sections above, `notebook.ipynb` was exported to a script. The name of the script is `train.py`. To do this, use the command:
```jupyter nbconvert --to python notebook.ipynb```
Then rename the script. 

## How to run the files in this repo

You need to run the files in a virtual environment. I used pipenv because it is very convenient and easy to use. Follow these steps:
1. Clone this repo first of all in your local environment. 
2. Create a virtual environment for it. To use pipenv first install pipenv:
``` pip install pipenv```
3. Then create the environment by navigating to the directory of the cloned repo and then running this command. The requirements.txt file is already in the repo. 
```pipenv shell```
4. Then install all the dependencies needed:
```pipenv install -r requirements.txt```

## Deploying in a web service using flask

I deployed the app locally using flask as the web service. Note that the following commands assume you are using a Linux machine. 

1. To run the server, use this command:
```gunicorn --bind 0.0.0.0:9696 predict:app```

If you are on a windows machine, install waitress first using `pipenv install waitress`. Waitress would replace gunicorn because gunicorn doesn't run on a Windows machine. Then run the server as:
```waitress-serve --listen=0.0.0.0:9696 predict:app```

2. Then open another terminal and pipe into the virtual environment shell:
```pipenv shell```
Then run the test script:
```python3 test.py```

It will print out the result, whether the mushroom species is edible or poisonous. 

You can predict for several mushroom species by editing the test.py file. Just input the figures for your desired species into the `mushroom` variable in the file. That variable is a python dictionary with the features as keys. 

## Deployment on streamlit

