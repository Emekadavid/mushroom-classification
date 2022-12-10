import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from lazypredict.Supervised import LazyClassifier
import xgboost as xgb

import pickle

# import the data set. 
mushrooms_df = pd.read_csv('data/mushrooms.csv')

# as it stands, we need to do a lot of mapping to make the letters understandable. 
# The code for the letters in the values are in the Kaggle competition description. 
# The class column is our target. We need to make it that edible species are 1 
# and poisonous species are 0

class_map = {'e': 1, 'p': 0}
mushrooms_df['class'] = mushrooms_df['class'].map(class_map)

# cap-shape feature mapping
capshape_map = {'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat', 'k':'knobbed', 's':'sunken'}
mushrooms_df['cap-shape']= mushrooms_df['cap-shape'].map(capshape_map) 

# cap-surface feature mapping
capsurface_map = {'f': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth'}
mushrooms_df['cap-surface'] = mushrooms_df['cap-surface'].map(capsurface_map)

# cap-color feature mapping
capcolor_map = {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 
    'g':'gray', 'r': 'green', 'p':'pink','u':'purple', 'e':'red', 'w':'white','y':'yellow'}
mushrooms_df['cap-color'] = mushrooms_df['cap-color'].map(capcolor_map)    

# bruises feature mapping
bruises_map = {'t': 'bruises', 'f': 'no'}
mushrooms_df.bruises = mushrooms_df.bruises.map(bruises_map)

# odor feature map
odor_map = {'a':'almond', 'l':'anise', 'c':'creosote', 'y':'fishy', 'f':'foul', 
    'm':'musty', 'n':'none', 'p':'pungent', 's':'spicy'}
mushrooms_df.odor = mushrooms_df.odor.map(odor_map)    

# gill-attachment feature mapping
gillattachment_map = {'a':'attached', 'd':'descending', 'f':'free', 'n':'notched'}
mushrooms_df['gill-attachment'] = mushrooms_df['gill-attachment'].map(gillattachment_map)

# gill-spacing feature mapping
gillspacing_map = {'c':'close', 'w':'crowded', 'd':'distant'}
mushrooms_df['gill-spacing'] = mushrooms_df['gill-spacing'].map(gillspacing_map)

# gill-size feature mapping
gillsize_map = {'b':'broad', 'n':'narrow'}
mushrooms_df['gill-size'] = mushrooms_df['gill-size'].map(gillsize_map)

# gill-color feature mapping
gillcolor_map = {'k':'black', 'n':'brown', 'b':'buff', 'h':'chocolate', 'g':'gray', 
    'r':'green', 'o':'orange', 'p':'pink', 'u':'purple', 'e':'red', 'w':'white', 
    'y':'yellow'}
mushrooms_df['gill-color'] = mushrooms_df['gill-color'].map(gillcolor_map)

# stalk-shape feature mapping
stalkshape_map = {'e':'enlarging', 't':'tapering'}
mushrooms_df['stalk-shape'] = mushrooms_df['stalk-shape'].map(stalkshape_map)

# stalk-root feature mapping
stalkroot_map = {'b':'bulbous', 'c':'club', 'u':'cup', 'e':'equal', 
    'z':'rhizomorphs', 'r':'rooted', '?':'missing'}
mushrooms_df['stalk-root'] = mushrooms_df['stalk-root'].map(stalkroot_map)    

# stalk-surface-above-ring feature mapping
stalksurfaceabovering_map = {'f':'fibrous', 'y': 'scaly', 'k':'silky', 's':'smooth'}
mushrooms_df['stalk-surface-above-ring'] = mushrooms_df['stalk-surface-above-ring'].map(stalksurfaceabovering_map)

# stalk-surface-below-ring feature mapping
stalksurfacebelowring_map = {'f':'fibrous', 'y': 'scaly', 'k': 'silky', 's':'smooth'}
mushrooms_df['stalk-surface-below-ring'] = mushrooms_df['stalk-surface-below-ring'].map(stalksurfacebelowring_map)

# stalk-color-above-ring feature mapping
stalkcolorabovering_map = {'n':'brown', 'b': 'buff', 'c': 'cinnamon', 'g':'gray', 
    'o':'orange', 'p':'pink', 'e': 'red', 'w':'white', 'y':'yellow'}
mushrooms_df['stalk-color-above-ring'] = mushrooms_df['stalk-color-above-ring'].map(stalkcolorabovering_map)    

# stalk-color-below-ring feature mapping
stalkcolorbelowring_map = {'n':'brown', 'b':'buff', 'c':'cinnamon', 'g':'gray', 
    'o':'orange', 'p':'pink', 'e':'red', 'w': 'white', 'y':'yellow'}
mushrooms_df['stalk-color-below-ring'] = mushrooms_df['stalk-color-below-ring'].map(stalkcolorbelowring_map)    

# veil-type feature mapping
veiltype_map = {'p':'partial', 'u':'universal'}
mushrooms_df['veil-type'] = mushrooms_df['veil-type'].map(veiltype_map)

# veil-color feature mapping
veilcolor_map = {'n':'brown', 'o':'orange', 'w':'white', 'y':'yellow'}
mushrooms_df['veil-color'] = mushrooms_df['veil-color'].map(veilcolor_map)

# ring-number feature mapping
ringnumber_map = {'n':'none', 'o':'one', 't':'two'}
mushrooms_df['ring-number'] = mushrooms_df['ring-number'].map(ringnumber_map)

# ring-type feature mapping
ringtype_map = {'c':'cobwebby', 'e':'evanescent', 'f':'flaring', 'l':'large',
    'n':'none', 'p':'pendant', 's':'sheathing', 'z':'zone'}
mushrooms_df['ring-type'] = mushrooms_df['ring-type'].map(ringtype_map)    

# spore-print-color feature mapping
sporeprintcolor_map = {'k':'black', 'n':'brown', 'b':'buff', 'h':'chocolate',
    'r':'green', 'o':'orange', 'u':'purple', 'w':'white', 'y':'yellow'}
mushrooms_df['spore-print-color'] = mushrooms_df['spore-print-color'].map(sporeprintcolor_map)

# population feature mapping
population_map = {'a':'abundant', 'c':'clustered', 'n':'numerous', 's':'scattered', 
    'v':'several', 'y':'solitary'}
mushrooms_df.population = mushrooms_df.population.map(population_map)

# habitat feature mapping
habitat_map = {'g':'grasses', 'l':'leaves', 'm':'meadows', 'p':'paths', 'u':'urban', 
    'w':'waste', 'd':'woods'}
mushrooms_df.habitat = mushrooms_df.habitat.map(habitat_map)    

# test train split 
full_train_df, test_df = train_test_split(mushrooms_df, test_size=0.2, random_state=1)

#Feature selection
# Mutual information score will be used for feature selection. 
# A mutual information score measures the amount of information 
# one can obtain from one random variable given another. 
# Mutual information is always larger than or equal to zero, where the larger 
# the value, the greater the relationship between the two variables. 
# If the calculated result is zero, then the variables are independent. 
# MI scores are usually used for categorical variables. 

# separate the full train data set so that we have 60% train, 20% validation, and 20% test
# we have previouusly separated the test data set
train_df, val_df = train_test_split(full_train_df, test_size=0.25, random_state=1)

# separating the targets from the variables, train dataframe
y_train = train_df['class']
train_df = train_df.drop("class", axis=1)

# separating the targets from the predictors, validation dataframe
y_val = val_df['class']
val_df = val_df.drop("class", axis=1)

# separation, test dataframe
y_test = test_df['class']
test_df = test_df.drop("class", axis=1)

# calculate the mutual information score on the train data frame
def score(series):
    return mutual_info_score(series, y_train)

mi = train_df.apply(score)
mi = mi.sort_values(ascending=False)

# iterating through the mi score to filter the predictors
# store the indexes in a list
keep_cols = []
for index, value in mi.items():
    if value > 0.06:
        keep_cols.append(index)

# now we filter the dataframes based on the columns to use
train_df = train_df[keep_cols]
val_df = val_df[keep_cols]
test_df = test_df[keep_cols]

# Feature Engineering
# For the feature engineering, I will only use one-hot encoding to transform all 
# the categorical features to numerical features. I was initially doubtful 
# about using OHE due to the fear of using a sparse dataset, but experts 
# online assured me that with the number of features I have, 
# the sparsity is insignificant. 
# 
# So now going for OHE. We'll use DictVectorizer for this because it is convenient. 

# dictvectorizer object
dv = DictVectorizer(sparse=False)

# train data frame OHE
train_dict = train_df.to_dict(orient="records")
X_train = dv.fit_transform(train_dict)

# val data frame for OHE
val_dict = val_df.to_dict(orient="records")
X_val = dv.transform(val_dict)

# test datafra,e for OHE
test_dict = test_df.to_dict(orient="records")
X_test = dv.transform(test_dict)


# Model Building

# The next step is to combine the train and validation data frames to get a 
# harmonious overall dataframe. 

# get the target class 
y_train_full = full_train_df['class']

full_train_df = full_train_df[keep_cols]

# we now do one hot encoding
dv_final = DictVectorizer(sparse=False)

# train data frame OHE
train_dict_full = full_train_df.to_dict(orient="records")
X_train_full = dv_final.fit_transform(train_dict_full)

# test dataframe for OHE
test_dict = test_df.to_dict(orient="records")
X_test_full = dv_final.transform(test_dict)

# In the notebook.ipynb file, we used the lazypredict framework to get an idea
# of models that give better performance for this dataset. That was why XGBoost
# was chosen as the final model. 

# instantiating the model and fitting it
model_final = xgb.XGBClassifier()
model_final.fit(X_train_full, y_train_full)

# let's predict on the final test and evaluate it using classification report
prediction_full = model_final.predict(X_test_full)
# check the classification report for f1 score
print(metrics.classification_report(y_test, prediction_full))

# After exhaustive analysis and consultation with experts on why the model is giving a perfect report, I can say that this is a good model for predicting the type of mushroom species, whether poisonous or edible. 

# It's now time to save the model

# The model and encoder is saved as a binary file using pickle
with open('./model/model.bin', 'wb') as f_out:
   pickle.dump((dv_final, model_final), f_out)
f_out.close() 
