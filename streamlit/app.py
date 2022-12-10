import streamlit as st
import pickle 
import numpy as np 
import pandas as pd

st.set_page_config(
    layout="wide", 
    page_title="Mushroom Classification")
mushroom = {'bruises': "no",
 'odor': "none",
 'gill-spacing': "crowded",
 'gill-size': "broad",
 'gill-color': "chocolate",
 'stalk-root': "equal",
 'stalk-surface-above-ring': "fibrous",
 'stalk-surface-below-ring': "fibrous",
 'stalk-color-above-ring': "white",
 'stalk-color-below-ring': "white",
 'ring-type': "evanescent",
 'spore-print-color':"black",
 'population': "abundant",
 'habitat': "grasses"}

with open('./model/model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

mushroom_test = dict()

# create the side bar. 
st.markdown("<h1 style='text-align: center; color: red;'>Mushroom Classification</h1>", unsafe_allow_html=True)
#st.markdown("Mushroom classification.")
cola, colb, colc = st.columns(spec=[0.5,10,0.5])
st.image("./image/mushrooms_pix.jpg", use_column_width=True)
st.write("Mushrooms have become popular once again. But people need to be aware that \
    there are dangers in mushroom eating. Some mushrooms can be poisonous when ingested. \
    these project builds a classification model that would help mushroom lovers to \
    find out if a given mushroom species is edible or poisonous. The classification is\
    based on the features of the mushroom species.")
st.markdown("The **Predict** tab below hosts the model where you can make predictions \
    on a given species while the **Test Data** tab gives you a host of test data \
    you can use to test the model. The model did not see the test data while model fitting.")


about, predict, test_data = st.tabs(["About The Model Tab", "Predict Tab", "Test Data Tab"])

with about:
    st.write("The features for the test.py file that was used to run the model \
        on Github was based on the following features:")
    st.write(mushroom)
    st.write("The keys in the above dictionary are the feature names while \
        the values are taken from a given species. On the Predict tab, you can \
        be able to change these default values based on the mushroom species you have. ")
    st.write("The chosen model to use was XGBClassifier because it gave the best F1 score. \
        For this dataset, other boosting models like Adaboost also give an F1 score of 1.0.")        


with predict:
    # provide 4 columns with checkboxes each for all the 14 features
    # detailing each of the options as stated on the kaggle page
    st.write("Each feature is given here with their options as stated on the Kaggle page. \
        Choose the option for a feature a given species has that you would like to classify.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:        
        bruises = st.radio("**Bruises**", ["bruises","no"])
        gillcolor = st.radio("**Gill Color**", ["black","brown","buff","chocolate","gray", "green","orange","pink","purple","red","white","yellow"]) 
        stalk_color_above_ring = st.radio("**Stalk Color Above Ring**", ["brown","buff","cinnamon","gray","orange","pink","red","white","yellow"])
        population = st.radio("**Population**", ["abundant","clustered","numerous","scattered","several","solitary"])    
    
    with col2:        
        odor = st.radio("**Odor**", ["almond","anise","creosote","fishy","foul","musty","none","pungent","spicy"])
        stalkroot = st.radio("**Stalk Root**", ["bulbous","club","cup","equal","rhizomorphs","rooted","missing"])
        stalk_color_below_ring = st.radio("**Stalk Color Below Ring**", ["brown","buff","cinnamon","gray","orange","pink","red","white","yellow"])
        habitat = st.radio("**Habitat**", ["grasses","leaves","meadows","paths","urban","waste","woods"])        

    with col3:        
        gillspacing = st.radio("**Gill Spacing**", ["close","crowded","distant"])    
        stalk_surface_above_ring = st.radio("**Stalk Surface Above Ring**", ["fibrous","scaly","silky","smooth"])
        ring_type = st.radio("**Ring Type**", ["cobwebby","evanescent","flaring","large","none","pendant","sheathing","zone"])
    
    with col4:       
        gillsize = st.radio("**Gill Size**", ["broad","narrow"])
        stalk_surface_below_ring = st.radio("**Stalk Surface Below Ring**", ["fibrous","scaly","silky","smooth"])
        spore_print_color = st.radio("**Spore Print Color**", ["black","brown","buff","chocolate","green","orange","purple","white","yellow"])
        

    # we need to pick what was selected
    # first check if the Check button was clicked
    mushroom_test["bruises"] = bruises
    mushroom_test['odor'] = odor
    mushroom_test['gill-spacing'] = gillspacing
    mushroom_test['gill-size'] = gillsize
    mushroom_test["gill-color"] = gillcolor
    mushroom_test["stalk-root"] = stalkroot
    mushroom_test["stalk-surface-above-ring"] = stalk_surface_above_ring
    mushroom_test["stalk-surface-below-ring"] = stalk_surface_below_ring
    mushroom_test["stalk-color-above-ring"] = stalk_color_above_ring
    mushroom_test["stalk-color-below-ring"] = stalk_color_below_ring
    mushroom_test["ring-type"] = ring_type
    mushroom_test["spore-print-color"] = spore_print_color
    mushroom_test["population"] = population
    mushroom_test["habitat"] = habitat
    st.markdown("Click the **Check Chosen Features** button to confirm what was chosen.")
    if st.button("**Check Chosen Features**"):
        
        st.write(mushroom_test)
    st.write("After confirmation that the data you have is what you wanted \
        to classify, you can now carry out the prediction.")
    st.write("Click the **Predict Mushroom Species** button below.")
    if st.button("**Predict Mushroom Species**"):
        X_test = dv.transform(mushroom_test)
        prediction = model.predict(X_test)
        result = prediction[0]
        if result == 1.0:
            st.markdown("The mushroom species is **Edible**.")
            result = -1.0
        elif result == 0.0:
            st.markdown("The mushroom species is **Poisonous**.")
            result = -1.0    
        st.balloons()       
                
with test_data:
    st.write("The test data is provided for you to verify the accuracy of the prediction \
        with the target, the class feature, provided for this. So you can compare the prediction \
        from the model with the actual ground truth from the data. Note that this test data \
        was not used in fitting the model.")
    st.write("If the target (class) is a 1, then it is an edible mushroom but if an \
        0 then it is a poisonous mushroom.")    
    test_dataframe = pd.read_csv("./streamlit/test_data.csv")    
    st.write(test_dataframe)           
