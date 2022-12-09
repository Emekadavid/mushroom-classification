import streamlit as st
import pickle 
import numpy as np 

tabs_font_css = """
<style>
button[data-baseweb="tab"] {
  font-size: 26px;
}
</style>
""" 
st.write(tabs_font_css, unsafe_allow_html=True)
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


# create the side bar. 
st.header("Mushroom classification Project")
st.image("./image/mushrooms_pix.jpg")
st.write("Mushrooms have become popular once again. But people need to be aware that \
    there are dangers in mushroom eating. Some mushrooms can be poisonous when ingested. \
    these project builds a classification model that would help mushroom lovers to \
    find out if a given mushroom species is edible or poisonous. The classification \
    based on the features of the mushroom species.")
st.markdown("The **Predict** tab below hosts the model where you can make predictions \
    on a given species while the **EDA** tab gives you a host of exploratory data analysis \
        that was done on the dataset used to build the model.")


about, predict, eda = st.tabs(["About the model", "Predict", "EDA"])

with about:
    st.write("The features for the test.py file that was used to run the model \
        on Github was based on the following features:")
    st.write(mushroom)
    st.write("The keys in the above dictionary are the feature names while \
        the values are taken from a given species. On the Predict tab, these values \
        are given as the default so you can just run the model with default values \
        and it will give you the same results as what was obtained from the files on \
        the Github repo.")
    st.write("The chosen model to use was XGBClassifier because it gave the best F1 score. \
        For this dataset, other boosting models like Adaboost also give an F1 score of 1.0.")        


with predict:
    # provide 4 columns with checkboxes each for all the 14 features
    # detailing each of the options as stated on the kaggle page
    st.write("Each feature is given here with their options as stated on the Kaggle page. \
        Choose the option for a feature a given species has that you would like to classify.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bruises = st.radio("**Bruises Feature**", ["bruises","no"], index=1)
        gillcolor = st.radio("**Gill Color Feature**", ["black","brown","buff","chocolate","gray"], index=3)
        stalk_surface_below_ring = st.radio("**Stalk Surface Below Ring Feature**", ["fibrous","scaly","silky","smooth"], index=0)
        spore_print_color = st.radio("**Spore Print Color Feature**", ["black","brown","buff","chocolate","green","orange","purple","white","yellow"], index=0)
        
    
    with col2:
        odor = st.radio("**Odor Feature**", ["almond","anise","creosote","fishy","foul","musty","none","pungent","spicy"], index=6 )
        stalk_color_above_ring = st.radio("**Stalk Color Above Ring Features**", ["brown","buff","cinnamon","gray","orange","pink","red","white","yellow"], index=7)
        habitat = st.radio("**Habitat Feature**", ["grasses","leaves","meadows","paths","urban","waste","woods"], index=0)        

    with col3:
        gillspacing = st.radio("**Gill Spacing Features**", ["close","crowded","distant"], index=1)    
        stalkroot = st.radio("**Stalk Root Feature**", ["bulbous","club","cup","equal","rhizomorphs","rooted","missing"], index=3)
        stalk_color_below_ring = st.radio("**Stalk Color Below Ring Feature**", ["brown","buff","cinnamon","gray","orange","pink","red","white","yellow"], index=7)
        
    
    with col4:
        gillsize = st.radio("**Gill Size Feature**", ["broad","narrow"], index=0)
        stalk_surface_above_ring = st.radio("**Stalk Surface Above Ring Feature**", ["fibrous","scaly","silky","smooth"], index=0)
        ring_type = st.radio("**Ring Type Feature**", ["cobwebby","evanescent","flaring","large","none","pendant","sheathing","zone"], index=1)
        population = st.radio("**Population Feature**", ["abundant","clustered","numerous","scattered","several","solitary"], index=0)
    # we need to pick what was selected
    # first check if the Check button was clicked
    st.markdown("Click the **Check Chosen Features** button to confirm was what chosen.")
    if st.button("**Check Chosen Features**"):
        mushroom["bruises"] = bruises
        mushroom['odor'] = odor
        mushroom['gill-spacing'] = gillspacing
        mushroom['gill-size'] = gillsize
        mushroom["gill-color"] = gillcolor
        mushroom["stalk-root"] = stalkroot
        mushroom["stalk-surface-above-ring"] = stalk_surface_above_ring
        mushroom["stalk-surface-below-ring"] = stalk_surface_below_ring
        mushroom["stalk-color-above-ring"] = stalk_color_above_ring
        mushroom["stalk-color-below-ring"] = stalk_color_below_ring
        mushroom["ring-type"] = ring_type
        mushroom["spore-print-color"] = spore_print_color
        mushroom["population"] = population
        mushroom["habitat"] = habitat
        st.write(mushroom)
    st.write("After confirmation that the data you have is what you wanted \
        to classify, you can now carry out the prediction.")
    st.write("Click the **Predict Mushroom Species** button below.")
    if st.button("**Predict Mushroom Species**"):
        X_test = dv.transform(mushroom)
        prediction = model.predict(X_test)
        result = prediction[0]
        if result == 1.0:
            st.markdown("The mushroom species is **Edible**")
        elif result == 0.0:
            st.markdown("The mushroom species is **Poisonous**")
        st.balloons()        
                

    

with eda:
    st.write("The eda tab")        
