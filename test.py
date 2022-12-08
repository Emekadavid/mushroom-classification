# use requests library to do the POST method
import requests

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

## the route we made for prediction
url = 'http://localhost:9696/mushroom_classification'
## post the mushroom information in json format 
response = requests.post(url, json=mushroom)
## get the server response 
result = response.json()
if result == 1.0:
    print("Edible")
elif result == 0.0:
    print("Poisonous") 
