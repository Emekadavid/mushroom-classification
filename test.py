# use requests library to do the POST method
import requests

mushroom = {'bruises': "no",
 'odor': "creosote",
 'gill-spacing': "crowded",
 'gill-size': "narrow",
 'gill-color': "gray",
 'stalk-root': "bulbous",
 'stalk-surface-above-ring': "smooth",
 'stalk-surface-below-ring': "smooth",
 'stalk-color-above-ring': "white",
 'stalk-color-below-ring': "white",
 'ring-type': "pendant",
 'spore-print-color':"brown",
 'population': "several",
 'habitat': "woods"}

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
