import pickle
import numpy as np
from flask import Flask, request, jsonify

with open('./model/model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

def predict_single(mushroom, dv, model):
    X_test = dv.transform(mushroom)
    prediction = model.predict(X_test)
    return prediction[0]


app = Flask("mushroom")

@app.route("/mushroom_classification", methods=['POST'])
def predict():
    mushroom = request.get_json()

    prediction = predict_single(mushroom, dv, model)
    result = float(prediction)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)    