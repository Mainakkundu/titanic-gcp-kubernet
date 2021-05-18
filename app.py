from flask import Flask,request, url_for, redirect, render_template, jsonify
#from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('regression_model_output_v0.0.1')
cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']






@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    data_unseen = data[cols]
    #prediction = predict_model(model, data=data_unseen)
    prediction = model.predict(data_unseen)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)
