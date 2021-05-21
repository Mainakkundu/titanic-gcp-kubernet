from flask import Flask,request, url_for, redirect, render_template, jsonify
#from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np
import config

app = Flask(__name__)

#model = pickle.load(open('regression_model_output_v0.0.1.pkl', 'rb'))
filename = 'finalized_model1.sav'
loaded_model = pickle.load(open(filename, 'rb'))

cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'cabin', 'embarked','title']


@app.route('/health/',methods=['GET'])
def heath():
    return ('API working!!!')

@app.route('/predict/',methods=['POST'])
def predict_api_updated():
    event=request.json
    data_unseen = pd.DataFrame([event['instances']])
    data_unseen.columns=cols
    prediction = str(loaded_model.predict(data_unseen)[0])
    return(jsonify({'predict':prediction}))

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=config.PORT, debug=config.DEBUG_MODE)
