from flask import Flask, jsonify, request, render_template
import json
import pickle
import pandas as pd 

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        X = pd.Series(request.form['news'])
        pred = model.predict(X)
        if pred == 1:
            result = "Recommended!"
        else:
            result = 'NOT Recommended!'
    return render_template('index.html', result = result)


if __name__ == "__main__":
    app.run(debug = False, host = '127.0.0.1', port = 5000)
    
