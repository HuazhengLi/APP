# POC Api built based on https://flask-restful.readthedocs.io/en/latest/quickstart.html#argument-parsing
# use "python api.py" to let the api run
# To get the feedback from the trained model, use curl, example include:
# curl http://127.0.0.1:5000/news -d "news=chase is going down" -X POST -v
# the result will be 0.28xxx


from flask import Flask
from flask_restful import reqparse, Api, Resource
import pandas as pd
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('news')

class News(Resource):
    def post(self):
        args = parser.parse_args()
        news = args['news']
        print(news)
        X = pd.Series(news)
        pred = model.predict_proba(X)[0][1]
        return pred, 201


api.add_resource(News, '/news')

if __name__ == "__main__":
    app.run(debug=False, host='127.0.0.1', port=5000)
