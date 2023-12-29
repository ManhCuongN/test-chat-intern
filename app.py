from flask import Flask, render_template, request, jsonify
from chat import get_response



app = Flask(__name__)

import requests
from io import StringIO
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response



@app.get("/")
def index_get():
    return "Xin Chao"


@app.post("/predict")
def predict():
    text = request.get_json().get("message")

    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}

    return jsonify(message)

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=80)