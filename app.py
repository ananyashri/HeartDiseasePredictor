#pip install flask
from flask import Flask, render_template
from functions import*

app = Flask(__name__)

@app.route('/')
def index():
    prediction_message = call_model(50,1,0,144,200,0,0,126,1,0.9,1,0,3)
    return render_template('index.html', prediction=prediction_message)
