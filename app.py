#pip install flask
from flask import Flask, render_template, send_from_directory, request, session
from functions import*

app = Flask(__name__)

@app.route('/')
def index():
    #prediction_message = call_model(50,1,0,144,200,0,0,126,1,0.9,1,0,3)
    #return render_template('index.html', prediction=prediction_message)
    return render_template('index.html')


@app.route('/enterData')
def predict(): 
    return render_template('enterData.html')

@app.route('/submitted', methods=['GET', 'POST'])
def submitted():
    a = request.form.get('Age')
    b = request.form.get('sex')
    c = request.form.get('chest_pain')
    d = request.form.get('resting_blood_pressure')
    e = request.form.get('cholesterol')
    f = request.form.get('fbs')
    g = request.form.get('restecg')
    h = request.form.get('thalatch')
    i = request.form.get('exang')
    j = request.form.get('oldpeak')
    k = request.form.get('slope')
    l = request.form.get('ca')
    m = request.form.get('thal')

    prediction_message = "" 
    try:
        values = list(map(float, list(a, b, c, d, e, f, g, h, i, j, k, l, m)))
        prediction_message = call_model(values[0], values[1], values[2], values[3], values[4],
                                        values[5], values[6], values[7], values[8], values[9],
                                        values[10], values[11], values[12])
    except:
        prediction_message = "You have typed a non-integer or a non-decimal value. Please re-enter the appropriate values"

    return render_template('enterData.html', prediction=prediction_message)