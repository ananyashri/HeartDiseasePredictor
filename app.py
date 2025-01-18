#pip install flask
from flask import Flask, render_template, send_from_directory, request, session
from functions import*

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/enterData')
def renderEnterData(): 
    return render_template('enterData.html')


@app.route('/medicalTerms')
def renderMedicalTerms():
    return render_template('medicalTerms.html')


@app.route('/submitted', methods=['POST'])
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
        values = list(map(float, [a, b, c, d, e, f, g, h, i, j, k, l, m]))
        prediction_message = call_model(*values)
    except ValueError as e:
        prediction_message = "Please enter valid numeric values for all fields."
    except Exception as e:
        prediction_message = "An unexpected error occurred. Please try again."

    return render_template('prediction.html', prediction=prediction_message)