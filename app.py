import pandas as pd
from flask import Flask, render_template, request, url_for, flash, redirect
import joblib
import os
app = Flask(__name__)

# Load models and symptom vocab once on startup
model = joblib.load('./models/model.pkl')
label_encoder = joblib.load('./models/label_encoder.pkl')
symptom_list = joblib.load('./models/symptom_vocab.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        symptoms = [request.form.get(f'symptom{i}') for i in range(1, 18)]
        symptoms = [s.lower().strip() for s in symptoms if s and s.strip() != '']

        input_vector = [0] * len(symptom_list)
        for symptom in symptoms:
            if symptom in symptom_list:
                idx = symptom_list.index(symptom)
                input_vector[idx] = 1

        pred_label = model.predict([input_vector])[0]
        prediction = label_encoder.inverse_transform([pred_label])[0]

    return render_template('index.html', symptom_list=symptom_list, prediction=prediction)


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    prediction = None
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        symptoms = [s.lower().strip() for s in symptoms if s and s.strip() != '']

        input_vector = [0] * len(symptom_list)
        for symptom in symptoms:
            if symptom in symptom_list:
                idx = symptom_list.index(symptom)
                input_vector[idx] = 1

        pred_label = model.predict([input_vector])[0]
        disease = label_encoder.inverse_transform([pred_label])[0]

        # You might want to return confidence or other metadata. For example:
        confidence = 90  # just dummy, replace with your model's output if available
        
        prediction = {
            'disease': disease,
            'confidence': confidence,
            'symptoms': symptoms
        }

    return render_template('predict.html', symptoms_list=symptom_list, prediction=prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
