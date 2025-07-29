from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extracted all the features
            features = [
    quarter := int(request.form['quarter']),
    department := int(request.form['department']),
    day := int(request.form['day']),
    team := int(request.form['team']),
    targeted_productivity := float(request.form['targeted_productivity']),
    smv := float(request.form['smv']),
    overtime := float(request.form['overtime']),
    incentive := float(request.form['incentive']),
    no_of_style_change := int(request.form['no_of_style_change']),
    no_of_workers := float(request.form['no_of_workers'])
]

            input_data = np.array([features])
            prediction = model.predict(input_data)[0]
            if prediction <= 0.3:
                text = 'The employee is Averagely Productive'
            elif prediction >0.3 and prediction <=0.8:
                text = 'The employee is Medium Productive'
            else:
                text = 'The employee is Highly Productive'


            return render_template('predict.html', result=text)
        except Exception as e:
            return render_template('predict.html', result=f"Error: {e}")
    return render_template('predict.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
