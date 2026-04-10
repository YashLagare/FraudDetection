from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        step           = float(request.form['step'])
        type_          = float(request.form['type'])
        amount         = float(request.form['amount'])
        oldbalanceOrg  = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])
        isFlaggedFraud = float(request.form['isFlaggedFraud'])

        input_data = np.array([[step, type_, amount,
                                 oldbalanceOrg, newbalanceOrig,
                                 oldbalanceDest, newbalanceDest,
                                 isFlaggedFraud]])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            result = "⚠️ FRAUDULENT TRANSACTION DETECTED!"
            result_class = "fraud"
        else:
            result = "✅ Transaction is Legitimate"
            result_class = "safe"

    except Exception as e:
        result = f"Error: {str(e)}"
        result_class = "fraud"

    return render_template('submit.html', result=result, result_class=result_class)

if __name__ == '__main__':
    app.run(debug=True)