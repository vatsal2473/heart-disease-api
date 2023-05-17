from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# load the random forest model from the pkl file
rf_model = joblib.load('random_forest_model.pkl')

# create a Flask app object
app = Flask(__name__)
CORS(app)

# define an endpoint URL and a function to handle incoming requests
@app.route('/predict', methods=['POST'])
def predict():
    # extract input feature values from the request data
    if request.method == "POST":
        age = int(request.form.get('age'))
        sex = int(request.form.get('sex'))
        cp = int(request.form.get('cp'))
        trestbps = int(request.form.get('trestbps'))
        chol = int(request.form.get('chol'))
        fbs = int(request.form.get('fbs'))
        restecg = int(request.form.get('restecg'))
        thalach = int(request.form.get('thalach'))
        exang = int(request.form.get('exang'))
        oldpeak = float(request.form.get('oldpeak'))
        slope = int(request.form.get('slope'))
        ca = int(request.form.get('ca'))
        thal = int(request.form.get('thal'))

    # transform the extracted input values into a format that can be used by the model for prediction
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # use the loaded model to make a prediction on the input data
    prediction = rf_model.predict(input_data)

    # return the predicted value as a JSON response
    return jsonify({'prediction': int(prediction)})


# run the app
if __name__ == '__main__':
    app.run(debug=True)
