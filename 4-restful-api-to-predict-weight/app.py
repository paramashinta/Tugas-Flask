from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def weight_prediction():
    if request.method == 'GET':
        return render_template("weight-prediction.html")
    elif request.method == 'POST':
        print(dict(request.form))
        weight_prediction = dict(request.form).values()
        weight_prediction = np.array([float(x) for x in weight_prediction])
        model, std_scaler = joblib.load("model-development/weight-prediction-using-linear-regression.pkl")
        weight_features = std_scaler.transform([weight_prediction])
        print(weight_features)
        result = model.predict(weight_features)
        weight = {
            '0' : 'Female',
            '1' : 'Male'
        }
        return render_template('weight-prediction.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5420, debug=True)