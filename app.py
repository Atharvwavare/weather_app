from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        humidity = float(request.form['humidity'])
        pressure = float(request.form['pressure'])
        wind_speed = float(request.form['wind_speed'])

        features = np.array([[humidity, pressure, wind_speed]])
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=round(prediction, 2))

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
