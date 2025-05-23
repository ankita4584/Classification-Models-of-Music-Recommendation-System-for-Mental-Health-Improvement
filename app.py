from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

encoders = joblib.load('encoder.pkl') 
model = joblib.load('model.pkl')
  
feature_names = ['Exploratory', 'Anxiety', 'Frequency [R&B]', 'BPM', 'Age', 'While working', 'Insomnia', 'Depression', 'Frequency [Pop]', 'Fav genre']


categorical_features = ['Exploratory','Frequency [R&B]', 'While working',
     'Fav genre', 'Frequency [Pop]']

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_features = []
        for feature in feature_names:
            value = request.form[feature]
            if feature in categorical_features:
                encoder = encoders[feature] 
                value = encoder.transform([value])[0]  
            else:
                value=float(value)
            input_features.append(value)

        input_array = np.array([input_features])
        prediction = model.predict(input_array)[0]

        return render_template('index.html', prediction=prediction, features=feature_names)

    return render_template('index.html', prediction=None, features=feature_names)

if __name__ == '__main__':
    app.run(debug=True)

