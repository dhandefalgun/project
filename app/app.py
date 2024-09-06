
from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd

app = Flask(__name__)

# Paths to the model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, '..', 'model.pkl')
scaler_path = os.path.join(BASE_DIR, '..', 'scaler.pkl')

# Load the model and scaler
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

# Home route - could be used for an introduction or welcome page
@app.route('/')
def home():
    return render_template('index.html')

# Route to render the prediction form
@app.route('/predict')
def index():
    return render_template('p.html')

@app.route('/About')
def about():
    return render_template('about.html')

@app.route('/explore')
def explore():
    return render_template('explore.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/Weather')
def weather():
    return render_template('rain.html')






   

# Route that handles the prediction logic
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting form data
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Creating a DataFrame from the input data
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Scaling the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Predicting the crop using the model
        prediction = model.predict(input_data_scaled)
        
        # Returning the prediction as a JSON response
        return jsonify({'crop': prediction[0]})
    
    except Exception as e:
        # Returning an error message if something goes wrong
        return jsonify({'error': str(e)}), 500

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)




