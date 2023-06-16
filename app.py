from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved Linear Regression model
with open('linear_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Define a function to preprocess the input data
# Update the preprocess_data function
def preprocess_data(total_sqft, bath, bhk,location):
    # Create a DataFrame with the input data
    data = {'location': [location],
        'total_sqft': [total_sqft],
            'bath': [bath],
            'bhk': [bhk]}
    df = pd.DataFrame(data)
    
    return df


# Define the route for the home page
@app.route('/')
def home():

    return render_template('index.html')

# Define the route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    location = request.form['location']
    total_sqft = float(request.form['total_sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    # Preprocess the input data
    input_data = preprocess_data(total_sqft, bath, bhk, location)
    
    # Make the prediction using the loaded model
    predicted_price = lr_model.predict(input_data)[0]
    
    # Render the prediction result in the template
    return render_template('index.html', prediction_text='Predicted Price: {:.2f} Lakhs'.format(predicted_price))

if __name__ == '__main__':
    app.run(debug=True)
