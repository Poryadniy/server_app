import os

import numpy as np
from flask import Flask, request, render_template
from tensorflow.python import keras
import tensorflow as tf
from tensorflow import keras



def create_app():
    app = Flask(__name__, template_folder='templates')

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # Load the trained model
            import os
            model_path = os.path.abspath('trained_model.h5')
            if os.path.exists(model_path):
                model = tf.keras.models.load_model('trained_model.h5')
            else:
                raise FileNotFoundError(
                    f"Model file '{model_path}' not found. Please ensure the model is trained and saved correctly.")

            # Get input values from the form
            input_data = [
                float(request.form['temperature']),
                float(request.form['pressure']),
                float(request.form['humidity'])
            ]

            # Prepare input for prediction
            input_array = np.array(input_data).reshape((1, -1, 1))
            prediction = model.predict(input_array)

            # Return the prediction result
            return render_template('index.html', prediction_text=f'Predicted Temperature: {prediction[0][0]:.2f} °C')
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

    return app


def main():
    # Create and run the web app
    app = create_app()
    app.run(debug=True)


if __name__ == "__main__":
    main()