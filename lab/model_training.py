import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from keras.layers import LSTM


def prepare_training_data(df, target_column='T (degC)', time_steps=24):
    # Prepare training data for LSTM model
    data = df[[target_column]].values
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])

    X, y = np.array(X), np.array(y)
    print("Training data prepared.")
    return X, y


def train_model(X, y):
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = keras.Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    print("Training the model...")
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)
    print("Model training complete.")

    # Save the trained model
    model.save('trained_model.h5')
    print("Model saved to trained_model.h5.")


def main():
    from data_preprocessing import preprocess_data
    from data_loader import load_data
    df = load_data()
    df_preprocessed = preprocess_data(df)
    X, y = prepare_training_data(df_preprocessed)
    train_model(X, y)


if __name__ == "__main__":
    main()