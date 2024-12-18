import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

def train_neural_network():
    """
    Train a neural network on the MNIST dataset and save it in HDF5 format.
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # One-hot encode the labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    # Build the neural network model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train_cat, epochs=10, validation_data=(x_test, y_test_cat))

    # Evaluate the model
    loss, acc = model.evaluate(x_test, y_test_cat)
    print(f'Neural Network Accuracy: {acc}')

    # Save the trained model in HDF5 format
    model.save('digit_recognition_model.h5')
    print('Neural Network model saved as "digit_recognition_model.h5".')

if __name__ == '__main__':
    print("\nTraining Neural Network Model (MNIST)...")
    train_neural_network()
