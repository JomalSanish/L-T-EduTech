from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from PIL import Image
import tensorflow as tf  # For deep learning model (digit prediction)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models
linear_model = joblib.load('models/linear_regression_model.pkl')
kmeans_model = joblib.load('models/kmeans_model.pkl')  # Pre-trained KMeans model
digit_model = tf.keras.models.load_model('models/digit_recognition_model.h5')  # Assuming a pre-trained digit prediction model

# Load datasets
customers = pd.read_csv('models/Mall_Customers.csv')
salary_data = pd.read_csv('models/Salary_Data.csv')

# Define cluster labels (Assumption: Cluster numbers 0, 1, 2, 3, and 4 are mapped to specific spenders)
CLUSTER_LABELS = {
    0: "Small Spender",
    1: "Medium Spender",
    2: "High Spender",
    3: "Frugal Spender",
    4: "Luxury Spender"
}

# Function to preprocess image and convert it to a 784-dimensional array (flattened)
def preprocess_image(image_data):
    # Convert the base64 image to a PIL image
    image = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize the image to 28x28 pixels
    
    # Convert the image to a numpy array and normalize the pixel values (0-1)
    image_array = np.array(image).reshape(1, 784).astype('float32') / 255.0
    return image_array

# Linear Regression Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        years_experience = np.array([[float(data.get('years_experience'))]])
        predicted_salary = linear_model.predict(years_experience)[0]
        return jsonify({"predicted_salary": predicted_salary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Linear Regression Plot
@app.route('/plot', methods=['GET'])
def plot():
    try:
        X = salary_data['YearsExperience']
        y = salary_data['Salary']

        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, color='blue', label='Data')
        plt.plot(X, linear_model.predict(X.values.reshape(-1, 1)), color='red', label='Prediction')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.title('Linear Regression Prediction vs Actual')
        plt.legend()

        # Encode plot to base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        return jsonify({"plot": img_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# K-Means Clustering Visualization
@app.route('/kmeans', methods=['POST'])
def kmeans():
    try:
        # Prepare input features
        X = customers[['Annual Income (k$)', 'Spending Score (1-100)']].values

        # Predict clusters
        clusters = kmeans_model.predict(X)
        customers['Cluster'] = clusters

        # Plot the clusters and label them
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=customers['Cluster'], cmap='viridis', s=50)
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.title('K-Means Clustering of Customers')

        # Add cluster labels to the plot
        for cluster_id, label in CLUSTER_LABELS.items():
            cluster_center = kmeans_model.cluster_centers_[cluster_id]
            plt.text(cluster_center[0], cluster_center[1], label, fontsize=10, ha='center', va='center', 
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

        # Encode plot to base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

        return jsonify({"plot": img_base64, "message": "Cluster plot created successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Predict Customer Type based on Input
@app.route('/predict_customer_type', methods=['POST'])
def predict_customer_type():
    try:
        data = request.json
        annual_income = float(data.get('annual_income'))
        spending_score = float(data.get('spending_score'))

        # Prepare input feature
        input_data = np.array([[annual_income, spending_score]])
        predicted_cluster = kmeans_model.predict(input_data)[0]
        customer_type = CLUSTER_LABELS.get(predicted_cluster, "Unknown")

        return jsonify({
            "predicted_cluster": int(predicted_cluster),
            "customer_type": customer_type,
            "message": f"Predicted Customer Type: {customer_type}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to predict the digit from the image
# Route to predict the digit from the image
@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    try:
        # Expect JSON payload with base64 image data
        data = request.json
        image_data = base64.b64decode(data['image'])  # Decode the base64 image data

        # Preprocess the image
        image = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28 pixels
        image_array = np.array(image).astype('float32') / 255.0  # Normalize pixel values to [0, 1]
        image_array = image_array.reshape(1, 28, 28, 1)  # Reshape to match the model input (batch_size, 28, 28, 1)

        # Predict the digit using the model
        prediction = digit_model.predict(image_array)
        predicted_digit = np.argmax(prediction)  # Get the class with the highest probability

        return jsonify({
            "predicted_digit": int(predicted_digit),
            "confidence": float(np.max(prediction)),  # Optional: Confidence score for the prediction
            "message": "Prediction successful!"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)
