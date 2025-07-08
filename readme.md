# 🧠 Machine Learning Flask API

This project is a Flask-based RESTful API that combines multiple machine learning models to perform:

- **Salary Prediction** using Linear Regression
- **Customer Segmentation** using K-Means Clustering
- **Handwritten Digit Recognition** using a Convolutional Neural Network (trained on the MNIST dataset)

It also provides data visualization in the form of plots encoded as base64 images.

---

## 🔧 How It Works

### 1. Linear Regression - Salary Prediction
Predicts an employee’s salary based on their years of experience using a trained linear regression model.

### 2. K-Means Clustering - Customer Segmentation
Clusters customers based on their annual income and spending score. Each cluster is labeled (e.g., "High Spender", "Frugal Spender", etc.).

### 3. Digit Recognition (MNIST)
Accepts a base64-encoded 28x28 grayscale image and uses a CNN model to predict the handwritten digit.

---

## 🧰 Models Used

### 1. **Linear Regression Model**
- Trained on `Salary_Data.csv`
- Predicts salary from years of experience

### 2. **KMeans Clustering**
- Trained on `Mall_Customers.csv`
- Identifies 5 distinct customer segments

### 3. **CNN for Digit Recognition**
- Trained on the MNIST dataset
- Architecture: Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)

---

## 📚 Libraries Used

| Library       | Purpose                                      |
|---------------|----------------------------------------------|
| Flask         | Backend server                               |
| Flask-CORS    | CORS handling                                |
| NumPy         | Numerical computations                       |
| Pandas        | Data loading and manipulation                |
| Matplotlib    | Data visualization                           |
| scikit-learn  | Linear Regression and KMeans Clustering      |
| TensorFlow    | Deep learning model (CNN for digit detection)|
| Pillow (PIL)  | Image processing                             |
| Joblib        | Model serialization                          |
| Base64        | Image encoding                               |

---

## ⚙️ Setup Instructions

### ✅ Requirements

- Python 3.7+
- `pip`

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
