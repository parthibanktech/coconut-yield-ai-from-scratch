import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CUSTOM LINEAR REGRESSION IMPLEMENTATION
# ==========================================
class LinearRegressionGradientDescent:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X, y):
        """
        Fits the model using Gradient Descent.
        X: Matrix of features (m samples, n features)
        y: Vector of target values (m samples)
        """
        # Linear Algebra: Get dimensions of the dataset
        n_samples, n_features = X.shape
        
        # Linear Algebra: Initialize parameters (weights) as vectors
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []

        # Optimization: Gradient Descent Loop
        for i in range(self.n_iterations):
            # 1. Linear Algebra: Compute predictions (Hypothesis function)
            # y_pred = X * w + b
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # 2. Statistics: Calculate Cost (Mean Squared Error)
            # Cost = (1/n) * sum((y_pred - y)^2)
            cost = (1 / n_samples) * np.sum((y_predicted - y) ** 2)
            self.cost_history.append(cost)
            
            # 3. Calculus: Compute Gradients (Derivatives)
            # dw = (2/n) * sum(x * (y_pred - y))
            # db = (2/n) * sum(y_pred - y)
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2 / n_samples) * np.sum(y_predicted - y)
            
            # 4. Optimization: Update Parameters
            # w = w - learning_rate * dw
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print cost every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}: Cost {cost:.2e}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# ==========================================
# DATA LOADING AND PREPROCESSING
# ==========================================
print("Loading dataset...")
try:
    df = pd.read_csv('crop_production.csv')
except FileNotFoundError:
    print("Error: 'crop_production.csv' not found.")
    exit()

# Cleanup
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
coconut_data = df[df['Crop'] == 'Coconut'].copy()
coconut_data = coconut_data.dropna(subset=['Production'])

# Use a smaller subset for demonstration if the dataset is huge, to prevent overflow/slow convergence
# (Optional, but good for stability with custom implementations)
coconut_data = coconut_data.head(500) 

# Select Features and Target
# We will use 'Area' to predict 'Production' for simplicity in visualization, 
# but the math works for multiple features too.
X_raw = coconut_data[['Area']].values
y_raw = coconut_data['Production'].values

# ==========================================
# STATISTICS: FEATURE SCALING
# ==========================================
# Gradient Descent works best when features are on a similar scale (Standardization)
# z = (x - mean) / std
mu_X = np.mean(X_raw, axis=0)
sigma_X = np.std(X_raw, axis=0)
X_scaled = (X_raw - mu_X) / sigma_X

mu_y = np.mean(y_raw)
sigma_y = np.std(y_raw)
y_scaled = (y_raw - mu_y) / sigma_y

# Split Data (Manual Split)
split_idx = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

print(f"\nTraining on {len(X_train)} samples.")

# ==========================================
# TRAINING THE MODEL
# ==========================================
model = LinearRegressionGradientDescent(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Predictions
y_pred_scaled = model.predict(X_test)

# Inverse Transform to get actual values
y_pred_actual = (y_pred_scaled * sigma_y) + mu_y
y_test_actual = (y_test * sigma_y) + mu_y

# ==========================================
# EVALUATION (STATISTICS)
# ==========================================
# Mean Squared Error
mse = np.mean((y_test_actual - y_pred_actual) ** 2)
# R-squared
ss_total = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
ss_residual = np.sum((y_test_actual - y_pred_actual) ** 2)
r2 = 1 - (ss_residual / ss_total)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")
print(f"Learned Weights (Vectors): {model.weights}")
print(f"Learned Bias: {model.bias}")

# ==========================================
# VISUALIZATION
# ==========================================
plt.figure(figsize=(15, 5))

# 1. Regression Line
plt.subplot(1, 3, 1)
plt.scatter(X_test * sigma_X + mu_X, y_test_actual, color='blue', label='Actual')
plt.plot(X_test * sigma_X + mu_X, y_pred_actual, color='red', label='Prediction (Line)')
plt.xlabel('Area')
plt.ylabel('Production')
plt.title('Linear Regression Fit')
plt.legend()

# 2. Optimization History (Calculus/Optimization)
plt.subplot(1, 3, 2)
plt.plot(model.cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Gradient Descent Optimization\n(Cost Reduction over Time)')

# 3. Residual Analysis (Probability/Statistics)
# Verify if residuals are normally distributed
residuals = y_test_actual - y_pred_actual
plt.subplot(1, 3, 3)
plt.hist(residuals, bins=20, edgecolor='k')
plt.xlabel('Residual Error')
plt.ylabel('Frequency')
plt.title('Residual Distribution\n(Should be Bell Curve)')

plt.tight_layout()
plt.show()
