# Mathematical Foundations of the Coconut Yield Prediction Model

This document explains the mathematical concepts implemented in `coconut_yield_regression.py` and `backend/app.py` from scratch.

## 1. Linear Algebra
Linear algebra allows us to process large datasets efficiently. Instead of looping through every single data point, we use **vectors** and **matrices** to perform calculations in one go.

### **Implementation in Code:**
- **Variables:**
  - `X`: A matrix of size $(m \times n)$, where $m$ is the number of samples (farms) and $n$ is the number of features (e.g., Area).
  - `self.weights` ($w$): A vector of size $n$.
  - `self.bias` ($b$): A scalar value.
  
- **Hypothesis Function (Equation of a Line/Plane):**
  We want to find $y = wx + b$. In matrix form:
  $$ \hat{y} = X \cdot w + b $$
  
  **Code Map:**
  ```python
  # y_predicted = np.dot(X, self.weights) + self.bias
  ```
  `np.dot` performs the matrix multiplication.

---

## 2. Statistics
Statistics provides the tools to measure how "wrong" our model is and to prepare our data.

### **A. Feature Scaling (Standardization)**
Gradient Descent struggles if features have vastly different ranges (e.g., Area=1000 vs Rain=5). We normalize them to have mean 0 and standard deviation 1.
$$ z = \frac{x - \mu}{\sigma} $$

**Code Map:**
```python
mu_X = np.mean(X_raw, axis=0)
sigma_X = np.std(X_raw, axis=0)
X_scaled = (X_raw - mu_X) / sigma_X
```

### **B. Cost Function (Mean Squared Error)**
To train the model, we need a single number that represents its error. We use MSE:
$$ J(w,b) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 $$

**Code Map:**
```python
cost = (1 / n_samples) * np.sum((y_predicted - y) ** 2)
```

### **C. Residual Analysis**
After training, we check the **Residuals** (difference between actual and predicted).
$$ Residual = y - \hat{y} $$
If the residuals follow a **Normal Distribution** (Bell Curve), our linear model is statistically valid.

---

## 3. Calculus
Calculus tells us **how to change** the weights to reduce the error. We need the "slope" of the cost function with respect to each weight.

### **Gradients (Partial Derivatives)**
We take the derivative of the Cost Function $J$ with respect to weights $w$ and bias $b$.

**Derivative for Weights ($dw$):**
$$ \frac{\partial J}{\partial w} = \frac{2}{m} \sum (x \cdot (\hat{y} - y)) $$

**Derivative for Bias ($db$):**
$$ \frac{\partial J}{\partial b} = \frac{2}{m} \sum (\hat{y} - y) $$

**Code Map:**
```python
dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
db = (2 / n_samples) * np.sum(y_predicted - y)
```

---

## 4. Optimization
Optimization is the process of actually moving towards the best solution.

### **Gradient Descent**
We iteratively update the weights by moving in the opposite direction of the gradient (downhill).
$$ w_{new} = w_{old} - \alpha \cdot \frac{\partial J}{\partial w} $$
$$ b_{new} = b_{old} - \alpha \cdot \frac{\partial J}{\partial b} $$
*Where $\alpha$ (alpha) is the **Learning Rate** (step size).*

**Code Map:**
```python
self.weights -= self.learning_rate * dw
self.bias -= self.learning_rate * db
```
