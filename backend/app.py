from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import base64
import io
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server
import matplotlib.pyplot as plt

app = Flask(__name__)
# Allow specific frontend origins (CORS)
CORS(app, resources={r"/*": {
    "origins": [
        "http://localhost:4200", 
        "http://127.0.0.1:4200",
        "https://coconut-yield-app.onrender.com"
    ]
}})

# ==========================================
# ML MODEL IMPLEMENTATION (Math from Scratch)
# ==========================================
# This implementation uses the 4 Pillars of Science:
# 1. Linear Algebra: Vectorization of weights and inputs.
# 2. Statistics: Feature scaling (Standardization) and R² score.
# 3. Calculus: Gradient calculation via partial derivatives.
# 4. Optimization: Parameter updates using Gradient Descent.

class LinearRegressionGradientDescent:
    def __init__(self, learning_rate=0.01, n_iterations=3000):
        self.learning_rate = learning_rate # Optimization: Step size for updates
        self.n_iterations = n_iterations   # Optimization: Number of training cycles
        self.weights = None                # Linear Algebra: Parameter vector (w)
        self.bias = None                   # Linear Algebra: Constant offset (b)
        self.cost_history = []             # Statistics: Tracking Error over time
         
    def fit(self, X, y):
        # Linear Algebra: Extract matrix dimensions (m samples, n features)
        n_samples, n_features = X.shape
        # Linear Algebra: Initialize weights as a zero vector of size n
        self.weights = np.zeros(n_features)
        # Statistics: Initialize bias (intercept) as zero
        self.bias = 0
        # Statistics: Clear history for a fresh training session
        self.cost_history = []

        # Optimization Loop
        for i in range(self.n_iterations):
            # 1. Hypothesis: y = Xw + b (Linear Algebra Dot Product)
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # 2. Statistics: Calculate Cost (Mean Squared Error Equation)
            # Cost = (1/n) * sum((pred - actual)^2)
            cost = (1 / n_samples) * np.sum((y_predicted - y) ** 2)
            # Statistics: Record cost for the Optimization progress graph
            self.cost_history.append(cost)

            # 3. Calculus: Compute Gradients (Partial Derivatives of Cost)
            # Derivative w.r.t weights (dw): Vectorized form of the chain rule
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            # Derivative w.r.t bias (db): Simple sum of residuals
            db = (2 / n_samples) * np.sum(y_predicted - y)
            
            # 4. Optimization: Update Parameters (Gradient Descent rule)
            # New_parameter = Old_parameter - (Learning_Rate * Gradient)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Linear Algebra: Apply learned parameters to new input matrix
        return np.dot(X, self.weights) + self.bias

# Global Storage for Model and Statistics
model = None            # The trained AI engine
feature_means = None    # Statistics: Training means (for normalization)
feature_stds = None     # Statistics: Training deviations (for normalization)
target_mean = 0         # Statistics: Production mean (for inverse scale)
target_std = 1          # Statistics: Production deviation (for inverse scale)
head_preview = []       # Data Preview: First 5 rows of raw data
historical_data = None  # Full Dataset: For historical lookup matches

# Global Analytics Storage (Base64 Images)
regression_plot_b64 = ""
cost_plot_b64 = ""
residual_plot_b64 = ""

def train_model():
    global model, feature_means, feature_stds, target_mean, target_std, head_preview, historical_data, regression_plot_b64, cost_plot_b64, residual_plot_b64
    try:
        # Statistics: Loading and Cleaning Data
        print("Loading dataset...")
        df = pd.read_csv('crop_production.csv')
        # Statistics: Remove whitespace from categorical names
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        # Statistics: Filter specifically for Coconuts (Feature Selection)
        data = df[df['Crop'] == 'Coconut'].copy()
        # Statistics: Remove rows with missing numeric values (Data Cleaning)
        data = data.dropna(subset=['Production', 'Area'])
        # Statistics: Convert Year to numeric to enable mathematical operations
        data['Crop_Year'] = pd.to_numeric(data['Crop_Year'], errors='coerce')
        data = data.dropna(subset=['Crop_Year'])
        historical_data = data
        # Statistics: Store a snippet for the frontend table
        head_preview = data.head(5).to_dict(orient='records')
        
        # Linear Algebra: Separate into Feature Matrix (X) and Target Vector (y)
        X_raw = data[['Area', 'Crop_Year']].values
        y_raw = data['Production'].values

        # Statistics: Calculate Normalization parameters (Mean and STD)
        feature_means = np.mean(X_raw, axis=0) # mu for inputs
        feature_stds = np.std(X_raw, axis=0)  # sigma for inputs
        target_mean = np.mean(y_raw)           # mu for output
        target_std = np.std(y_raw)             # sigma for output

        # Statistics: Apply Standardization (Z-Score) -> z = (x - mu) / sigma
        X_scaled = (X_raw - feature_means) / feature_stds
        y_scaled = (y_raw - target_mean) / target_std

        # Statistics: Manual Train/Test Split (80% for learning, 20% for testing)
        split_idx = int(0.8 * len(X_scaled))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

        # Optimization: Instantiate and train the manual model
        print(f"Training on {len(X_train)} samples...")
        model = LinearRegressionGradientDescent(learning_rate=0.01, n_iterations=3000)
        model.fit(X_train, y_train)
        
        # --- CALCULUS & STATISTICS: DIAGNOSTIC PLOTS ---
        
        # Diagnostic 1: Regression Fit Visualization
        plt.figure(figsize=(6, 4))
        # Select sub-sample for plot clarity
        X_test_viz = X_test[:100]
        y_test_viz = y_test[:100]
        # Linear Algebra: Predict using the learned weights
        y_pred_viz = model.predict(X_test_viz)
        
        # Statistics: Inverse transformation to bring data back to real-world units
        X_area_actual = X_test_viz[:, 0] * feature_stds[0] + feature_means[0]
        y_actual = (y_test_viz * target_std) + target_mean
        y_pred = (y_pred_viz * target_std) + target_mean
        
        # Probability: Scatter plot of real samples
        plt.scatter(X_area_actual, y_actual, color='#3b82f6', label='Actual', alpha=0.6)
        # Geometry: The Linear Equation line
        plt.plot(X_area_actual, y_pred, color='#ef4444', label='AI Fit Line', linewidth=2)
        plt.title('Plot 1: Linear Regression Fit')
        plt.xlabel('Area (Hectares)')
        plt.ylabel('Production')
        plt.legend()
        plt.tight_layout()
        img1 = io.BytesIO()
        plt.savefig(img1, format='png')
        img1.seek(0)
        regression_plot_b64 = base64.b64encode(img1.getvalue()).decode()
        plt.close()

        # Diagnostic 2: Optimization History (Calculus progress)
        plt.figure(figsize=(6, 4))
        # Optimization: Plotting the "descent" into the error minimum
        plt.plot(model.cost_history, color='#7c3aed', linewidth=2)
        plt.title('Plot 2: Optimization Progress')
        plt.xlabel('Iterations')
        plt.ylabel('Cost (MSE Error)')
        plt.tight_layout()
        img2 = io.BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        cost_plot_b64 = base64.b64encode(img2.getvalue()).decode()
        plt.close()

        # Diagnostic 3: Residual Analysis (Statistical Error check)
        y_full_pred = model.predict(X_test)
        # Statistics: Calculate Errors (Residuals = Actual - predicted)
        residuals = (y_test - y_full_pred) * target_std
        plt.figure(figsize=(6, 4))
        # Statistics: Verify if distribution is Gaussian (Normal)
        plt.hist(residuals, bins=30, color='#10b981', edgecolor='white')
        plt.title('Plot 3: Error Distribution (Residuals)')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.tight_layout()
        img3 = io.BytesIO()
        plt.savefig(img3, format='png')
        img3.seek(0)
        residual_plot_b64 = base64.b64encode(img3.getvalue()).decode()
        plt.close()

        print("Model trained and all 3 diagnostic charts generated.")
        
    except Exception as e:
        print(f"CRITICAL ERROR training model: {e}")

# Train immediately
train_model()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts yield based on Area and Year.
    Steps:
    1. Scale user input using Z-Score Statistics.
    2. Apply Linear Algebra Dot Product with learned weights.
    3. Calculus-based explanation of the result.
    4. Inverse scale to get actual production value.
    """
    global model, historical_data
    try:
        if model is None:
            return jsonify({'error': 'Model is still training. Please wait.'}), 503

        data = request.json
        
        # 1. Parse and process input (Statistics: Handling missing values)
        try:
            area = float(data.get('area'))
        except:
            return jsonify({'error': 'Area is required'}), 400
            
        year_input = data.get('year')
        if year_input is None or year_input == "":
            year = feature_means[1] # Use mean year as default (Imputation)
        else:
            year = float(year_input)

        # 2. Linear Algebra: Prepare input feature vector
        input_features = np.array([[area, year]])
        
        # 3. Statistics: Scale input features using training Z-scores
        input_scaled = (input_features - feature_means) / feature_stds
        
        # 4. Calculus/Linear Algebra Prediction
        # y_pred = w1*x1 + w2*x2 + b
        pred_scaled = model.predict(input_scaled)
        
        # 5. Statistics: Inverse Scale output (z -> actual)
        # Actual = (z * sigma) + mu
        pred_actual = (pred_scaled * target_std) + target_mean
        result = max(0, float(pred_actual[0]))

        # 6. Check Historical Reality
        actual_match = None
        if historical_data is not None:
            # Find exact match for Area and Year (and maybe approximate Area since float)
            # Using a small tolerance for float comparison or exact match if int
            match = historical_data[
                (np.isclose(historical_data['Area'], area, atol=0.1)) & 
                (historical_data['Crop_Year'] == year)
            ]
            if not match.empty:
                row = match.iloc[0]
                actual_match = {
                    'production': float(row['Production']),
                    'state': row['State_Name'],
                    'district': row['District_Name']
                }

        # 7. Generate Math Explanation
        math_steps = {
            "step1_scaling": f"Normalized Input: Area Z-Score = ({area} - {feature_means[0]:.2f}) / {feature_stds[0]:.2f} = {input_scaled[0][0]:.4f}",
            "step2_prediction": f"Linear Equation: y = ({model.weights[0]:.4f} * Area_Z) + ({model.weights[1]:.4f} * Year_Z) + {model.bias:.4f}",
            "step3_result": f"Raw Prediction (Scaled): {pred_scaled[0]:.4f}",
            "step4_inverse": f"Final Output: ({pred_scaled[0]:.4f} * {target_std:.2f}) + {target_mean:.2f} = {result:.2f}"
        }

        # 8. Dynamic Graph
        plt.figure(figsize=(8, 4))
        df_head = pd.DataFrame(head_preview)
        if not df_head.empty:
            plt.scatter(df_head['Area'], df_head['Production'], color='lightgray', label='Historical Data', alpha=0.5)
        
        # Plot Prediction vs Actual (if available)
        plt.scatter(area, result, color='red', s=150, zorder=5, label='AI Prediction')
        if actual_match:
             plt.scatter(area, actual_match['production'], color='green', s=150, zorder=6, marker='*', label='Actual History')

        plt.xlabel('Area (Hectares)')
        plt.ylabel('Production (Coconuts)')
        plt.title('Prediction vs Reality')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        prediction_plot = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # 9. Global View Graph (Personalized Plot 1)
        plt.figure(figsize=(8, 4))
        # Plot a subset of historical data for background (Statistics: Reference Distribution)
        if historical_data is not None:
             sample_bg = historical_data.sample(min(200, len(historical_data)))
             plt.scatter(sample_bg['Area'], sample_bg['Production'], color='lightgray', label='National Data', alpha=0.3, s=20)
        
        # Plot the AI Trend Line
        x_range = np.linspace(0, historical_data['Area'].max(), 100)
        # Simplify to area-based trend for visual clarity on 2D plot
        y_trend = ( ( (x_range - feature_means[0])/feature_stds[0] ) * model.weights[0] + model.bias) * target_std + target_mean
        plt.plot(x_range, y_trend, color='#ef4444', label='AI Trend Line', linewidth=2)
        
        # THE IMPORTANT PART: Plot USER DATA on the Global Map
        plt.scatter(area, result, color='red', s=200, zorder=10, label='YOU (Prediction)', edgecolor='white', linewidth=2)
        
        plt.xlabel('Area (Hectares)')
        plt.ylabel('Production')
        plt.title('Nation-wide Context: Where you fit in India')
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        
        img_global = io.BytesIO()
        plt.savefig(img_global, format='png')
        img_global.seek(0)
        global_context_plot = base64.b64encode(img_global.getvalue()).decode()
        plt.close()

        return jsonify({
            'success': True,
            'input': {'area': area, 'year': year},
            'predicted_production': result,
            'explanation': math_steps,
            'context_plot': f"data:image/png;base64,{prediction_plot}",
            'global_context_plot': f"data:image/png;base64,{global_context_plot}",
            'historical_match': actual_match
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/data', methods=['GET'])
def get_data_insight():
    try:
        # 1. Generate Graph
        plt.figure(figsize=(10, 6))
        # We plot Area vs Production for visualization (simplification)
        df_head = pd.DataFrame(head_preview)
        if not df_head.empty:
            plt.scatter(df_head['Area'], df_head['Production'], color='blue', label='Actual Samples', s=100)
            plt.xlabel('Area (Hectares)')
            plt.ylabel('Production (Coconuts)')
            plt.title('Sample Dataset Visualization (First 5 Rows)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Convert plot to Base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return jsonify({
            'head': head_preview,
            'plot_image': f"data:image/png;base64,{regression_plot_b64}",
            'cost_plot': f"data:image/png;base64,{cost_plot_b64}",
            'residual_plot': f"data:image/png;base64,{residual_plot_b64}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

import os

@app.route('/', methods=['GET'])
def health_check():
    """Simple health check for Render/Cloud monitoring"""
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'message': 'Parthiban AI Backend is running'
    }), 200

if __name__ == '__main__':
    train_model() # Train before starting server
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
