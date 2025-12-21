# 🥥 Coconut Yield Predictor: ML From Scratch

Successfully bridge the gap between complex mathematics and real-world agricultural insights. This project is a **Full-Stack Machine Learning application** designed to predict coconut production in India using a Linear Regression engine built entirely from scratch.

---

## 🌟 Key Features

- **Brain From Scratch:** Implements Linear Regression using fundamental **Linear Algebra**, **Calculus**, and **Statistics**. No high-level ML libraries were used for the core model.
- **Scientific Validation:** Implements a manual **80/20 Train-Test Split** using `train_test_split` to calculate real-world validation scores (R²) on unseen data.
- **Dual-Engine Benchmarking:** Integrated **Scikit-Learn** as a side-by-side benchmark, allowing users to compare the custom "Scratch" results against industry-standard libraries.
- **Educational UI:** Every prediction comes with a "Step-by-Step" math explanation, showing exactly how the Z-Scores, Dot Products, and Inverse Scaling were calculated.
- **Historical Reality Check:** Automatically matches user input against 15,000+ historical records to show "AI Prediction vs. Historical Reality."
- **Data Science Dashboard:** 
  - **Plot 1: Regression Fit** (Actual vs. AI Trendline vs. Industry Fit)
  - **Plot 2: Optimization Progress** (Gradient Descent Cost reduction)
  - **Plot 3: Residual Analysis** (Statistical error distribution)

---

## 🧠 The Mathematics Applied

This project explicitly implements the four pillars of Data Science:

1.  **Linear Algebra:** Vectorized operations and Mat-Mul for predictions ($y = Xw + b$).
2.  **Calculus:** Manual derivation of gradients (Partial Derivatives) to guide the Gradient Descent descent.
3.  **Statistics:** Z-Score Standardization, R² Score calculation, and Normal Distribution checks for residuals.
4.  **Optimization:** A custom Gradient Descent optimizer that iteratively minimizes the Mean Squared Error (MSE).

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, NumPy (for Matrix math), Pandas (for data cleaning).
- **Frontend:** Angular 19, Vanilla CSS (Premium Design), RxJS.
- **Deployment:** Docker-ready for Google Vertex AI or Render.

---

## 🚀 Quick Start (Local)

### 1. Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```
*Port: 8080*

### 2. Frontend
```bash
cd frontend
npm install
npm start
```
*Port: 4200*

---

## 📂 Project Structure
- `/backend`: Flask API & Custom ML Engine.
- `/frontend`: Angular UI & Interactive Charts.
- `Math_Implementation_Details.md`: Deep dive into every formula used in the code.
- `RENDER_DEPLOY.md`: Guide for free cloud hosting.

---

## 📈 Dataset
The model is trained on the **Indian Crop Production Dataset**, filtered specifically for Coconut yields across all Indian states and districts from 1997-2015.

---
*Created with ❤️ for Data Science Education.*
