# Deploying Coconut Yield Predictor to Google Vertex AI

This guide will help you deploy your Python Flask Backend to Google Vertex AI (Cloud Run or Custom Model Serving).

## Prerequisites
1.  **Google Cloud SDK** installed and authenticated (`gcloud init`).
2.  **Docker** installed and running.
3.  A **Google Cloud Project** with billing enabled.

## Step 1: Set Environment Variables
Open your terminal (PowerShell or Command Prompt) and set your project ID:
```powershell
$PROJECT_ID="your-google-cloud-project-id"
$REGION="us-central1"
$REPO_NAME="coconut-predictor-repo"
$IMAGE_NAME="coconut-backend"
```

## Step 2: Enable Required APIs
```bash
gcloud services enable artifactregistry.googleapis.com aiplatform.googleapis.com
```

## Step 3: Create Artifact Registry Repository
```bash
gcloud artifacts repositories create $REPO_NAME --repository-format=docker --location=$REGION --description="Docker repository for Coconut Yield Predictor"
```

## Step 4: Build and Push Docker Image
Navigate to the `backend` directory:
```bash
cd backend
```

Build the image (Note: Ensure Docker is running):
```bash
gcloud auth configure-docker ${REGION}-docker.pkg.dev
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest .
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest
```

## Step 5: Deploy to Vertex AI (as a Model)
1.  Go to the **Vertex AI** section in Google Cloud Console.
2.  Click **Model Registry** -> **Import**.
3.  **Name:** `coconut-yield-model`
4.  **Settings:**
    *   **Model Framework:** Custom
    *   **Container Image:** Browse and select the image you just pushed to Artifact Registry (`.../coconut-backend:latest`).
    *   **Prediction Route:** `/predict`
    *   **Health Route:** `/` (or add a simple `/health` endpoint to app.py if needed, usually `/` works if it returns 200).
    *   **Ports:** `8080`

## Step 6: Deploy to Endpoint
1.  Select your imported model in Vertex AI Model Registry.
2.  Click **Deploy to Endpoint**.
3.  Create a new Endpoint name (e.g., `coconut-endpoint`).
4.  Select machine type (e.g., `n1-standard-2`).
5.  Click **Deploy**.

## How the Architecture Moves to the Cloud
When you run these steps, **both the AI (Prediction) and the Data Science (Head 5, Plots) logic move together.** 

In our project:
*   **The Backend Container:** Houses the `app.py` script. This script contains the *Linear Regression Engine* (AI) AND the *Data Insight Logic* (Data Science). One deployment server handles both.
*   **The Frontend:** Once hosted, it simply makes calls to the single IP address/URL of your Vertex AI Endpoint.

## Step 7: Update Frontend for Production
Once you have your **Cloud Run URL** or **Vertex Endpoint URL**, you must update the `apiUrl` in your frontend code so it stops looking at your local computer (`127.0.0.1`).

1.  Open `c:\study\AI\Agent\ML\LinearRegression\frontend\src\app\app.component.ts`
2.  Change line 28:
```typescript
// FROM:
private apiUrl = 'http://127.0.0.1:8080';

// TO:
private apiUrl = 'https://coconut-backend-xyz-uc.a.run.app'; // <--- Your real cloud URL
```

## Summary of the "Big Move"
| Component | Local Location | Cloud Destination |
| :--- | :--- | :--- |
| **AI Model** | `app.py` (Custom Class) | Vertex AI Model Registry |
| **Data Science** | `app.route('/data')` | Vertex AI Model Registry |
| **The Dataset** | `crop_production.csv` | Bundled inside the Docker Image |
| **The User UI** | Angular dev server | Firebase Hosting / GCS |

This means the "AI" and "DS" parts **stay together** in the same backend container for efficiency. They are "bundled" together.
