# Deploying to Render (Blueprint Method - RECOMMENDED)

This method uses the `render.yaml` file to deploy both your **Backend** and **Frontend** at the same time. This is the "Correct Solution" to ensure they work together perfectly.

## Prerequisites
1.  A **GitHub** account.
2.  Your project pushed to a GitHub repository.
3.  A **Render.com** account.

---

## Step 1: Deploy with Blueprints (The "Correct" way)
1.  Log in to [Render Dashboard](https://dashboard.render.com).
2.  Click **New +** -> **Blueprint**.
3.  Connect your GitHub repository.
4.  Render will find the `render.yaml` file automatically.
5.  Click **Apply**.

**What happens now?**
Render will start building two things:
*   `coconut-backend` (Web Service)
*   `coconut-yield-app` (Static Site)

---

## Step 2: Link the Backend to the Frontend
Once the builds are finished:
1.  Go to your Render Dashboard and find **`coconut-backend`**.
2.  Copy its URL (e.g., `https://coconut-backend.onrender.com`).
3.  Open `frontend/src/environments/environment.prod.ts` in your code editor.
4.  Update the `apiUrl` with your copied URL:
    ```typescript
    export const environment = {
        production: true,
        apiUrl: 'https://coconut-backend.onrender.com' // <-- Paste your URL here
    };
    ```
5.  **Commit and Push** this change to GitHub. Render will automatically re-deploy your frontend.

---

## Troubleshooting "Why only one shown?"
If you followed the old manual steps, Render might be confused. The Blueprint method fixes this because:
1.  **Grouped Services:** Both services will now appear under a single "Blueprint" group.
2.  **Correct Paths:** The `rootDir` settings are already handled in the code.

## Crucial Local Fix (If you deleted your services)
If you deleted your Render services to start over:
1.  Go to **Blueprints** in Render.
2.  Delete any old blueprints if they exist.
3.  Follow **Step 1** above again.

---

## Technical Details (For your reference)
*   **Backend Port:** `8080` (Docker).
*   **Frontend Output:** `dist/frontend/browser` (Angular 21).
*   **Spin-down:** The free tier sleeps after 15 mins. The first load will be slow!
