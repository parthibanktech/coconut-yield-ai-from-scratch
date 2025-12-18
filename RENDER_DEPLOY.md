# Deploying to Render (Free & Simple)

Render is an excellent alternative to Vertex AI for free hosting. It will automatically build your Docker container whenever you push code to GitHub.

## Prerequisites
1.  A **GitHub** account.
2.  Your project pushed to a GitHub repository.
3.  A **Render.com** account (Free).

---

## Step 1: Prepare your Repository
Ensure your folder structure looks like this on GitHub:
```text
/ (Root)
  ├── backend/
  │    ├── app.py
  │    ├── Dockerfile
  │    ├── requirements.txt
  │    └── crop_production.csv
  └── frontend/ (Angular project)
```

## Step 2: Deploy the Backend (API)
1.  Log in to [Render Dashboard](https://dashboard.render.com).
2.  Click **New +** -> **Web Service**.
3.  Connect your GitHub repository.
4.  **Name:** `coconut-backend`
5.  **Region:** Select one closest to you (e.g., Singapore or US West).
6.  **Branch:** `main` (or yours).
7.  **Root Directory:** `backend`  <-- **CRITICAL: Set this to "backend"**
8.  **Runtime:** `Docker` (Render will detect your `Dockerfile`).
9.  **Instance Type:** `Free`.
10. Click **Create Web Service**.

**Wait for build:** Render will take 2-5 minutes to build your container. Once done, you will get a URL like `https://coconut-backend-xyz.onrender.com`.

---

## Step 3: Deploy the Frontend (Static Site)
1.  In Render Dashboard, click **New +** -> **Static Site**.
2.  Connect the same GitHub repository.
3.  **Name:** `coconut-yield-app`
4.  **Root Directory:** `frontend`
5.  **Build Command:** `npm install && npm run build`
6.  **Publish Directory:** `dist/frontend/browser` (Check your Angular project to confirm the path, usually starts with `dist/`).
7.  Click **Create Static Site**.

---

## Step 4: Link the Two
Once your backend is live on Render:
1.  Open `frontend/src/app/app.component.ts`.
2.  Update the `apiUrl` to your new Render backend URL:
    ```typescript
    private apiUrl = 'https://coconut-backend-xyz.onrender.com';
    ```
3.  Push this change to GitHub. Render will automatically re-deploy your frontend.

---

## Crucial Note on the Free Tier
*   **Spin-down:** Render's free tier "sleeps" after 15 minutes of inactivity. When you visit the app after it sleeps, the first request might take **30-60 seconds** to wake up the server. This is normal for free hosting!
*   **No Credit Card:** Unlike Google Vertex AI, Render does not require a credit card for the free tier.
