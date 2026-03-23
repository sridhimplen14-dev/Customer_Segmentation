# Deploy to Streamlit Cloud - Make Your App Public

## Step 1: Push to GitHub (if not done)

Make sure your code is pushed to GitHub:
```bash
git push -u origin main --force
```

## Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io
   - Sign in with your **GitHub account**

2. **Create New App**
   - Click the **"New app"** button
   - Or click **"Deploy an app"**

3. **Configure Your App**
   - **Repository**: Select `SridhimpleNuthalapati/Customer-Segmentation-System`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Will be auto-generated (e.g., `customer-segmentation-system.streamlit.app`)

4. **Advanced Settings** (Optional)
   - Click **"Advanced settings"**
   - Python version: `3.11` or `3.12` (auto-detected)
   - Dependencies: Streamlit will auto-detect `requirements.txt`

5. **Deploy**
   - Click **"Deploy"** button
   - Wait 2-3 minutes for deployment

## Step 3: Share Your App

Once deployed, you'll get a public URL like:
```
https://customer-segmentation-system.streamlit.app
```

**Share this URL with anyone!** They can:
- Access your app from any browser
- No installation needed
- Works on mobile, tablet, desktop

## Step 4: Auto-Deploy Updates

Every time you push to GitHub:
- Streamlit Cloud automatically redeploys your app
- Updates go live in 1-2 minutes
- No manual redeployment needed!

## Troubleshooting

### App Won't Deploy
- Check that `requirements.txt` exists and has all dependencies
- Verify `app.py` is in the root directory
- Check deployment logs in Streamlit Cloud dashboard

### App Crashes
- Check logs in Streamlit Cloud dashboard
- Verify data file path is correct (or use file uploader)
- Ensure all imports work

### Can't Find Repository
- Make sure repository is **Public** (or you've granted Streamlit Cloud access)
- Verify you're signed in with the correct GitHub account

## Making Repository Public (if needed)

If your repo is private:
1. Go to your GitHub repository
2. Click **Settings** → **General**
3. Scroll to **Danger Zone**
4. Click **Change visibility** → **Make public**

Or keep it private and grant Streamlit Cloud access (requires GitHub Pro/Team).
