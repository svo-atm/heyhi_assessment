# Deploy Your Chatbot

## Step 1: Test Locally

```bash
pip install streamlit
streamlit run app.py
```

Open http://localhost:8501 to test your chatbot.

## Step 2: Deploy to Streamlit Cloud (Free)

1. **Push to GitHub:**
   - Create a new repository on GitHub
   - Upload all your files

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set main file path: `app.py`

3. **Add your API key:**
   - In Streamlit Cloud, go to app settings
   - Add secrets: `OPENAI_API_KEY = "your-api-key-here"`

4. **Done!** Your chatbot will be live at: `https://your-app-name.streamlit.app`

## That's it!

Your chatbot is now deployed and accessible to anyone with the URL.
