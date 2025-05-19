import streamlit as st
import pandas as pd
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.title("📈 Stock Price Direction Predictor with Sentiment")

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

uploaded_file = st.file_uploader("Upload your stock CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", df.tail())

    # Check for sentiment
    if 'headline' in df.columns:
        analyzer = SentimentIntensityAnalyzer()
        df['avg_sentiment'] = df['headline'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    else:
        df['avg_sentiment'] = 0  # fallback if no headlines

    # Make sure feature columns exist
    feature_cols = ['Open', 'High', 'Low', 'Close/Last', 'Volume']
    if all(col in df.columns for col in feature_cols):
        latest = df[feature_cols].tail(1)
        prediction = model.predict(latest)

        st.subheader("🔮 Tomorrow's Price Direction Prediction:")
        if prediction[0] == 1:
            st.success("📈 The model predicts the price will go UP tomorrow.")
        else:
            st.error("📉 The model predicts the price will go DOWN tomorrow.")
    else:
        st.error("❗ Your file is missing one or more required columns: " + ", ".join(feature_cols))
