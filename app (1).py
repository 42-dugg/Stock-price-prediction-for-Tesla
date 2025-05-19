
import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Add your model loading and prediction logic here

st.title("📈 Stock Price Direction Predictor with Sentiment")

uploaded_file = st.file_uploader("Upload your stock CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", df.head())

    if 'headline' in df.columns:
        analyzer = SentimentIntensityAnalyzer()
        df['sentiment'] = df['headline'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
        st.write(df[['headline', 'sentiment']].head())
