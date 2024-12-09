import streamlit as st
import pandas as pd
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

@st.cache_resource
def load_transformers_pipeline():
    return pipeline("sentiment-analysis", model = "siebert/sentiment-roberta-large-english")

@st.cache_resource
def load_vader_analyzer():
    return SentimentIntensityAnalyzer()

transformers_analyzer = load_transformers_pipeline()
vader_analyzer = load_vader_analyzer()

# Helper functions for VADER and TextBlob
def analyze_with_vader(text):
    scores = vader_analyzer.polarity_scores(text)
    return "POSITIVE" if scores["compound"] > 0 else "NEGATIVE" if scores["compound"] < 0 else "NEUTRAL"

def analyze_with_textblob(text):
    polarity = TextBlob(text).sentiment.polarity
    return "POSITIVE" if polarity > 0 else "NEGATIVE" if polarity < 0 else "NEUTRAL"

# Streamlit App Title
st.title("Sentiment Analysis App")
st.write("Perform sentiment analysis using Transformers, VADER, and TextBlob simultaneously.")

# Sidebar for input options
st.sidebar.title("Input Options")
input_option = st.sidebar.radio(
    "Choose an input method:",
    ("Type Text", "Upload Excel File")
)

if input_option == "Type Text":
    # Text input
    st.subheader("Type in your text for analysis")
    user_text = st.text_area("Enter your text here:", placeholder="Type something...")
    if st.button("Analyze Sentiment"):
        if user_text.strip():
            transformers_result = transformers_analyzer(user_text)[0]["label"]
            vader_result = analyze_with_vader(user_text)
            textblob_result = analyze_with_textblob(user_text)
            
            st.write("**Results**")
            st.write(f"**Transformers Sentiment:** {transformers_result}")
            st.write(f"**VADER Sentiment:** {vader_result}")
            st.write(f"**TextBlob Sentiment:** {textblob_result}")
        else:
            st.warning("Please enter some text to analyze.")
elif input_option == "Upload Excel File":
    # File uploader
    st.subheader("Upload an Excel file containing reviews")
    uploaded_file = st.file_uploader("Upload your file (Excel format)", type=["xlsx"])
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(df.head())
            
            if "review" not in df.columns:
                st.error("The uploaded file must contain a column named 'review'.")
            else:
                st.write("Analyzing Sentiments...")

                # Perform sentiment analysis for all rows
                df["Transformers Sentiment"] = df["review"].apply(lambda x: transformers_analyzer(x)[0]["label"])
                df["VADER Sentiment"] = df["review"].apply(analyze_with_vader)
                df["TextBlob Sentiment"] = df["review"].apply(analyze_with_textblob)

                st.write("Sentiment Analysis Results:")
                st.dataframe(df)

                # Download button for results
                def convert_df_to_excel(dataframe):
                    return dataframe.to_excel(index=False, engine='openpyxl')

                st.download_button(
                    label="Download Results as Excel",
                    data=convert_df_to_excel(df),
                    file_name="sentiment_analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as e:
            st.error(f"Error processing the file: {e}")
    else:
        st.info("Please upload an Excel file to begin.")

# Footer
st.write("---")
st.caption("Powered by Hugging Face Transformers, VADER, TextBlob, and Streamlit.")
