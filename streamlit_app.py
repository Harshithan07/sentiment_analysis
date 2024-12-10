import streamlit as st
import pandas as pd
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer as NLTKAnalyzer
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')


# Load Sentiment Analysis Models
@st.cache_resource
def load_transformers_pipeline():
    return pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

# @st.cache_resource
# def load_bert_pipeline():
#     return pipeline("sentiment-analysis", model="bert-base-uncased")

@st.cache_resource
def load_vader_analyzer():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_nltk_analyzer():
    return NLTKAnalyzer()

# @st.cache_resource
# def load_summarization_model():
#     return pipeline("text2text-generation", model="google/pegasus-xsum")


transformers_analyzer = load_transformers_pipeline()
# bert_analyzer = load_bert_pipeline()
vader_analyzer = load_vader_analyzer()
nltk_analyzer = load_nltk_analyzer()
# summarizer = load_summarization_model()


# Helper functions for VADER, TextBlob, NLTK, and BERT
def analyze_with_vader(text):
    scores = vader_analyzer.polarity_scores(text)
    if scores["compound"] > 0:
        return "POSITIVE"
    elif scores["compound"] <= 0:  # Treat neutral and negative compound scores as Negative
        return "NEGATIVE"

def analyze_with_textblob(text):
    polarity = TextBlob(text).sentiment.polarity
    return "POSITIVE" if polarity > 0 else "NEGATIVE"  # Treat neutral polarity as Negative

def analyze_with_nltk(text):
    scores = nltk_analyzer.polarity_scores(text)
    if scores["compound"] > 0:
        return "POSITIVE"
    elif scores["compound"] <= 0:  # Treat neutral and negative compound scores as Negative
        return "NEGATIVE"

def sentiment_to_emoji(sentiment):
    if sentiment == "POSITIVE":
        return "😊 Positive"
    elif sentiment == "NEGATIVE":
        return "😠 Negative"

def model_summary():
    summary = """
    ### Model Strengths and Weaknesses
    - **Transformers**: 
      - Captures nuanced sentiment and complex sentence structures.
      - Handles ambiguous or mixed sentiment effectively.
      - Excels with context-aware language understanding.
    - **VADER**: 
      - Rule-based and fast, suitable for explicit word polarity.
      - Struggles with complex sentences or sarcasm.
    - **TextBlob**: 
      - Simple polarity scoring, great for quick sentiment detection.
      - Limited in handling nuanced or contextual sentiments.
    - **NLTK Sentiment Analyzer**: 
      - Predefined lexicon-based analysis.
      - Good for basic sentiment detection but lacks adaptability to modern language patterns.
    """
    return summary

# def generate_product_overview(df):
#     # Extract insights
#     positive_count = (df["Transformers Sentiment"] == "😊 Positive").sum()
#     negative_count = (df["Transformers Sentiment"] == "😠 Negative").sum()
#     neutral_count = (df["Transformers Sentiment"] == "😐 Neutral").sum()
#     total_reviews = len(df)

#     top_reviews = df["review"].value_counts().head(5).to_dict()
#     top_positive = df[df["Transformers Sentiment"] == "😊 Positive"]["review"].head(3).tolist()
#     top_negative = df[df["Transformers Sentiment"] == "😠 Negative"]["review"].head(3).tolist()
#     positive_aspects = ", ".join(top_positive) if top_positive else "N/A"
#     negative_aspects = ", ".join(top_negative) if top_negative else "N/A"

#     summary_input = (
#         f"The product received mixed reviews based on user feedback. "
#         f"Out of a total of {total_reviews} reviews:\n"
#         f"- {positive_count} users were satisfied and left positive feedback.\n"
#         f"- {negative_count} users expressed dissatisfaction with the product.\n"
#         f"- {neutral_count} users had a neutral opinion.\n\n"
#         "Key themes and insights from the reviews include:\n"
#         f"1. Positive Highlights: Customers frequently praised the product for its {positive_aspects}.\n"
#         f"2. Negative Feedback: Common issues raised by users include {negative_aspects}.\n"
#         "3. Neutral Opinions: Some users mentioned that the product was neither exceptional nor poor.\n\n"
#         "Based on the trends:\n"
#         "- The overall sentiment indicates a mixed response leaning towards positive.\n"
#         "- Users recommend the product for daily use and occasional tasks.\n"
#         "- To improve satisfaction, consider addressing the issues related to delayed delivery and durability.\n\n"
#         "Here are a few notable review excerpts:\n"
#     )
#     for review, count in top_reviews.items():
#         summary_input += f"- \"{review}\" (mentioned {count} times)\n"

#     # Generate summary
#     output = summarizer(summary_input, max_length=150, min_length=50, do_sample=True)
#     return output[0]["generated_text"]

# Streamlit App Title
st.title("Sentiment Analysis App")
st.write("Perform sentiment analysis using Transformers, VADER, TextBlob, and NLTK.")

# Sidebar for input options
st.sidebar.title("Input Options")
input_option = st.sidebar.radio(
    "Choose an input method:",
    ("Type Text", "Upload Excel File")
)

if input_option == "Type Text":
    st.subheader("Type in your text for analysis")
    user_text = st.text_area("Enter your text here:", placeholder="Type something...")
    if st.button("Analyze Sentiment"):
        if user_text.strip():
            transformers_result = transformers_analyzer(user_text)[0]["label"]
            nltk_result = analyze_with_nltk(user_text)
            vader_result = analyze_with_vader(user_text)
            textblob_result = analyze_with_textblob(user_text)
            
            st.write(f"**Transformers Sentiment:** {sentiment_to_emoji(transformers_result)}")
            st.write(f"**VADER Sentiment:** {sentiment_to_emoji(vader_result)}")
            st.write(f"**TextBlob Sentiment:** {sentiment_to_emoji(textblob_result)}")
            st.write(f"**NLTK Sentiment:** {sentiment_to_emoji(nltk_result)}")

            st.write(model_summary())

        else:
            st.warning("Please enter some text to analyze.")
elif input_option == "Upload Excel File":
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
                df["Transformers Sentiment"] = df["review"].apply(lambda x: sentiment_to_emoji(transformers_analyzer(x)[0]["label"]))
                df["VADER Sentiment"] = df["review"].apply(lambda x: sentiment_to_emoji(analyze_with_vader(x)))
                df["TextBlob Sentiment"] = df["review"].apply(lambda x: sentiment_to_emoji(analyze_with_textblob(x)))
                df["NLTK Sentiment"] = df["review"].apply(lambda x: sentiment_to_emoji(analyze_with_nltk(x)))

                st.write("Sentiment Analysis Results:")
                st.dataframe(df)

                # Add model summary
                st.write(model_summary())

                # # Sentiment Distribution Charts
                # st.subheader("Sentiment Distribution Across Models")
                # for model in ["Transformers Sentiment"]:
                #     sentiment_counts = df[model].value_counts()
                #     st.write(f"**{model} Distribution**")
                #     fig, ax = plt.subplots()
                #     sentiment_counts.plot(kind="bar", ax=ax)
                #     ax.set_title(f"{model} Sentiment Distribution")
                #     ax.set_xlabel("Sentiment")
                #     ax.set_ylabel("Count")
                #     st.pyplot(fig)

                # Sentiment Trends Over Time
                # if 'timestamp' in df.columns:
                #     st.subheader("Sentiment Trends Over Time (Transformers)")
                #     df['timestamp'] = pd.to_datetime(df['timestamp'])  # Ensure timestamps are in datetime format
                #     sentiment_over_time = df.groupby([df['timestamp'].dt.date, 'Transformers Sentiment']).size().unstack(fill_value=0)

                #     # Plotting trends over time
                #     fig, ax = plt.subplots(figsize=(10, 6))
                #     sentiment_over_time.plot(ax=ax)
                #     ax.set_title("Sentiment Trends Over Time (Transformers)")
                #     ax.set_xlabel("Date")
                #     ax.set_ylabel("Count")
                #     st.pyplot(fig)
                # else:
                #     st.info("No 'timestamp' column found for sentiment trends over time.")

                # # Model Comparisons: Percentage Distribution
                # st.subheader("Model Comparisons: Percentage Distribution")
                # model_comparisons = {}
                # for model in ["Transformers Sentiment", "VADER Sentiment", "TextBlob Sentiment", "NLTK Sentiment"]:
                #     sentiment_counts = df[model].value_counts(normalize=True) * 100
                #     model_comparisons[model] = sentiment_counts

                # comparison_df = pd.DataFrame(model_comparisons).fillna(0)
                # st.write("Percentage Sentiment Distribution Across Models")
                # st.dataframe(comparison_df)

                # fig, ax = plt.subplots(figsize=(10, 6))
                # comparison_df.plot(kind='bar', ax=ax)
                # ax.set_title("Percentage Distribution Across Models")
                # ax.set_xlabel("Sentiment")
                # ax.set_ylabel("Percentage")
                # st.pyplot(fig)

                # # Product Overview and Recommendations
                # st.subheader("Product Overview and Recommendations")
                # overview = generate_product_overview(df)
                # st.write(overview)

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
st.caption("Powered by Hugging Face Transformers, VADER, TextBlob, NLTK, and Streamlit.")
