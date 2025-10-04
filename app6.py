import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

strong_neg_words = {"idiot", "stupid", "dumb", "fool", "bitch", "asshole"}

def smart_clean_text(text, threshold):
    count = 0
    words = text.split()
    cleaned_words = []

    for w in words:
        word = re.sub(r'[^\w\s]', '', w).lower()
        sentiment = sia.polarity_scores(word)["compound"]

        if word in strong_neg_words or sentiment < threshold:
            cleaned_words.append("*" * len(word))
            count += 1
        else:
            cleaned_words.append(w)

    return " ".join(cleaned_words), count


st.title("NLP Offensive Word Filter")

sample_texts = {
    "Sample 1": "This restaurant is dumb. The pizza was cold and the service was stupid slow. The taste was bad and the place looked messy. Honestly, it felt like they didn’t care",
    "Sample 2": "This person is a complete idiot for how they handled the situation. It was a stupid move that completely ruined our chance of success. This whole dumb plan was orchestrated by a fool, and the final result is honestly the worst outcome imaginable. I find it deeply infuriating",
    "Sample 3": " cannot believe the audacity of the driver on the highway. He was incredibly rude and kept making reckless maneuvers, which was totally terrifying. That man is such an idiot for driving the way he did in the heavy traffic. Everything about that ten-minute drive was truly horrible and completely unacceptable, leaving me stressed and furious",
    "Sample 4": "This is amazing work. Great job everyone!"
}

st.sidebar.header("Select a Sample Paragraph")
selected_sample = st.sidebar.selectbox(
    "Choose a sample paragraph:",
    list(sample_texts.keys())
)

input_text = st.text_area("Enter your text:", value=sample_texts[selected_sample])

# Slider in sidebar
threshold = st.sidebar.slider(
    "Sentiment Threshold (-1 to 0):", 
    min_value=-1.0, 
    max_value=0.0, 
    value=-0.4, 
    step=0.05
)

st.write(f"Current Sentiment Threshold: {threshold}")

if st.button("Clean Text"):
    if input_text.strip():
        cleaned, count = smart_clean_text(input_text, threshold)
        st.header("Cleaned Text")
        st.write(cleaned)

        st.header("Analytics:")
        st.write("Total words processed ->", len(input_text.split()))
        st.write("Total offensive words ->", count)

    else:
        st.warning("Please enter some text!")
