
import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 AI Powered Fake News Detection")
st.write("Detect whether a news article is Real or Fake using Machine Learning.")

news = st.text_area("Enter News Article")

if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news.")
    else:
        news_vector = vectorizer.transform([news])
        prediction = model.predict(news_vector)

        if prediction[0] == 0:
            st.error("🔴 Fake News")
        else:
            st.success("🟢 Real News")
