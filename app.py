import streamlit as st
import pickle

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main{
    background-color:#0E1117;
}
.big-title{
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:white;
}
.subtitle{
    text-align:center;
    color:#bbbbbb;
    font-size:18px;
    margin-bottom:25px;
}
.footer{
    text-align:center;
    color:gray;
    font-size:14px;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="big-title">📰 AI Powered Fake News Detection</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Detect whether a news article is <b>Real</b> or <b>Fake</b> using Machine Learning.</div>',
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Input
# -----------------------------
news = st.text_area(
    "📝 Enter News Article",
    height=220,
    placeholder="Paste your news article here..."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict", use_container_width=True):

    if news.strip() == "":
        st.warning("⚠ Please enter a news article.")

    else:

        news_vector = vectorizer.transform([news])

        prediction = model.predict(news_vector)[0]

        confidence = model.predict_proba(news_vector).max() * 100

        st.divider()

        if prediction == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")

        st.progress(int(confidence))

        st.info(f"Prediction Confidence: **{confidence:.2f}%**")

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.markdown(
"""
<div class="footer">
Made with ❤️ using Streamlit | Machine Learning Project
</div>
""",
unsafe_allow_html=True
)