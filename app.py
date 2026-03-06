import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load("models/best_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_models()
    st.success("✅ Model and vectorizer loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model files not found. Please train the model first using `python fake_news_detector.py --train`")
    st.stop()

# Custom styling
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .stTextArea textarea { font-size: 16px; }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-weight: bold;
    }
    .real { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .fake { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📰 Fake News Detection System")
st.markdown("Enter a news headline or article text below, and the model will predict whether it's **Real** or **Fake**.")

# Input
text_input = st.text_area(
    "News text:",
    height=150,
    placeholder="Paste the news headline or article here..."
)

# Predict button
if st.button("🔍 Predict", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            # Vectorize and predict
            X = vectorizer.transform([text_input])
            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            confidence = proba[prediction]
            label = "Real" if prediction == 1 else "Fake"
            proba_fake, proba_real = proba[0], proba[1]

        # Display result
        box_class = "real" if prediction == 1 else "fake"
        st.markdown(
            f"<div class='prediction-box {box_class}'>"
            f"<h2>{label}</h2>"
            f"<p>Confidence: {confidence:.4f} ({confidence*100:.2f}%)</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Show probability bar (optional)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fake probability", f"{proba_fake:.4f}")
        with col2:
            st.metric("Real probability", f"{proba_real:.4f}")

        # Simple probability chart
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.barh(["Fake", "Real"], [proba_fake, proba_real], color=["#dc3545", "#28a745"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        st.pyplot(fig)

        # Optional: Show top influential words (if model is Logistic Regression)
        if hasattr(model, "coef_"):
            feature_names = vectorizer.get_feature_names_out()
            coef = model.coef_[0]
            indices = X.nonzero()[1]
            word_weights = [(feature_names[i], coef[i]) for i in indices]
            word_weights.sort(key=lambda x: x[1], reverse=True)

            st.markdown("#### 🧠 Top words influencing the prediction")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔴 Pushing toward FAKE**")
                for word, weight in word_weights[-5:]:  # most negative
                    st.write(f"- {word}: {weight:.4f}")
            with col2:
                st.markdown("**🟢 Pushing toward REAL**")
                for word, weight in word_weights[:5]:   # most positive
                    st.write(f"- {word}: {weight:.4f}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Model trained on [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)")