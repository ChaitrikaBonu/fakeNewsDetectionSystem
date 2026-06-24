import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page Configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Title
st.title("📰 Fake News Detection System")
st.write("Enter a news article or headline to check whether it is Real or Fake.")

# Text Input
news_text = st.text_area(
    "Enter News Text",
    height=250,
    placeholder="Paste your news article here..."
)

# Predict Button
if st.button("Check News"):

    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        # Transform text
        text_vector = vectorizer.transform([news_text])

        # Prediction
        prediction = model.predict(text_vector)

        # Probability
        probability = model.predict_proba(text_vector)

        if prediction[0] == 1:
            st.success("✅ Real News")
            st.write(
                f"Confidence: {max(probability[0])*100:.2f}%"
            )
        else:
            st.error("❌ Fake News")
            st.write(
                f"Confidence: {max(probability[0])*100:.2f}%"
            )

# Footer
st.markdown("---")
st.markdown(
    "Built using Machine Learning, TF-IDF Vectorization, and Logistic Regression."
)