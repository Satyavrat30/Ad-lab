import streamlit as st
import joblib

# 1. Load the newly saved models and vectorizer
try:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    rating_model = joblib.load('rating_model.pkl')
    buy_model = joblib.load('buy_model.pkl')
except FileNotFoundError:
    st.error("❌ Models not found! Please run 'train_models.py' first to generate the .pkl files.")
    st.stop()

# 2. Build the Streamlit App interface
st.set_page_config(page_title="Review Analyzer", page_icon="⭐")
st.title("Amazon Product Review Analyzer 🚀")
st.write("Analyze a review to find its Sentiment, Star Rating, and Buying Recommendation!")

# 3. Text input box for the user
user_review = st.text_area("Enter a product review here:")

if st.button("Analyze Review"):
    if user_review:
        # Transform text
        transformed_text = vectorizer.transform([user_review])
        
        # Predict the Exact Star Rating (Our "Boss" Model)
        pred_rating = int(rating_model.predict(transformed_text)[0])
        
        # Predict the Worth Buying Percentage
        probabilities = buy_model.predict_proba(transformed_text)[0]
        buy_percentage = round(probabilities[1] * 100, 2)
        
        st.markdown("---")
        
        # --- DISPLAY RESULTS ---
        
        # Feature 1: Sentiment Analysis (Controlled purely by the Rating)
        st.subheader("📝 Sentiment Analysis")
        if pred_rating >= 4:
            st.success("🌟 POSITIVE Review")
        elif pred_rating == 3:
            st.info("😐 NEUTRAL Review")
        else:
            st.error("📉 NEGATIVE Review")
            
        # Feature 2: Exact Rating
        st.subheader("⭐ Predicted Star Rating")
        st.write(f"Based on the text, the predicted rating is: **{pred_rating} out of 5 stars**")
        
        # Feature 3: Worth Buying (Smart Logic)
        st.subheader("🛍️ Is this product worth buying?")
        if pred_rating <= 2:
            st.error(f"**NO.** The review suggests avoiding this product. (Positive Match: {buy_percentage}%)")
        elif pred_rating >= 4 and buy_percentage >= 50:
            st.success(f"**YES!** Highly recommended based on this review. (Positive Match: {buy_percentage}%)")
        else:
            st.warning(f"**MAYBE.** The review contains mixed signals. (Positive Match: {buy_percentage}%)")
            
    else:
        st.warning("Please enter a review to analyze.")