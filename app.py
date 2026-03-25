import streamlit as st
import joblib
import numpy as np

# 1. Page Config MUST be the first Streamlit command
st.set_page_config(page_title="Review Analyzer", page_icon="⭐", layout="wide")

# 2. Load the newly saved models and vectorizer
@st.cache_resource # This caches the models so they don't reload on every button click!
def load_models():
    try:
        vec = joblib.load('tfidf_vectorizer.pkl')
        lr = joblib.load('model_lr.pkl')
        nb = joblib.load('model_nb.pkl')
        rf = joblib.load('model_rf.pkl')
        return vec, lr, nb, rf
    except FileNotFoundError:
        return None, None, None, None

vectorizer, model_lr, model_nb, model_rf = load_models()

if vectorizer is None:
    st.error("❌ Models not found! Please run 'train_models.py' first to generate the .pkl files.")
    st.stop()

# --- SIDEBAR: Historical Accuracies ---
st.sidebar.title("📊 Model Accuracies")
st.sidebar.write("Based on our 50,000 row test data:")
st.sidebar.info("**Naive Bayes:** ~68.45%")
st.sidebar.success("**Random Forest:** ~66.10%")
st.sidebar.warning("**Logistic Regression:** ~61.20%")

# --- MAIN APP ---
st.title("Amazon Product Review Analyzer")
st.write("Analyze a review using our 3-Model AI Engine to find its Sentiment and Star Rating!")

# Text input box for the user
user_review = st.text_area("Enter a product review here:", height=150)

if st.button("Analyze Review", type="primary"):
    if user_review:
        # Transform text
        transformed_text = vectorizer.transform([user_review])
        
        # --- GET PREDICTIONS & CONFIDENCE SCORES ---
        # predict() gets the 1-5 star rating. 
        # predict_proba() gets the percentage of certainty for that rating.
        
        pred_lr = int(model_lr.predict(transformed_text)[0])
        conf_lr = max(model_lr.predict_proba(transformed_text)[0]) * 100
        
        pred_nb = int(model_nb.predict(transformed_text)[0])
        conf_nb = max(model_nb.predict_proba(transformed_text)[0]) * 100
        
        pred_rf = int(model_rf.predict(transformed_text)[0])
        conf_rf = max(model_rf.predict_proba(transformed_text)[0]) * 100
        
        # Group the results into a list of dictionaries for easy sorting
        results = [
            {"Model": "Logistic Regression", "Rating": pred_lr, "Confidence": conf_lr},
            {"Model": "Naive Bayes", "Rating": pred_nb, "Confidence": conf_nb},
            {"Model": "Random Forest", "Rating": pred_rf, "Confidence": conf_rf}
        ]
        
        # Find the model that is MOST confident for this specific text
        best_result = max(results, key=lambda x: x['Confidence'])
        
        st.markdown("---")
        
        # --- DISPLAY BEST RESULT ---
        st.header(f"Best Prediction (by {best_result['Model']})")
        st.write(f"This model was **{best_result['Confidence']:.2f}% confident** in this result.")
        
        # Logic based on the winning model's rating
        final_rating = best_result['Rating']
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.subheader("⭐ Star Rating")
            st.write(f"**{final_rating} out of 5 stars**")
            
        with col_b:
            st.subheader("📝 Sentiment")
            if final_rating >= 4:
                st.success("🌟 POSITIVE")
            elif final_rating == 3:
                st.info("😐 NEUTRAL")
            else:
                st.error("📉 NEGATIVE")
                
        with col_c:
            st.subheader("🛍️ Worth Buying?")
            if final_rating >= 4:
                st.success("**YES!** Recommended.")
            elif final_rating == 3:
                st.warning("**MAYBE.** Mixed signals.")
            else:
                st.error("**NO.** Avoid this.")

        st.markdown("---")
        
        # --- DISPLAY ALL MODEL RESULTS ---
        st.subheader("How the other models voted:")
        
        # Create 3 columns for a clean UI
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Logistic Regression**\n\nPredicted: {pred_lr} Stars\n\nConfidence: {conf_lr:.2f}%")
        with col2:
            st.info(f"**Naive Bayes**\n\nPredicted: {pred_nb} Stars\n\nConfidence: {conf_nb:.2f}%")
        with col3:
            st.info(f"**Random Forest**\n\nPredicted: {pred_rf} Stars\n\nConfidence: {conf_rf:.2f}%")

    else:
        st.warning("⚠️ Please enter a review to analyze.")