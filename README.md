# ⭐ Amazon Product Review Analyzer

An interactive web application built with Streamlit that analyzes product reviews using a 3-Model AI Engine. The app predicts the **Star Rating (1-5)**, determines the **Sentiment** (Positive, Neutral, or Negative), and provides a **Worth Buying?** recommendation based on the text.

## 🚀 Features

* **3-Model AI Engine**: Evaluates reviews using Logistic Regression, Naive Bayes, and Random Forest algorithms.
* **Confidence Scoring**: Automatically selects and displays the "Best Prediction" from the model with the highest confidence score.
* **Sentiment Analysis**: Maps predicted star ratings to easy-to-understand sentiment categories (🌟 POSITIVE, 😐 NEUTRAL, 📉 NEGATIVE).
* **Purchase Recommendation**: Gives a quick "YES", "MAYBE", or "NO" recommendation based on the review.
* **Interactive UI**: A clean, user-friendly sidebar and main interface built entirely in Streamlit.

## 🛠️ Tech Stack

* **Python**
* **Streamlit** (Web Interface)
* **Scikit-Learn** (Machine Learning & Vectorization)
* **Pandas** (Data Manipulation)
* **Joblib** (Model Saving/Loading)
* **Jupyter Notebook** (Model Training & Evaluation)

## 📁 Project Structure

* `app.py`: The main Streamlit web application script.
* `Model_training.ipynb`: Jupyter Notebook containing the data cleaning, TF-IDF vectorization, model training, and performance evaluation steps.
* `requirements.txt`: List of Python dependencies required to run the project.
* `*.pkl` files: Saved pre-trained models and vectorizers generated from the notebook (`tfidf_vectorizer.pkl`, `model_lr.pkl`, `model_nb.pkl`, `model_rf.pkl`).

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Satyavrat30/Ad-lab.git
cd Ad-lab
```

### 2. Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Train the Models (Prerequisite)

Before running the app, you must generate the model files.

1. Open `Model_training.ipynb`
2. Ensure you have the dataset available
3. Run all the cells in the notebook

This will:
- Clean the text
- Train the three models
- Evaluate performance
- Save the required `.pkl` files to your directory

### 4. Run the Web App

Once the `.pkl` files (`tfidf_vectorizer.pkl`, `model_lr.pkl`, `model_nb.pkl`, `model_rf.pkl`) are generated, start the Streamlit server:

```bash
streamlit run app.py
```

## 💡 Usage

1. Open the local Streamlit URL provided in your terminal (usually `http://localhost:8501`)
2. Type or paste a product review into the text box
3. Click **"Analyze Review"**
4. View:
   - Winning model's prediction
   - Confidence percentage
   - Star rating
   - Comparison of all model outputs
