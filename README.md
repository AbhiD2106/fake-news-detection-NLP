# 📰 Fake & Real News Detection Using NLP

**Academic Lab Project — MSc Semester 3 (NLP Activity)**  
**Max Marks:** 10 Marks  
**Technology Stack:** Python, Streamlit, Scikit-Learn, TF-IDF, Logistic Regression, BeautifulSoup  

---

## 🎯 Project Overview
This project detects whether a news article or headline is **Real News** or **Fake News** using Natural Language Processing (NLP) and Machine Learning.

### Key Objectives:
1. Load and merge the **Fake News** (`Fake.csv`) and **True News** (`True.csv`) datasets.
2. Clean text data and remove publisher dateline leakage so genuine real-world news is correctly identified.
3. Transform text into numerical features using **TF-IDF (Term Frequency - Inverse Document Frequency)** with unigrams and bigrams.
4. Train and evaluate a **Logistic Regression** classifier.
5. Provide a simple, interactive **Streamlit web interface** running on `localhost:8501`.

---

## 📊 Model Evaluation Results (Held-Out Test Data: 20%)

| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **Accuracy** | **97.93%** | Overall correct predictions |
| **Precision** | **97.35%** | Percentage of identified real news that is truly real |
| **Recall** | **98.87%** | Percentage of actual real news correctly identified |
| **F1-Score** | **98.10%** | Harmonic balance between Precision and Recall |
| **Dataset Size** | **39,112 articles** | Cleaned and deduplicated |

---

## 🚀 How to Run on Localhost

### 1. Open your terminal / PowerShell:
```powershell
cd e:\msc_sem3\activity2\NLP-Fake_News
```

### 2. Install dependencies (if not already installed):
```powershell
pip install -r requirements.txt
```

### 3. Run the Streamlit Application:
```powershell
streamlit run app.py
```

### 4. View in Browser:
The application will automatically start and open in your default browser at:
👉 **`http://localhost:8501`**

---

## 🖥️ Application Features

1. **🔗 Tab 1: Predict from News URL**
   - Paste a direct link to any news article (e.g. from BBC, The Hindu, NDTV, or any online blog).
   - Automatically extracts headline and article body using `BeautifulSoup`.
   - Runs the prediction immediately and shows Real vs Fake probability.

2. **🌐 Tab 2: Live News Feed (Google News RSS)**
   - Enter a country or topic (e.g., *India*, *Technology*, *Sports*).
   - Fetches the latest 10 breaking headlines dynamically from Google News RSS.
   - Click **"⚡ Test with Model"** on any story to classify it on the fly.
   - Click **"⚡ Test with Model"** to classify live news on the fly.

---

## 🎓 Viva Questions & Answers (For Full 10 Marks)

**Q1: What algorithm did you use and why?**  
*Answer:* We used **Logistic Regression** paired with **TF-IDF Vectorization**. For text classification with high-dimensional sparse feature spaces, Logistic Regression is computationally efficient, less prone to overfitting with regularization, and provides calibrated probability scores.

**Q2: What is TF-IDF?**  
*Answer:* TF-IDF stands for *Term Frequency - Inverse Document Frequency*. It evaluates how important a word is to an article relative to the entire dataset. Common words (like "the", "is") receive low weights, while distinctive journalistic words receive higher weights.

**Q3: How did you solve the Real News misclassification issue?**  
*Answer:* In the raw dataset, real articles started with `WASHINGTON (Reuters) -`. Standard models memorize `"reuters"` as the only indicator of truth. In our `clean_text()` preprocessing, we stripped publisher prefixes so the model learns genuine journalistic writing style (e.g., attribution verbs like *"said"*, neutral tone) rather than publisher brand names.
