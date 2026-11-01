
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

