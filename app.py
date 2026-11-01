import os
import re
import joblib
import requests
import feedparser
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -------------------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Fake & Real News Detector",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Fake & Real News Detector")


# -------------------------------------------------------------------
# TEXT PREPROCESSING
# -------------------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"^[A-Z\s,/]+(?:\([A-Za-z\s]+\))?\s*-\s*", "", text)
    text = re.sub(r"\bReuters\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------------------------------------------
# MODEL LOADING & CACHE
# -------------------------------------------------------------------
MODEL_PATH = "model_cache.joblib"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    fake = pd.read_csv("Fake.csv", usecols=["title", "text"])
    real = pd.read_csv("True.csv", usecols=["title", "text"])

    fake["label"] = 0
    real["label"] = 1

    fake["clean_text"] = fake["text"].apply(clean_text)
    real["clean_text"] = real["text"].apply(clean_text)

    data = pd.concat([
        pd.DataFrame({"content": fake["title"].fillna("") + " " + fake["clean_text"], "label": 0}),
        pd.DataFrame({"content": real["title"].fillna("") + " " + real["clean_text"], "label": 1})
    ], ignore_index=True)

    data = data[data["content"].str.len() > 30].drop_duplicates(subset=["content"])

    X_train, X_test, y_train, y_test = train_test_split(
        data["content"], data["label"], test_size=0.20, random_state=42, stratify=data["label"]
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=15000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    acc = accuracy_score(y_test, predictions)

    artifacts = {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": acc
    }
    joblib.dump(artifacts, MODEL_PATH)
    return artifacts


try:
    model_data = load_or_train_model()
    model = model_data["model"]
    vectorizer = model_data["vectorizer"]
    acc = model_data.get("accuracy", 0.98)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# -------------------------------------------------------------------
# SIDEBAR (MINIMAL & CLEAN)
# -------------------------------------------------------------------
with st.sidebar:
    st.subheader("Model Status")
    st.metric("Accuracy", f"{acc * 100:.1f}%")
    if st.button("🔄 Retrain Model"):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        st.cache_resource.clear()
        st.rerun()


# -------------------------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------------------------
def predict_news(title, text):
    content = f"{title} {text}".strip()
    if len(content) < 20:
        return None

    cleaned_content = clean_text(content)
    X = vectorizer.transform([cleaned_content])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    fake_prob = float(probabilities[0]) * 100
    real_prob = float(probabilities[1]) * 100

    return {
        "prediction": prediction,
        "label": "🟢 REAL NEWS" if prediction == 1 else "🔴 FAKE NEWS",
        "confidence": max(fake_prob, real_prob),
        "fake_prob": fake_prob,
        "real_prob": real_prob
    }


# Recognized news outlets
VERIFIED_DOMAINS = [
    "indiatoday.in", "thehindu.com", "ndtv.com", "indianexpress.com",
    "timesofindia.indiatimes.com", "hindustantimes.com", "bbc.com", "bbc.co.uk",
    "reuters.com", "apnews.com", "aljazeera.com", "cnn.com", "bloomberg.com",
    "livemint.com", "business-standard.com", "firstpost.com", "deccanherald.com",
    "theprint.in", "scroll.in", "thequint.com", "aninews.in", "ptinews.com"
]


# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔗 Predict from URL", "🌐 Live News Feed"])


# --------------------------------------------------
# TAB 1: PREDICT FROM URL
# --------------------------------------------------
with tab1:
    url_input = st.text_input("News Article URL", placeholder="https://www.indiatoday.in/...")

    if st.button("🔍 Check URL", type="primary"):
        if not url_input.strip():
            st.warning("Please enter a URL.")
        else:
            try:
                with st.spinner("Analyzing article..."):
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                    }
                    response = requests.get(url_input.strip(), headers=headers, timeout=12)
                    response.raise_for_status()

                    soup = BeautifulSoup(response.text, "html.parser")

                    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
                        tag.decompose()

                    # Headline
                    title = ""
                    og_title = soup.find("meta", property="og:title")
                    if og_title and og_title.get("content"):
                        title = og_title["content"].strip()
                    elif soup.title:
                        title = soup.title.get_text(" ", strip=True)

                    # Body
                    body = ""
                    article_tag = soup.find("article")
                    if article_tag and len(article_tag.get_text(" ", strip=True)) >= 100:
                        body = article_tag.get_text(" ", strip=True)

                    if len(body) < 100:
                        itemprop_body = soup.find(attrs={"itemprop": "articleBody"})
                        if itemprop_body and len(itemprop_body.get_text(" ", strip=True)) >= 100:
                            body = itemprop_body.get_text(" ", strip=True)

                    if len(body) < 100:
                        paragraphs = [
                            p.get_text(" ", strip=True)
                            for p in soup.find_all("p")
                            if len(p.get_text(" ", strip=True)) > 25
                        ]
                        body = " ".join(paragraphs)

                    if len(body) < 50:
                        og_desc = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
                        if og_desc and og_desc.get("content"):
                            body = og_desc["content"].strip()

                    body = re.sub(r"\s+", " ", body).strip()

                if len(body) < 30 and len(title) < 20:
                    st.warning("Could not extract enough text from this URL.")
                else:
                    domain = (urlparse(url_input.strip()).hostname or "").lower()
                    if domain.startswith("www."):
                        domain = domain[4:]

                    is_verified = any(v in domain for v in VERIFIED_DOMAINS)
                    result = predict_news(title, body)

                    st.markdown("---")
                    st.write(f"**Headline:** {title}")

                    # Verdict
                    if is_verified and result["real_prob"] >= 40:
                        st.success(f"### 🟢 REAL NEWS (Verified Source: {domain})")
                    elif result["prediction"] == 1:
                        st.success(f"### 🟢 REAL NEWS ({result['confidence']:.1f}% confidence)")
                    else:
                        st.error(f"### 🔴 FAKE NEWS ({result['confidence']:.1f}% confidence)")

                    c1, c2 = st.columns(2)
                    c1.metric("Real News Probability", f"{result['real_prob']:.1f}%")
                    c2.metric("Fake News Probability", f"{result['fake_prob']:.1f}%")

                    with st.expander("View Extracted Text"):
                        st.write(body[:1200] + "..." if len(body) > 1200 else body)

            except Exception as e:
                st.error(f"Failed to fetch URL: {e}")


# --------------------------------------------------
# TAB 2: LIVE NEWS FEED
# --------------------------------------------------
with tab2:
    c_in, c_btn = st.columns([3, 1])
    with c_in:
        topic = st.text_input("Topic", value="India", label_visibility="collapsed")
    with c_btn:
        fetch_clicked = st.button("🔄 Fetch News")

    if fetch_clicked or "feed_articles" not in st.session_state:
        try:
            with st.spinner("Loading live news..."):
                query = f"{topic} when:24h"
                rss_url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
                resp = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
                resp.raise_for_status()
                feed = feedparser.parse(resp.content)

                articles = []
                for entry in feed.entries[:8]:
                    publisher = "Unknown"
                    if hasattr(entry, "source"):
                        publisher = entry.source.get("title", "Unknown")

                    summary_text = ""
                    if entry.get("summary"):
                        summary_soup = BeautifulSoup(entry["summary"], "html.parser")
                        summary_text = summary_soup.get_text(" ", strip=True)

                    articles.append({
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "publisher": publisher,
                        "summary": summary_text
                    })
                st.session_state["feed_articles"] = articles
        except Exception as e:
            st.error(f"Could not load news feed: {e}")

    feed_articles = st.session_state.get("feed_articles", [])
    if feed_articles:
        for i, art in enumerate(feed_articles):
            with st.expander(f"📰 {art['title']} ({art['publisher']})"):
                if art["summary"]:
                    st.write(art["summary"])
                if art["link"]:
                    st.markdown(f"[Read full story]({art['link']})")

                if st.button(f"⚡ Check News #{i+1}", key=f"feed_btn_{i}"):
                    res = predict_news(art["title"], art["summary"])
                    if res:
                        if res["prediction"] == 1:
                            st.success(f"**Result:** 🟢 REAL NEWS ({res['real_prob']:.1f}%)")
                        else:
                            st.error(f"**Result:** 🔴 FAKE NEWS ({res['fake_prob']:.1f}%)")
