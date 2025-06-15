from flask import Flask, request, render_template
import joblib
from sentence_transformers import SentenceTransformer
import re
import string
import pandas as pd
from datetime import datetime
import numpy as np

app = Flask(__name__)


sentiment_model = joblib.load("sentiment_xgb_model.pkl")
engagement_model = joblib.load("xgboost_engagement_model.pkl")
engagement_before_model = joblib.load("xgb_engagement_before_posting.pkl")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

label_map = {0: "Positive 😊", 1: "Negative 😠", 2: "Neutral 😐"}

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_features(text, timestamp):
    text = str(text)
    cleaned = clean_text(text)
    

    dt = pd.to_datetime(timestamp)
    
    
    return pd.DataFrame([{
        "TextLength": len(text),
        "WordCount": len(text.split()),
        "HashtagCount": len(re.findall(r"#\w+", text)),
        "MentionCount": len(re.findall(r"@\w+", text)),
        "EmojiCount": len(re.findall(r"[^\w\s,]", text)),
        "ExclamationCount": text.count("!"),
        "Hour": dt.hour,
        "IsWeekend": dt.weekday() >= 5,
    }])

def categorize_engagement(pred):
    if pred < 50:
        return "Low 🔹"
    elif pred < 200:
        return "Medium 🔸"
    elif pred < 500:
        return "High 🔶"
    else:
        return "Viral 🔥"


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/sentiment", methods=["GET", "POST"])
def sentiment():
    prediction = ""
    if request.method == "POST":
        user_input = request.form["text"]
        cleaned = clean_text(user_input)
        embedded = embedder.encode([cleaned])
        pred = sentiment_model.predict(embedded)[0]
        prediction = label_map[pred]
    return render_template("sentiment.html", prediction=prediction)

@app.route("/analyze")
def index():
    return render_template("analyze.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form["text"]
    likes = int(request.form["likes"])
    comments = int(request.form["comments"])
    shares = int(request.form["shares"])
    timestamp = request.form["timestamp"]

    cleaned = clean_text(text)
    embedded = embedder.encode([cleaned])
    sentiment = sentiment_model.predict(embedded)[0]
    sentiment_label = label_map[sentiment]

    engagement = likes + comments + shares


    engagement_cat = categorize_engagement(engagement)

    return render_template("analyze.html", prediction=sentiment_label,
                           engagement=engagement, engagement_cat=engagement_cat)

@app.route("/estimate", methods=["GET", "POST"])
def estimate():
    if request.method == "POST":
        text = request.form["text"]
        timestamp = request.form["timestamp"]
        
        features = extract_features(text, timestamp)
        prediction = engagement_before_model.predict(features)[0]
        category = categorize_engagement(prediction)

        return render_template("estimate.html", predicted_engagement=prediction, category=category)

    return render_template("estimate.html", predicted_engagement=None, category=None)

