import pandas as pd
import re
import string
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv("data/projectML_augmented_final.csv", encoding="ISO-8859-1")

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['Cleaned_Text'] = df['Text'].apply(clean_text)

print("Generating sentence embeddings...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X = embedder.encode(df['Cleaned_Text'].tolist(), show_progress_bar=True)
y = df['Sentiment'].values

print("Balancing with SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

print("Training XGBoost classifier...")
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


joblib.dump(model, "sentiment_xgb_model.pkl")

custom_sentences = [
    "I love this product, it's fantastic!",
    "This is the worst service I’ve ever experienced.",
    "Not bad, but could be better.",
    "What a beautiful day!",
    "I'm not happy with the quality.",
    "Absolutely amazing work!",
    "Terrible experience, never coming back.",
    "Going to the store to pick up some groceries.",
    "Feeling exhausted and overwhelmed with everything today.",
    "Had a fantastic workout. Feeling strong and motivated!"
]

custom_cleaned = [clean_text(sentence) for sentence in custom_sentences]
custom_embeddings = embedder.encode(custom_cleaned)
custom_predictions = model.predict(custom_embeddings)

print("\nCustom Sentence Predictions:\n")

label_map = {0: "Positive", 1: "Negative", 2: "Neutral"}
for sent, pred in zip(custom_sentences, custom_predictions):
    print(f"'{sent}' → Predicted Sentiment: {label_map[pred]}")