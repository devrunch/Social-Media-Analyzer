import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv("feature_engineered_dataset.csv")

features = [
    "Likes", "Comments", "Shares",
    "TextLength", "WordCount", "HashtagCount", "MentionCount",
    "EmojiCount", "ExclamationCount", "Hour", "IsWeekend"
]
X = df[features]
y = df["Engagement"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae:.2f}")
print(f"Test R²: {r2:.4f}")

cv_mae = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()

print(f"CV MAE: {cv_mae:.2f}")
print(f"CV R²: {cv_r2:.4f}")

df["Predicted_Engagement"] = model.predict(X)

df.to_csv("predicted_engagement_dataset.csv", index=False, encoding="utf-8")
print("Saved: predicted_engagement_dataset.csv")

joblib.dump(model, "xgboost_engagement_model.pkl")
print("Saved model: xgboost_engagement_model.pkl")

