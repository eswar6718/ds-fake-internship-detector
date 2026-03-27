import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print("Loading dataset...")

df = pd.read_csv("data/fake_job_postings.csv")

# Keep only needed columns
df = df[["description", "requirements", "fraudulent"]]

# Clean text
df["description"] = df["description"].fillna("")
df["requirements"] = df["requirements"].fillna("")

# Combine text fields
df["text"] = df["description"] + " " + df["requirements"]

X = df["text"]
y = df["fraudulent"]

# TF-IDF
tfidf = TfidfVectorizer(max_features=2000)
X_vec = tfidf.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Logistic Regression...")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
print("LR accuracy:", log_model.score(X_test, y_test))

print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200)
rf_model.fit(X_train, y_train)
print("RF accuracy:", rf_model.score(X_test, y_test))

# Save files
joblib.dump(tfidf, "data/tfidf.pkl")
joblib.dump(log_model, "data/log_model.pkl")
joblib.dump(rf_model, "data/rf_model.pkl")

print("Training complete!")
