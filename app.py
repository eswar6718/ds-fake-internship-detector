from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load saved models
tfidf = joblib.load("data/tfidf.pkl")
model_lr = joblib.load("data/log_model.pkl")
model_rf = joblib.load("data/rf_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    final_prob = None

    if request.method == "POST":

        description = request.form["description"]
        requirements = request.form["requirements"]

        # Combine text fields
        text = description + " " + requirements
        text_lower = text.lower()

        # TF-IDF transform
        X = tfidf.transform([text])

        # Predictions from both models
        p1 = model_lr.predict_proba(X)[0][1]
        p2 = model_rf.predict_proba(X)[0][1]

        # Ensemble probability
        prob = (p1 + p2) / 2

        # --------------------------
        # BOOST DS-RELATED KEYWORDS
        # --------------------------
        ds_keywords = [
            "python", "sql", "pandas", "numpy", "machine learning",
            "scikit", "matplotlib", "tensorflow", "power bi",
            "tableau", "deep learning", "github"
        ]

        keyword_hits = sum(1 for k in ds_keywords if k in text_lower)

        # Strong DS signal → boost probability
        if keyword_hits >= 3:
            prob += 0.40   # strong boost for genuine DS postings
        elif keyword_hits >= 1:
            prob += 0.20   # mild boost

        # Keep probability within limits
        prob = min(prob, 1.0)

        final_prob = round(prob * 100, 2)

        # --------------------------
        # CUSTOM THRESHOLD RULE
        # --------------------------
        if prob < 0.30:
            prediction = "FAKE"
        else:
            prediction = "LEGIT"

    return render_template("index.html",
                           prediction=prediction,
                           probability=final_prob)

if __name__ == "__main__":
    app.run(debug=True)
