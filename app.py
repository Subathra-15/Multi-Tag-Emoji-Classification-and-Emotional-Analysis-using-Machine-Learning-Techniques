"""
app.py
Flask GUI for sentiment analysis using VADER and TextBlob (Figure 14)
Run: python app.py
Open: http://127.0.0.1:5000
"""

from flask import Flask, request, render_template_string
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis</title>
    <style>
        body { font-family: Arial; max-width: 700px; margin: 60px auto; background: #f5f5f5; }
        h2   { color: #333; }
        input[type=text] { width: 70%; padding: 10px; font-size: 16px; border: 1px solid #ccc; border-radius: 4px; }
        button { padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
        .result { margin-top: 20px; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .positive { color: #4CAF50; font-weight: bold; }
        .negative { color: #f44336; font-weight: bold; }
        .neutral  { color: #FF9800; font-weight: bold; }
        h3 { color: #555; }
    </style>
</head>
<body>
    <h2>Sentiment Analysis Using VADER and TextBlob</h2>
    <form method="POST">
        <input type="text" name="text" placeholder="Enter a sentence..." value="{{ text or '' }}" required>
        <button type="submit">Get Prediction</button>
    </form>

    {% if vader_result %}
    <div class="result">
        <h3>Sentiment Analysis Using VADER</h3>
        <p>The sentiment is <span class="{{ vader_result.lower() }}">{{ vader_result }}</span></p>
        <small>Compound score: {{ vader_score }}</small>
    </div>
    <div class="result">
        <h3>Sentiment Analysis Using TextBlob</h3>
        <p>The sentiment is <span class="{{ tb_result.lower() }}">{{ tb_result }}</span></p>
        <small>Polarity: {{ tb_polarity }}</small>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    vader_result = tb_result = vader_score = tb_polarity = text = None
    if request.method == "POST":
        text = request.form.get("text", "")

        # VADER
        scores = analyzer.polarity_scores(text)
        vader_score = round(scores['compound'], 4)
        if vader_score >= 0.05:
            vader_result = "Positive"
        elif vader_score <= -0.05:
            vader_result = "Negative"
        else:
            vader_result = "Neutral"

        # TextBlob
        tb_polarity = round(TextBlob(text).sentiment.polarity, 4)
        if tb_polarity > 0:
            tb_result = "Positive"
        elif tb_polarity < 0:
            tb_result = "Negative"
        else:
            tb_result = "Neutral"

    return render_template_string(HTML,
        text=text,
        vader_result=vader_result,
        vader_score=vader_score,
        tb_result=tb_result,
        tb_polarity=tb_polarity
    )

if __name__ == "__main__":
    print("Starting Flask app...")
    print("Open your browser at: http://127.0.0.1:5000")
    app.run(debug=True)
