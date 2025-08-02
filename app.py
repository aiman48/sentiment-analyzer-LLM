from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def map_to_emoji(label: str, score: float):
    # Normalize score to float between 0 and 1 (already is)
    # If score is near 0.5, treat as neutral
    if abs(score - 0.5) < 0.1:
        return "😐", "NEUTRAL"
    if label.upper() == "POSITIVE":
        return "🙂", "POSITIVE"
    elif label.upper() == "NEGATIVE":
        return "🙁", "NEGATIVE"
    else:
        return "😐", label.upper()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text', '').strip()
    if not text:
        return render_template('result.html', error="Please provide some text.", text="", label="", score=0.0, emoji="😐")
    try:
        result = sentiment_pipeline(text)[0]
        label_raw = result['label']
        score_raw = result['score']  # between 0 and 1
        emoji, label = map_to_emoji(label_raw, score_raw)
        score = round(score_raw, 4)
        return render_template('result.html', text=text, label=label, score=score, emoji=emoji)
    except Exception as e:
        return render_template('result.html', error=str(e), text=text, label="", score=0.0, emoji="😐")

if __name__ == '__main__':
    app.run(debug=True)
