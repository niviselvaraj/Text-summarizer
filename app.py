from flask import Flask, render_template, request
import nltk
import re
import heapq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download resources (first run only)
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# --- Summarizer Function ---
def summarize_text(text, n_sentences=3):
    # Clean text
    text = re.sub(r'\s+', ' ', text)

    # Tokenize sentences
    sentences = sent_tokenize(text)

    # Tokenize words & remove stopwords
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    word_frequencies = {}
    for word in words:
        if word.isalpha() and word not in stop_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    # Normalize word frequencies
    max_freq = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / max_freq

    # Score sentences
    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                if len(sent.split(" ")) < 30:  # ignore very long sentences
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

    # Pick top N sentences
    summary_sentences = heapq.nlargest(n_sentences, sentence_scores, key=sentence_scores.get)
    summary = " ".join(summary_sentences)

    return summary


# --- Flask Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        input_text = request.form["text"]
        if input_text.strip():
            summary = summarize_text(input_text, n_sentences=3)
    return render_template("index.html", summary=summary)


if __name__ == "__main__":
    app.run(debug=True)