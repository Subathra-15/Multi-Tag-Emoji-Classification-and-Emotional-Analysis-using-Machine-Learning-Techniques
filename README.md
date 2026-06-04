# Multi-Tag Emoji Classification and Emotional Analysis

> Using B-PMI, VADER, and TextBlob on SemEval-2018 Task 1 dataset

---

## Project Overview

This project implements multi-label emotion classification on tweets using emojis as features. It proposes a **Balanced Pointwise Mutual Information (B-PMI)** method that assigns weighted emotion labels to emojis based on semantic similarity, extends pre-trained Word2Vec embeddings with emoji embeddings, and performs sentiment analysis using **VADER** and **TextBlob**. A Flask-based GUI is included for live sentiment prediction.

---

## Project Structure

```
emoji_emotion_project/
├── data/
│   ├── 2018-E-c-En-train.txt       # SemEval-2018 training set
│   ├── 2018-E-c-En-dev.txt         # SemEval-2018 dev set
│   └── 2018-E-c-En-test.txt        # SemEval-2018 test set
├── emotion_analysis.ipynb          # Main notebook (all analysis)
├── app.py                          # Flask web app (HTML/CSS inline)
├── requirements.txt                # Python dependencies
└── README.md
```

> **Note:** Word2Vec embeddings are trained on-the-fly using Gensim directly from the tweet corpus — no separate embeddings folder needed. The Flask GUI has all HTML and CSS self-contained inside `app.py`.

---

## Methods

| Method | Description |
|--------|-------------|
| **B-PMI** | Balanced Pointwise Mutual Information — assigns weighted emotion labels to emojis using co-occurrence statistics |
| **Word2Vec** | Gensim Word2Vec trained on tweet corpus (300 dimensions); emoji embeddings are built by summing emotion word vectors |
| **VADER** | Rule-based sentiment analysis; outputs positive/negative/neutral with compound score |
| **TextBlob** | Polarity-based sentiment classification in range [-1, 1] |
| **TF-IDF + Logistic Regression** | Baseline multi-label classifier per emotion |

---

## Emotion Classes (SemEval-2018 Task 1)

`anger` · `anticipation` · `disgust` · `fear` · `joy` · `love` · `optimism` · `pessimism` · `sadness` · `surprise` · `trust`

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Subathra-15/emoji_emotion_project.git
cd emoji_emotion_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK data (first run only)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Running the Project

### Notebook (Full Analysis)

Open and run all cells in order:

```bash
jupyter notebook emotion_analysis.ipynb
```

The notebook covers:
1. Data loading and label exploration
2. Text preprocessing and emoji/hashtag extraction
3. B-PMI computation and emoji-emotion lexicon construction
4. Word2Vec training and emoji embedding creation
5. Sentiment analysis with VADER and TextBlob
6. F1-score comparison (B-PMI vs VADER vs TextBlob)

### Flask GUI

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. Enter any sentence to get VADER and TextBlob sentiment predictions.

---

## Results

### Sentiment Distribution on Test Set

| Sentiment | TextBlob | VADER |
|-----------|----------|-------|
| Positive  | 38.2%    | 23.0% |
| Negative  | 31.2%    | 24.1% |
| Neutral   | 30.5%    | 53.0% |

VADER tends to classify more tweets as neutral; TextBlob skews slightly positive.

### Key Finding

B-PMI outperforms both VADER and TextBlob on supervised multi-label emotion F1-score, especially for emotions like **joy**, **optimism**, and **sadness**, where lexicon-based tools struggle with context.

---

## Dataset

**SemEval-2018 Task 1: Affect in Tweets (E-c)**
- ~7,000 English tweets manually annotated across 11 emotion categories
- Multi-label: a tweet can belong to multiple emotion classes
- Source: [SemEval-2018 Task 1](https://competitions.codalab.org/competitions/17751)

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
nltk
vaderSentiment
textblob
flask
scikit-learn
emoji
gensim
scipy
```

---

