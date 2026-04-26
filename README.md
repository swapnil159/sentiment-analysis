# 🎬 Sentiment Analysis with Classical NLP (BoW, TF-IDF, NB, LR, SVM)

## 📌 Overview

This project builds a **Sentiment Analysis system** to classify movie reviews as **Positive** or **Negative**, while systematically exploring:

* Text representation techniques: **Bag of Words (BoW)** vs **TF-IDF**
* Effect of **n-grams (unigram vs bigram)**
* Model comparison:

  * **Naive Bayes**
  * **Logistic Regression**
  * **Linear SVM**

The focus is not just performance, but understanding:

> **How feature engineering and model choice impact results in NLP**

---

## 🚀 Features

* Custom implementation of:

  * Bag of Words (BoW)
  * TF-IDF
* Support for **n-grams (unigram, bigram)**
* Multiple models:

  * Naive Bayes
  * Logistic Regression
  * Linear SVM
* Evaluation using:

  * Accuracy
  * Precision, Recall, F1-score
  * Confusion Matrix
* CLI-based experimentation

---

## 🧠 Approach

### 1. Preprocessing

* Text cleaning and tokenization

### 2. Feature Engineering

* BoW and TF-IDF representations
* Controlled vocabulary size (2000)
* Experiments with:

  * Unigrams
  * Bigrams

### 3. Modeling

* Compared three model families:

  * Generative: Naive Bayes
  * Discriminative: Logistic Regression
  * Margin-based: Linear SVM

### 4. Evaluation

* Quantitative metrics + confusion matrix analysis

---

## 📂 Project Structure

```text
sentiment-analysis/
│
├── data/
│   └── sample_sentiment.csv
│
├── src/
│   ├── preprocessing.py
│   ├── bow.py
│   ├── tfidf.py
│   ├── model.py
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash
git clone <your-repo-link>
cd sentiment-analysis
pip install -r requirements.txt
```

### Run experiments:

```bash
python src/main.py --features bow --ngram 1
python src/main.py --features bow --ngram 2

python src/main.py --features tfidf --ngram 1
python src/main.py --features tfidf --ngram 2
```

---

## 📊 Results

Evaluation performed on **10,000 test samples**

### 🔹 Naive Bayes

| Features | N-gram | Accuracy |
| -------- | ------ | -------- |
| TF-IDF   | 1      | 84.76%   |
| TF-IDF   | 2      | 82.24%   |
| BoW      | 1      | 82.35%   |
| BoW      | 2      | 80.58%   |

---

### 🔹 Logistic Regression (TF-IDF, Unigram)

```
Accuracy: 87.24%

Confusion Matrix:
[[4265  696]
 [ 580 4459]]
```

---

### 🔹 Linear SVM (TF-IDF, Unigram)

```
Accuracy: 88.27%

Confusion Matrix:
[[4328  633]
 [ 540 4499]]
```

---

## 📈 Model Comparison

| Model               | Accuracy  |
| ------------------- | --------- |
| Naive Bayes         | ~84.7%    |
| Logistic Regression | 87.2%     |
| Linear SVM          | **88.3%** |

---

## 🔍 Key Insights

### 1. TF-IDF > Bag of Words

* Term weighting improves signal quality
* Reduces impact of frequent but uninformative words

---

### 2. Unigrams > Bigrams (under constraints)

* Bigrams increased sparsity
* Vocabulary cap (2000) limited useful bigram coverage
* Resulted in degraded performance

---

### 3. Model Choice Matters More Than Feature Complexity

* Switching from Naive Bayes → Logistic Regression gave ~3% gain
* Further improvement with Linear SVM

---

### 4. Why Linear Models Work Well for Text

* Text data is **high-dimensional and sparse**
* Linear models scale well and generalize effectively

---

### 5. Feature Importance Reveals Model Behavior

Top learned words:

**Positive:**

* great, excellent, amazing, wonderful

**Negative:**

* bad, worst, awful, boring

Also observed:

* Dataset-specific artifacts (e.g., “minutes”, “very”)

---

## ⚠️ Limitations

* Fixed vocabulary size (2000)
* Sparse representations ignore semantic similarity
* Cannot capture:

  * word order (effectively)
  * context
  * sarcasm / nuanced sentiment

---

## 🔮 Future Work

* Word embeddings (Word2Vec, GloVe)
* Deep learning models (LSTM)
* Transformer-based models (BERT)
* Hyperparameter tuning
* Larger vocabulary with sparse optimization

---

## 🎯 Conclusion

* Best model: **Linear SVM with TF-IDF (unigram)**
* Simpler features (unigrams) outperformed more complex ones under constraints
* Demonstrates:

  * Importance of feature engineering
  * Impact of model choice
  * Trade-offs in NLP systems

---

## 👤 Author

Swapnil Mahajan
