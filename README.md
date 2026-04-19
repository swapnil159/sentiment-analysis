# 🎬 Sentiment Analysis using Bag of Words & TF-IDF

## 📌 Overview

This project implements a **Sentiment Analysis System** to classify movie reviews as **Positive** or **Negative**.

The goal is to explore how different text representation techniques impact model performance, specifically:

- Bag of Words (BoW)
- TF-IDF (Term Frequency–Inverse Document Frequency)
- Effect of n-grams (unigram vs bigram)

---

## 🚀 Features

- Text preprocessing (tokenization, cleaning)
- Custom implementation of:
  - Bag of Words (BoW)
  - TF-IDF
- Support for **n-grams (unigram, bigram)**
- SpamModel-based classification (Naive Bayes)
- Evaluation using:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
- CLI-based experimentation

---

## 🧠 How It Works

1. **Preprocessing**
   - Clean and tokenize text
   - Prepare corpus for vectorization

2. **Vectorization**
   - Convert text into numerical features using:
     - BoW
     - TF-IDF
   - Option to use:
     - Unigrams
     - Bigrams

3. **Model Training**
   - Train a Naive Bayes classifier

4. **Evaluation**
   - Measure performance using classification metrics

---

## 📂 Project Structure

sentiment-analysis/
│
├── data/
│ └── sample_sentiment.csv
│
├── src/
│ ├── preprocessing.py
│ ├── bow.py
│ ├── tfidf.py
│ ├── model.py
│ └── main.py
│
├── requirements.txt
└── README.md


---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone <your-repo-link>
cd sentiment-analysis
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run experiments

```bash
python src/main.py --features bow --ngram 1
python src/main.py --features bow --ngram 2

python src/main.py --features tfidf --ngram 1
python src/main.py --features tfidf --ngram 2
```

---

## 📊 Results

The model was evaluated on a test set of 10,000 reviews.

### 🔵 TF-IDF (Unigram)
```text
Accuracy: 84.76%

Confusion Matrix:

[[4158  803]
 [ 721 4318]]
```

### 🔵 TF-IDF (Bigram)
```text
Accuracy: 82.24%

Confusion Matrix:

[[3974  987]
 [ 789 4250]]
```

### 🟢 Bag of Words (Unigram)
```text
Accuracy: 82.35%

Confusion Matrix:

[[4136  825]
 [ 940 4099]]
```

### 🟢 Bag of Words (Bigram)
```text
Accuracy: 80.58%

Confusion Matrix:

[[3932 1029]
 [ 913 4126]]
```

## 📈 Model Comparison

| Features | N-gram | Accuracy   |
| -------- | ------ | ---------- |
| TF-IDF   | 1      | **84.76%** |
| TF-IDF   | 2      | 82.24%     |
| BoW      | 1      | 82.35%     |
| BoW      | 2      | 80.58%     |

---

## 🔍 Key Insights

- TF-IDF outperformed Bag of Words, highlighting the importance of term weighting
- Unigram models performed better than bigrams in this setup
- Increasing n-grams introduced:
    -  data sparsity
    - fragmented features
- Vocabulary size constraint (2000) limited bigram effectiveness
- TF-IDF unigram achieved the best balance of performance and efficiency

---

## 🧠 Observations

- Sentiment analysis is less keyword-driven and more context-based
- TF-IDF helps focus on meaningful words instead of frequent ones
- Higher n-grams require:
    - more data
    - larger vocabulary

---

## ⚠️ Limitations

- Uses dense vector representation (memory inefficient)
- Limited vocabulary size affects performance
- Cannot capture deep semantic meaning
- No handling of advanced language structures

---

## 🔍 Future Improvements

- Use sparse matrix representation
- Try Logistic Regression / SVM
- Increase vocabulary size with optimized storage
- Implement n-gram combinations (1 + 2)
- Explore word embeddings (Word2Vec, GloVe)

---

## 🎯 Conclusion

- Best performing model: TF-IDF with unigram features
- Simpler representations outperformed more complex ones under constraints
- Demonstrates the importance of:
    - feature engineering
    - data characteristics
    - model simplicity

---

## 📌 Author

Swapnil Mahajan