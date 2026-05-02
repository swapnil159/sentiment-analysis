# 🎬 Sentiment Analysis: From Classical NLP to Word Embeddings

## 📌 Overview

This project builds a **Sentiment Analysis system** to classify movie reviews as **Positive** or **Negative**, while progressively exploring:

### 🔹 Classical NLP
- Bag of Words (BoW)
- TF-IDF
- N-grams (unigram vs bigram)

### 🔹 Machine Learning Models
- Naive Bayes
- Logistic Regression
- Linear SVM

### 🔹 Word Embeddings
- Word2Vec (attempted)
- GloVe (100d, 300d)
- Sentence representations via:
  - Averaging
  - TF-IDF weighted averaging

---

## 🚀 Features

- Custom implementation of:
  - Bag of Words (BoW)
  - TF-IDF
- Support for **n-grams (unigram, bigram)**
- Multiple models:
  - Naive Bayes
  - Logistic Regression
  - Linear SVM
- Word embedding pipeline:
  - Pretrained embeddings (GloVe)
  - Sentence vectorization
- Evaluation using:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
- CLI-based experimentation

---

## 🧠 Approach

### 1. Preprocessing
- Lowercasing and punctuation removal

### 2. Feature Engineering

#### Classical
- BoW and TF-IDF
- Vocabulary size: 2000
- Unigram vs Bigram experiments

#### Embeddings
- Pretrained GloVe embeddings
- Sentence representation:
  - Mean pooling
  - TF-IDF weighted pooling

### 3. Modeling
- Naive Bayes
- Logistic Regression
- Linear SVM

### 4. Evaluation
- Quantitative metrics + confusion matrix

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
│   ├── embeddings.py
│   ├── vectorizer.py
│   ├── model.py
│   └── main.py
│
├── requirements.txt
└── README.md
---

## ▶️ How to Run

```bash
git clone <your-repo-link>
cd sentiment-analysis
pip install -r requirements.txt
```

### Classical NLP:

```bash
python src/main.py --features bow --ngram 1
python src/main.py --features tfidf --ngram 1
```

### Embeddings:

```bash
python src/main.py --features embeddings
```

---

## 📊 Results

Evaluation performed on **10,000 test samples**

---

### 🔹 Classical Models

| Model                | Features         | Accuracy  |
|---------------------|------------------|-----------|
| Naive Bayes         | TF-IDF (1-gram)  | ~84.7%    |
| Logistic Regression | TF-IDF (1-gram)  | 87.2%     |
| Linear SVM          | TF-IDF (1-gram)  | **88.3%** |

---

### 🔹 Word Embeddings

| Method                        | Accuracy |
|------------------------------|----------|
| Word2Vec (avg)               | ~86%     |
| GloVe 100d (TF-IDF weighted) | ~76%     |
| GloVe 300d (TF-IDF weighted) | ~80%     |

---

## 📈 Key Insights

### 1. TF-IDF Remains a Strong Baseline

- Outperformed embedding-based methods in this task  
- Captures **task-specific word importance effectively**

---

### 2. Embeddings Introduce Semantics

- Capture similarity:
  - *good ≈ great*
  - *bad ≈ terrible*
- Provide **dense representations**

---

### 3. Aggregation is a Bottleneck

- Simple averaging loses:
  - word importance  
  - word interactions  
  - contextual meaning  
- Even TF-IDF weighting cannot fully recover this  

---

### 4. Domain Matters for Embeddings

- GloVe (trained on Wikipedia) underperformed on movie reviews  
- Performance depends on:
  - embedding quality  
  - domain alignment  

---

### 5. Sparse vs Dense Tradeoff

| TF-IDF | Embeddings |
|--------|------------|
| Sparse | Dense |
| High dimensional | Low dimensional |
| No semantics | Semantic similarity |
| Strong performance | Needs better modeling |

---

### 6. Why Classical Models Still Win

- Sentiment is often **keyword-driven**  
- Linear models + TF-IDF:
  - preserve strong signals  
  - avoid information dilution  

---

## ⚠️ Limitations

- Fixed vocabulary size (2000)  
- Embedding averaging loses structure  
- No handling of:
  - word order  
  - negation  
  - compositional meaning  

---

## 🔮 Future Work

- Neural networks on embeddings  
- Sequence models (LSTM)  
- Transformer models (BERT)  
- Better sentence representations  
- Domain-specific embeddings  

---

## 🎯 Conclusion

- Best model: **Linear SVM with TF-IDF**  
- Embeddings introduce semantic understanding but require:
  - better aggregation  
  - more advanced models  

- Demonstrates:
  - importance of representation  
  - limits of simple averaging  
  - need for deep learning in modern NLP  

---

## 👤 Author

Swapnil Mahajan
