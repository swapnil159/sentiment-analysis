from collections import Counter
import math

def calculate_tf_idf(corpus: list, vocab: dict, ngrams:int=1) -> list:
    tf = _calculate_tf(corpus, vocab, ngrams)
    idf = _calculate_idf(corpus, vocab, ngrams)

    tfidf = [
        [t * i for t, i in zip(doc, idf)] for doc in tf
    ]

    return tfidf


def _calculate_tf(corpus: list, vocab: dict, ngrams:int=1) -> list:
    vectors=[]

    for text in corpus:
        vector = [0.0] * len(vocab)
        tokens = _generate_ngrams(text.split(), ngrams)
        total_count = len(tokens)
        counter = Counter(tokens)
        counter.update(tokens)

        for token, count in counter.items():
            if token in vocab:
                vector[vocab[token]] = float(count) / total_count
        
        vectors.append(vector)
    
    return vectors


def _calculate_idf(corpus: list, vocab: dict, ngrams:int=1) -> list:
    vector = [0.0] * len(vocab)
    doc_count = len(corpus)
    token_count = {}

    for text in corpus:
        tokens = set(_generate_ngrams(text.split(), ngrams))
        for token in tokens:
            token_count[token] = token_count.get(token, 0) + 1

    for token, count in token_count.items():
        if token in vocab:
            vector[vocab[token]] = math.log(doc_count / count)
    
    return vector



def _generate_ngrams(words: list, ngrams: int=1) -> list:
    if(ngrams==1):
        return words
    
    return [" ".join(words[i:i+ngrams]) for i in range(len(words) - ngrams +1)]