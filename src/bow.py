from collections import Counter

def vectorize_bow(corpus: list[str], vocab: dict, ngrams: int=1) -> list:
    vectors = []

    for text in corpus:
        vector = [0] * len(vocab)

        tokens = _generate_ngrams(text.split(), ngrams)
        token_counts = Counter(tokens)

        for token, count in token_counts.items():
            if token in vocab:
                vector[vocab[token]] += count 
        
        vectors.append(vector)

    return vectors


def _generate_ngrams(words: list, ngrams: int=1) -> list:
    if(ngrams == 1):
        return words
    
    return [" ".join(words[i:i+ngrams]) for i in range(len(words) - ngrams + 1)]
