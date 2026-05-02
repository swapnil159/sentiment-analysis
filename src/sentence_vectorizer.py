import numpy as np

class SentenceVectorizer:

    def __init__(self, embedding_model, vocab: dict):
        """
        embedding_model: instance of Word2VecLoader
        """
        self.embedding_model = embedding_model
        self.vector_size = embedding_model.vector_size()
        self.vocab = vocab

    def transform(self, texts: list[str], tfidf: list) -> list:
        """
        texts: list of raw sentences
        tfidf: TF-IDF matrix (same order)
        returns: vectors of size (n_samples, vector_size)
        """
        vectors = []
        for text, tfidf_vector in zip(texts, tfidf):
            sentence_vector = self._vectorize(text, tfidf_vector)
            vectors.append(sentence_vector)
        
        oov_rate = self._compute_oov_rate(texts)
        print("OOV Rate:", oov_rate)

        return vectors

    def _vectorize(self, text: str, tfidf_vector: list) -> list:
        word_vectors = []
        weights = []
        for word in text.split():
            if word not in self.vocab:
                continue
            idx = self.vocab[word]
            weight = tfidf_vector[idx]
            vec = self.embedding_model.get_vector(word)
            if vec is not None and weight > 0:
                word_vectors.append(vec * weight)
                weights.append(weight)
        
        # Handle case: no valid words
        if not word_vectors:
            return np.zeros(self.vector_size).tolist()

        # Average vectors
        return np.sum(word_vectors, axis = 0) / np.sum(weights)
    
    def _compute_oov_rate(self, texts: list):
        total = 0
        missing = 0

        for text in texts:
            words = text.split()

            for word in words:
                total += 1
                if self.embedding_model.get_vector(word) is None:
                    missing += 1

        return missing / total if total > 0 else 0

