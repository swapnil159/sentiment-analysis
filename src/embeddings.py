from gensim.models import KeyedVectors
import gensim.downloader as api
from typing import cast

class Word2VecLoader:

    def __init__(self):
        self.model: KeyedVectors | None = None

    def load_model(self):
        print("Loading Word2Vec model...")
        self.model = cast(KeyedVectors, api.load("glove-wiki-gigaword-300"))
        print("Model loaded!")

    def get_vector(self, word):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() before get_vector().")
        if self._has_word(word):
            return self.model[word]
        return None # keep for now (important for sentence averaging)
    
    def _has_word(self, word) -> bool:
        return word in self.model
    
    def similarity(self, word1, word2):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() before get_vector().")
        if self._has_word(word1) and self._has_word(word2):
            return self.model.similarity(word1, word2)
        else:
            raise ValueError(f"One or both words ('{word1}', '{word2}') not in vocabulary.")
        
    def vector_size(self):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() before get_vector().")
        return self.model.vector_size