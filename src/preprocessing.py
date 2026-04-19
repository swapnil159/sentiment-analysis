import string
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter

def load_data(path: str):
    df = pd.read_csv(path)
    df['text'] = [_clean_data(str(t)) for t in df['text']]

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    return df

def create_vocab(corpus: list, ngrams: int=1, max_features: int=2000) -> dict:
    counter = Counter()
    for text in corpus:
        tokens = _generate_ngrams(text.split(), ngrams)
        counter.update(tokens)

    top_features = counter.most_common(max_features)
    
    vocab = {token: idx for idx, (token, _) in enumerate(top_features)}
    return vocab



def _clean_data(text: str) -> str:
    text = text.lower()

    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def _generate_ngrams(words: list, ngrams: int=1) -> list:
    if(ngrams == 1):
        return words
    
    return [" ".join(words[i:i+ngrams]) for i in range(len(words) - ngrams + 1)]
