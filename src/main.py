import argparse
from preprocessing import load_data, create_vocab
from bow import vectorize_bow
from tfidf import calculate_tf_idf
from model import Model
from embeddings import Word2VecLoader
from sentence_vectorizer import SentenceVectorizer

def parse_args():
    args = argparse.ArgumentParser("Sentiment analyser")

    args.add_argument(
        "--features",
        type=str,
        default="bow",
        choices=["bow", "tfidf"],
        help="Feature type: BoW or TF-IDF",
    )
    args.add_argument(
        "--data_path",
        type=str,
        default="../data/sample_data.csv",
        help="Patth of the dataset",
    )
    args.add_argument(
        "--ngrams",
        type=int,
        default=1,
        choices=[1, 2],
        help="N-grams to consider",
    )

    return args.parse_args()


def main():
    args = parse_args()
    df = load_data(args.data_path)

    corpus = df['text'].tolist()
    labels = df['label'].tolist()

    vocab = create_vocab(corpus, args.ngrams)
    print("Vocab Size: ", len(vocab))

    w2v = Word2VecLoader()
    w2v.load_model()

    tfidf = calculate_tf_idf(corpus, vocab, args.ngrams)
    vectorizer = SentenceVectorizer(w2v, vocab)
    X = vectorizer.transform(corpus, tfidf)


    # if args.features == "bow":
    #     X = vectorize_bow(corpus, vocab, args.ngrams)
    # else:
    #     X = calculate_tf_idf(corpus, vocab, args.ngrams)
    
    model = Model(X, labels)
    model.train()
    model.evaluate()

    # model.get_features_names(vocab)



if __name__ == "__main__":
    main()