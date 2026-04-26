from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, X, y):
        # self.model = naive_bayes.MultinomialNB()
        # self.model = LogisticRegression()
        self.model = LinearSVC()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        preds = self.model.predict(self.X_test)

        print("Accuracy:", accuracy_score(self.y_test, preds))
        print("\nClassification Report:\n", classification_report(self.y_test, preds))
        print("\nConfusion Matrix:\n", confusion_matrix(self.y_test, preds))
    
    def get_features_names(self, vocab: dict):
        index_to_words = {idx: word for word,idx in vocab.items()}
        weights = self.model.coef_[0]
        word_weights = [(weights[i], index_to_words[i]) for i in range(len(weights))]

        top_positive = sorted(word_weights, reverse=True)[:20]
        top_negative = sorted(word_weights)[:20]

        print("\nTop Positive Words:")
        for weight, word in top_positive:
            print(word + ": " + str(weight))

        print("\nTop Negative Words:")
        for weight, word in top_negative:
            print(word + ": " + str(weight))
