from sklearn import naive_bayes
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, X, y):
        self.model = naive_bayes.MultinomialNB()
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