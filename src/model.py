import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


class ModelTrainer:
    """Class to train and evaluate machine learning models."""

    def __init__(self, data_dir='data\\processed', model_dir='models'):
        """Initialize with data and model paths."""
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
        }
        self.best_model = None
        self.best_accuracy = 0
        self.best_model_name = ""

    def load_data(self):
        """Load preprocessed data."""
        print(f"Current working directory: {os.getcwd()}")
        print(f"Attempting to load X_train from: {os.path.abspath(os.path.join(self.data_dir, 'X_train.csv'))}")
        try:
            self.X_train = pd.read_csv(os.path.join(self.data_dir, 'X_train.csv'))
            self.X_test = pd.read_csv(os.path.join(self.data_dir, 'X_test.csv'))
            self.y_train = pd.read_csv(os.path.join(self.data_dir, 'y_train.csv')).values.ravel()
            self.y_test = pd.read_csv(os.path.join(self.data_dir, 'y_test.csv')).values.ravel()
        except FileNotFoundError as e:
            print(f"Error: Preprocessed data not found in {self.data_dir}. Details: {e}")
            exit(1)

    def train_and_evaluate(self):
        """Train and evaluate all models."""
        print("Training models...")
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.2f}")
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(self.y_test, y_pred))
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name

        if self.best_model_name == 'Random Forest':
            importances = self.best_model.feature_importances_
            feature_names = self.X_train.columns
            print("\nFeature Importance:")
            for feat, imp in zip(feature_names, importances):
                print(f"{feat}: {imp:.4f}")

    def save_model(self):
        """Save the best model."""
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.best_model, os.path.join(self.model_dir, 'best_model.pkl'))
        print(f"\nBest model saved: {self.best_model_name} with accuracy {self.best_accuracy:.2f}")

    def run(self):
        """Run the full training pipeline."""
        self.load_data()
        self.train_and_evaluate()
        self.save_model()


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()