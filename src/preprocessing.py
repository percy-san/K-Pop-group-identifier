import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


class DataPreprocessor:
    """Class to preprocess data for machine learning."""

    def __init__(self, input_path='data\\raw\\raw_data.csv',
                 output_dir='data\\processed', features=None):
        """Initialize with input path and feature list."""
        self.input_path = input_path  # Path to raw data CSV
        self.output_dir = output_dir  # Directory for processed data
        self.features = features or ['danceability', 'energy', 'loudness', 'speechiness',
                                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        self.scaler = StandardScaler()  # For feature scaling
        self.label_encoder = LabelEncoder()  # For encoding artist labels
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Load raw data from CSV."""
        print(f"Current working directory: {os.getcwd()}")  # Debug: Show script location
        print(f"Attempting to load: {os.path.abspath(self.input_path)}")  # Debug: Show full path
        try:
            df = pd.read_csv(self.input_path)  # Load the CSV
            print(f"Loaded {len(df)} tracks")  # Confirm row count
            return df
        except FileNotFoundError as e:
            print(f"Error: File {self.input_path} not found. Details: {e}")
            exit(1)

    def preprocess(self):
        """Preprocess data: select features, encode labels, split, and scale."""
        df = self.load_data()  # Start with raw data

        # Select features and target
        X = df[self.features]  # Extract feature columns
        y = df['artist']  # Target column (artist)

        print("Handling missing values...")  # Clean data
        X = X.dropna()  # Drop rows with missing values
        y = y[X.index]  # Align target with cleaned features

        print("Encoding labels...")  # Convert artists to numbers
        y_encoded = self.label_encoder.fit_transform(y)  # e.g., TWICE=0, BLACKPINK=1

        print("Splitting data...")  # Prepare for training
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42  # 80% train, 20% test
        )

        print("Scaling features...")  # Normalize features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)  # Fit and transform train
        self.X_test_scaled = self.scaler.transform(self.X_test)  # Transform test (no refit)

    def save_data(self):
        """Save preprocessed data and models."""
        os.makedirs(self.output_dir, exist_ok=True)  # Create output directory if needed
        print("Saving preprocessed data...")
        pd.DataFrame(self.X_train_scaled, columns=self.features).to_csv(
            os.path.join(self.output_dir, 'X_train.csv'), index=False
        )  # Save scaled training features
        pd.DataFrame(self.X_test_scaled, columns=self.features).to_csv(
            os.path.join(self.output_dir, 'X_test.csv'), index=False
        )  # Save scaled test features
        pd.DataFrame(self.y_train, columns=['artist']).to_csv(
            os.path.join(self.output_dir, 'y_train.csv'), index=False
        )  # Save training labels
        pd.DataFrame(self.y_test, columns=['artist']).to_csv(
            os.path.join(self.output_dir, 'y_test.csv'), index=False
        )  # Save test labels
        joblib.dump(self.scaler, 'models\\scaler.pkl')  # Save scaler model
        joblib.dump(self.label_encoder, 'models\\label_encoder.pkl')  # Save label encoder
        print("Preprocessing complete. Data saved to data\\processed\\")

    def run(self):
        """Run the full preprocessing pipeline."""
        self.preprocess()  # Execute preprocessing
        self.save_data()  # Save results


if __name__ == "__main__":
    preprocessor = DataPreprocessor()  # Instantiate the class
    preprocessor.run()  # Run the pipeline