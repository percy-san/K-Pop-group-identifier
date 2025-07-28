import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import pandas as pd
import joblib
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np


class GradioApp:
    """Class to run a Gradio interface for K-pop artist predictions."""

    def __init__(self, client_id=None, client_secret=None, model_path=r'models\best_model.pkl',
                 scaler_path=r'models\scaler.pkl', encoder_path=r'models\label_encoder.pkl'):
        """Initialize Spotify client and load models."""
        self.sp = self._init_spotify_client(client_id, client_secret)
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        self.features = ['danceability', 'energy', 'loudness', 'speechiness',
                         'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        self.genre_mapping = {
            'TWICE': {'danceability': 0.7, 'energy': 0.8, 'valence': 0.9, 'tempo': 128, 'loudness': -4.5,
                      'speechiness': 0.05, 'acousticness': 0.1, 'instrumentalness': 0.0, 'liveness': 0.15},
            'BLACKPINK': {'danceability': 0.8, 'energy': 0.9, 'valence': 0.7, 'tempo': 130, 'loudness': -4.0,
                          'speechiness': 0.08, 'acousticness': 0.05, 'instrumentalness': 0.0, 'liveness': 0.2},
            'Dreamcatcher': {'danceability': 0.6, 'energy': 0.9, 'valence': 0.5, 'tempo': 140, 'loudness': -3.5,
                             'speechiness': 0.1, 'acousticness': 0.03, 'instrumentalness': 0.01, 'liveness': 0.3}
        }
        np.random.seed(42)

    def _init_spotify_client(self, client_id, client_secret):
        """Initialize Spotify API client."""
        print(f"Current working directory: {os.getcwd()}")
        env_path = os.path.join(os.getcwd(), '.env')
        print(f"Looking for .env file at: {env_path}")
        if os.path.exists(env_path):
            print(".env file found!")
            load_dotenv(env_path)
        else:
            print(f"Error: .env file not found at {env_path}")

        client_id = client_id or os.getenv('CLIENT_ID')
        client_secret = client_secret or os.getenv('CLIENT_SECRET')
        print(f"CLIENT_ID: {'Set' if client_id else 'Not set'}")
        print(f"CLIENT_SECRET: {'Set' if client_secret else 'Not set'}")

        if not client_id or not client_secret:
            print("Error: CLIENT_ID or CLIENT_SECRET not found. Please ensure .env file is correctly set up.")
            exit(1)

        try:
            sp = spotipy.Spotify(
                auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))
            print("Successfully connected to Spotify API")
            return sp
        except Exception as e:
            print(f"Error connecting to Spotify API: {e}")
            exit(1)

    def generate_synthetic_features(self, track_data):
        """Generate synthetic features for a track."""
        artist_guess = None
        for artist in self.genre_mapping:
            if artist.lower() in [a['name'].lower() for a in track_data['artists']]:
                artist_guess = artist
                break
        if not artist_guess:
            artist_guess = 'TWICE'

        base_features = self.genre_mapping[artist_guess]
        popularity_factor = track_data['popularity'] / 100
        duration_factor = min(track_data['duration_ms'] / 240000, 1)

        features = [
            max(0, min(1, base_features['danceability'] + np.random.normal(0, 0.1))),
            max(0, min(1, base_features['energy'] + np.random.normal(0, 0.1))),
            np.random.normal(base_features['loudness'], 1),
            np.random.beta(2, 8),
            np.random.beta(2, 5),
            np.random.beta(1, 9),
            np.random.beta(1, 9),
            max(0, min(1, base_features['valence'] + np.random.normal(0, 0.1))),
            np.random.normal(base_features['tempo'], 10)
        ]
        return features

    def predict_artist(self, track_url):
        """Predict artist from a Spotify track URL."""
        try:
            track_id = track_url.split('/')[-1].split('?')[0]
            track = self.sp.track(track_id, market='US')
            if not track:
                return "Error: Could not fetch track data.", None
            input_features = self.generate_synthetic_features(track)
            scaled_features = self.scaler.transform([input_features])
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
            artist = self.label_encoder.inverse_transform([prediction])[0]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(self.label_encoder.classes_, probabilities, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_ylabel('Probability')
            ax.set_title(f'Predicted Artist: {artist} for "{track["name"]}"')
            plt.tight_layout()
            return f"Predicted Artist: {artist}", fig
        except Exception as e:
            return f"Error: Invalid URL or API issue: {str(e)}", None

    def launch(self, share=False):
        """Launch the Gradio interface."""
        iface = gr.Interface(
            fn=self.predict_artist,
            inputs=gr.Textbox(label="Spotify Track URL",
                              placeholder="e.g., https://open.spotify.com/track/7n2FZQsaLb7ZS8v05djTwP"),
            outputs=[gr.Textbox(label="Prediction"), gr.Plot(label="Probabilities")],
            title="K-pop Artist Classifier",
            description="Enter a Spotify track URL to predict if the song is by TWICE, BLACKPINK, or Dreamcatcher."
        )
        iface.launch(share=True)  # Changed to share=True


if __name__ == "__main__":
    load_dotenv()
    app = GradioApp(os.getenv('CLIENT_ID'), os.getenv('CLIENT_SECRET'))
    app.launch(share=True)