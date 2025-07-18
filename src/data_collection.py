import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import numpy as np
import time


class SpotifyDataCollector:
    """Class to collect Spotify track metadata and generate synthetic audio features."""

    def __init__(self, client_id=None, client_secret=None, artists=['TWICE', 'BLACKPINK', 'Dreamcatcher'],
                 num_tracks=50):
        """Initialize Spotify client and set parameters."""
        self.artists = artists
        self.num_tracks = num_tracks
        self.genre_mapping = {
            'TWICE': {'danceability': 0.7, 'energy': 0.8, 'valence': 0.9, 'tempo': 128, 'loudness': -4.5,
                      'speechiness': 0.05, 'acousticness': 0.1, 'instrumentalness': 0.0, 'liveness': 0.15},
            'BLACKPINK': {'danceability': 0.8, 'energy': 0.9, 'valence': 0.7, 'tempo': 130, 'loudness': -4.0,
                          'speechiness': 0.08, 'acousticness': 0.05, 'instrumentalness': 0.0, 'liveness': 0.2},
            'Dreamcatcher': {'danceability': 0.6, 'energy': 0.9, 'valence': 0.5, 'tempo': 140, 'loudness': -3.5,
                             'speechiness': 0.1, 'acousticness': 0.03, 'instrumentalness': 0.01, 'liveness': 0.3}
        }
        np.random.seed(42)  # For reproducible synthetic features
        self.sp = self._init_spotify_client(client_id, client_secret)

    def _init_spotify_client(self, client_id, client_secret):
        """Initialize Spotify API client."""
        # Debug: Print current working directory
        print(f"Current working directory: {os.getcwd()}")

        # Load .env file
        env_path = os.path.join(os.getcwd(), '.env')
        print(f"Looking for .env file at: {env_path}")
        if os.path.exists(env_path):
            print(f".env file found!")
            load_dotenv(env_path)
        else:
            print(f"Error: .env file not found at {env_path}")

        # Get credentials from environment variables if not provided
        client_id = client_id or os.getenv('CLIENT_ID')
        client_secret = client_secret or os.getenv('CLIENT_SECRET')

        # Debug: Check if credentials were loaded
        print(f"CLIENT_ID: {'Set' if client_id else 'Not set'}")
        print(f"CLIENT_SECRET: {'Set' if client_secret else 'Not set'}")

        if not client_id or not client_secret:
            print("Error: CLIENT_ID or CLIENT_SECRET not found. Please ensure .env file is correctly set up.")
            print("Example .env content:")
            print("CLIENT_ID=your_client_id")
            print("CLIENT_SECRET=your_client_secret")
            exit(1)

        try:
            sp = spotipy.Spotify(
                auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))
            print("Successfully connected to Spotify API")
            return sp
        except Exception as e:
            print(f"Error connecting to Spotify API: {e}")
            exit(1)

    def collect_track_metadata(self):
        """Collect track metadata from Spotify API."""
        data = []
        for artist in self.artists:
            print(f"Collecting tracks for {artist}...")
            try:
                results = self.sp.search(q=f'artist:{artist}', type='track', limit=self.num_tracks, market='US')
                tracks = results['tracks']['items']
                for track in tracks:
                    if artist in [a['name'] for a in track['artists']]:
                        data.append({
                            'track_name': track['name'],
                            'artist': artist,
                            'album': track['album']['name'],
                            'release_date': track['album']['release_date'],
                            'popularity': track['popularity'],
                            'duration_ms': track['duration_ms'],
                            'explicit': track['explicit'],
                            'track_id': track['id']
                        })
                time.sleep(1)  # Avoid rate limits
            except Exception as e:
                print(f"Error processing {artist}: {e}")
                continue
        return pd.DataFrame(data)

    def generate_synthetic_features(self, df):
        """Add synthetic audio features to the DataFrame."""
        if df.empty:
            print("No data to process. Please collect tracks first.")
            return df

        # Initialize feature columns
        for feature in ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'speechiness', 'acousticness',
                        'instrumentalness', 'liveness']:
            df[feature] = 0.0

        # Generate synthetic features for each track
        for i, row in df.iterrows():
            artist = row['artist']
            base_features = self.genre_mapping.get(artist, self.genre_mapping['TWICE'])
            popularity_factor = row['popularity'] / 100
            duration_factor = min(row['duration_ms'] / 240000, 1)

            df.loc[i, 'danceability'] = max(0, min(1, base_features['danceability'] + np.random.normal(0, 0.1)))
            df.loc[i, 'energy'] = max(0, min(1, base_features['energy'] + np.random.normal(0, 0.1)))
            df.loc[i, 'valence'] = max(0, min(1, base_features['valence'] + np.random.normal(0, 0.1)))
            df.loc[i, 'tempo'] = np.random.normal(base_features['tempo'], 10)
            df.loc[i, 'loudness'] = np.random.normal(base_features['loudness'], 1)
            df.loc[i, 'speechiness'] = np.random.beta(2, 8)
            df.loc[i, 'acousticness'] = np.random.beta(2, 5)
            df.loc[i, 'instrumentalness'] = np.random.beta(1, 9)
            df.loc[i, 'liveness'] = np.random.beta(1, 9)

        return df

    def save_data(self, df, output_path='data\\raw\\raw_data.csv'):
        """Save DataFrame to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path} with {len(df)} tracks")
        print("Sample data:")
        print(df[['track_name', 'artist', 'danceability', 'energy', 'valence']].head())

    def run(self):
        """Run the full data collection pipeline."""
        df = self.collect_track_metadata()
        if not df.empty:
            df = self.generate_synthetic_features(df)
            self.save_data(df)
        else:
            print("No data collected. Please check API credentials or try again later.")


if __name__ == "__main__":
    # Load environment variables explicitly for debugging
    load_dotenv()
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    collector = SpotifyDataCollector(client_id, client_secret)
    collector.run()