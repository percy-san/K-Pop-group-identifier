# K-pop Artist Classifier

A machine learning project that predicts whether a K-pop song is by TWICE, BLACKPINK, or Dreamcatcher based on Spotify track features. The project includes a Gradio interface for predictions with audio playback of track previews.

## Features
- Predicts the artist of a K-pop song using synthetic audio features.
- Supports up to three track URLs for batch predictions.
- Includes a dropdown to hint at the artist for improved accuracy.
- Displays probability charts and prediction history.
- Plays 30-second audio previews (if available) for predicted tracks.

## Requirements
- Python 3.8 or higher
- Required packages: `spotipy`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `gradio`, `python-dotenv`, `joblib`, `requests`

## Installation

1. **Clone the Repository**  
   If this is in a Git repository, clone it:
   ```bash
   git clone https://github.com/yourusername/kpop-classifier.git
   cd kpop-classifier
   ```
   Otherwise, copy the folder to your desired location.

2. **Set Up a Virtual Environment**  
   ```cmd
   py -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**  
   ```cmd
   py -m pip install -r requirements.txt
   ```
   If `requirements.txt` doesn’t exist, create it with:
   ```
   spotipy
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
   gradio
   python-dotenv
   joblib
   requests
   ```

4. **Configure Spotify API Credentials**  
   - Create a `.env` file in the project root:
     ```plaintext
     CLIENT_ID=your_spotify_client_id
     CLIENT_SECRET=your_spotify_client_secret
     ```
   - Obtain these from the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) by creating an app.

5. **Run the Preprocessing and Model Training**  
   Ensure the raw data is collected (via `data_collection.py`) and preprocessed:
   ```cmd
   py src\preprocessing.py
   py src\model.py
   ```

## Usage

1. **Launch the Gradio Interface**  
   ```cmd
   py src\gradio_app.py
   ```
   - A browser will open to a public URL (e.g., `https://<random-hash>.gradio.app`).
   - Enter up to three Spotify track URLs and optionally select an artist hint.
   - Click "Predict" to see results, charts, audio previews, and history.

2. **Example URLs**  
   - TWICE: `https://open.spotify.com/track/7n2FZQsaLb7ZS8v05djTwP` ("FANCY")
   - BLACKPINK: `https://open.spotify.com/track/2CvOqDpQIMw69cCzWqr5yr` ("How You Like That")
   - Dreamcatcher: `https://open.spotify.com/track/2bDGej33ZtBmRzwTdfTDuM` ("SCREAM")

## Project Structure
```
KPOP group classifier/
├── data/
│   ├── raw/              # Raw data (e.g., kpop_data_with_synthetic_features.csv)
│   ├── processed/        # Preprocessed data (e.g., X_train.csv)
│   └── samples/          # Sample data (if any)
├── models/               # Saved models (e.g., best_model.pkl)
├── notebooks/            # Jupyter notebooks (e.g., data_exploration.ipynb)
├── src/                  # Source code
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── model.py
│   └── gradio_app.py
├── .env                  # Spotify API credentials
├── requirements.txt      # Dependencies
├── README.md             # This file
└── venv/                 # Virtual environment
```

## Contributing
Feel free to fork this repository, submit issues, or pull requests. Suggestions for improvements (e.g., more artists, better features) are welcome!

## Credits
- Developed by Buhlebethu Mkhonta  
- Uses Spotify API via `spotipy`  
- Built with Gradio for the interface  

## License
This project is open-source under the MIT License (see LICENSE file for details).
```
