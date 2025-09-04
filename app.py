from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

# Flask app setup
app = Flask(__name__)
CORS(app)

# =========================
# Load Models and Data
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load dataset
songs_path = os.path.join(MODELS_DIR, "songs.csv")
data = pd.read_csv(songs_path)

# Load Content-based models
tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
cosine_sim = joblib.load(os.path.join(MODELS_DIR, "cosine_sim.pkl"))

# Load Collaborative model (if available)
svd_model = None
svd_path = os.path.join(MODELS_DIR, "svd_model.pkl")
if os.path.exists(svd_path):
    svd_model = joblib.load(svd_path)


# =========================
# Content-Based Recommendation
# =========================
@app.route('/recommend/content', methods=['POST'])
def recommend_content():
    req = request.get_json()
    song_name = req.get("song")

    if song_name not in data['Song'].values:
        return jsonify({"error": "Song not found in dataset"})

    # Find index of the song
    idx = data[data['Song'] == song_name].index[0]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Top 5 similar songs (skip first, as it is the same song)
    top_indices = [i[0] for i in sim_scores[1:6]]
    recommendations = data.iloc[top_indices][['Song', 'Artist', 'Genre']].to_dict(orient="records")

    return jsonify({"recommendations": recommendations})


# =========================
# Collaborative Filtering Recommendation
# =========================
@app.route('/recommend/collaborative', methods=['POST'])
def recommend_collaborative():
    if svd_model is None:
        return jsonify({"error": "Collaborative model not trained"})

    req = request.get_json()
    user_id = req.get("user")

    if "UserID" not in data.columns or "Song" not in data.columns:
        return jsonify({"error": "Dataset does not contain user ratings"})

    # Get all songs
    all_songs = data['Song'].unique()

    # Predict ratings for all songs
    predictions = []
    for song in all_songs:
        try:
            pred = svd_model.predict(user_id, song)
            predictions.append((song, pred.est))
        except:
            continue

    # Sort by estimated rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_songs = predictions[:5]

    recommendations = []
    for song, score in top_songs:
        row = data[data['Song'] == song].iloc[0]
        recommendations.append({
            "Song": song,
            "Artist": row['Artist'],
            "Genre": row['Genre'],
            "PredictedRating": score
        })

    return jsonify({"recommendations": recommendations})


# =========================
# Root Endpoint
# =========================
@app.route('/')
def home():
    return jsonify({"message": "Music Recommendation API is running"})


# =========================
# Run App
# =========================
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
