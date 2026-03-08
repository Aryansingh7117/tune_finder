import pickle
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

with open('models/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/feature_matrix.pkl', 'rb') as f:
    feature_matrix = pickle.load(f)

df = pd.read_csv('data/songs_clean.csv')
df['song_uid'] = df.index.astype(str)
song_index = pd.Series(df.index, index=df['song_uid'])

FEATURE_COLS = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

FEATURE_DEFAULTS = df[FEATURE_COLS].mean().to_dict()

def _run_knn(feature_vector, k=5):
    scaled = scaler.transform(pd.DataFrame([feature_vector], columns=FEATURE_COLS))
    distances, indices = knn_model.kneighbors(scaled, n_neighbors=k+1)

    rec_indices = indices[0]
    rec_distances = distances[0]

    recommendations = df.loc[rec_indices].copy()
    recommendations['similarity_score'] = 1 - rec_distances
    recommendations['final_score'] = (
        0.7 * recommendations['similarity_score'] +
        0.3 * recommendations['track_popularity'] / 100
    )

    return recommendations[
        ['display_name', 'album_name', 'track_popularity', 'final_score']
    ].sort_values('final_score', ascending=False).reset_index(drop=True).to_dict(orient='records')


def search_by_name(query, k=5):
    choices = df['display_name'].tolist()
    matches = process.extract(query, choices, scorer=fuzz.WRatio, limit=5)

    good_matches = [m for m in matches if m[1] >= 60]

    if not good_matches:
        return {'error': f"No songs found matching '{query}'"}

    # Take top match
    best = good_matches[0]
    matched_idx = best[2]

    # Get that song's audio features
    feature_vector = df.loc[matched_idx, FEATURE_COLS].values.tolist()

    # Run KNN
    recommendations = _run_knn(feature_vector, k)

    # Remove input song from results
    recommendations = [
        r for r in recommendations
        if r['display_name'] != df.loc[matched_idx, 'display_name']
    ][:k]

    return {
        'input_song': df.loc[matched_idx, 'display_name'],
        'matched_score': round(best[1], 2),
        'recommendations': recommendations
    }

def search_by_features(k=5, **kwargs):
    feature_vector = [
        kwargs.get(col, FEATURE_DEFAULTS[col])
        for col in FEATURE_COLS
    ]

    recommendations = _run_knn(feature_vector, k)

    return {
        'input_features': {
            col: kwargs.get(col, round(FEATURE_DEFAULTS[col], 3))
            for col in FEATURE_COLS
        },
        'recommendations': recommendations
    }