from flask import Flask, request, jsonify, render_template
from recommender import search_by_name, search_by_features

app = Flask(__name__)
app.secret_key = "spotify-recommender-secret-key"


# ── Home route ────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


# ── Search by name route ──────────────────────────────────────────
@app.route('/recommend/name', methods=['POST'])
def recommend_by_name():
    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({'error': 'No song name provided'}), 400

    result = search_by_name(query)
    return jsonify(result)


# ── Search by features route ──────────────────────────────────────
@app.route('/recommend/features', methods=['POST'])
def recommend_by_features():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No features provided'}), 400

    # Extract only valid feature keys from request
    valid_features = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]

    kwargs = {
        k: float(v)
        for k, v in data.items()
        if k in valid_features
    }

    result = search_by_features(**kwargs)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)