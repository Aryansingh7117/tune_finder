# 🎵 TuneFinder — Music Recommendation System

A content-based music recommendation system that suggests similar songs based on audio features like energy, tempo, danceability, and valence.

---

## 🚀 Live Demo
> (https://tune-finder.onrender.com)

---

## 📌 Features

- 🔍 **Search by Song Name** — type any song name with typo tolerance powered by fuzzy matching
- 🎛️ **Search by Vibe** — use interactive sliders to describe your ideal sound and get matching songs
- 📊 **Popularity Blending** — recommendations are ranked by a blend of audio similarity (70%) and track popularity (30%)
- ⚡ **Fast** — KNN model is pre-trained and pickled, no retraining on every request

---

## 🧠 How It Works

1. **Data** — 1000 songs from a Spotify dataset with audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo)
2. **Preprocessing** — features are scaled using `StandardScaler`
3. **Model** — K-Nearest Neighbors (KNN) with cosine similarity finds the most similar songs in feature space
4. **Hyperparameter Tuning** — best `k` and `metric` selected using intra-list similarity score
5. **Scoring** — final recommendations are ranked by blending similarity score with track popularity
6. **API** — Flask serves two REST endpoints consumed by the frontend

---

## 🏗️ Project Structure

```
tune_finder/
│
├── data/
│   ├── spotifydataset.csv        # original dataset
│   └── songs_clean.csv           # cleaned dataset used at runtime
│
├── models/
│   ├── knn_model.pkl             # trained KNN model
│   ├── scaler.pkl                # fitted StandardScaler
│   └── feature_matrix.pkl        # scaled feature matrix
│
├── notebook/
│   └── analysis_music.ipynb      # EDA, training, evaluation
│
├── static/css/
│   └── style.css
│
├── templates/
│   └── index.html                # frontend
│
├── recommender.py                # recommendation logic
├── app.py                        # Flask app
└── requirements.txt
```

---

## 📡 API Endpoints

### `POST /recommend/name`
Search by song name with fuzzy matching.

**Request:**
```json
{ "query": "Shape of You" }
```

**Response:**
```json
{
  "input_song": "Shape of You — Ed Sheeran",
  "matched_score": 90.0,
  "recommendations": [
    {
      "display_name": "Song Name — Artist",
      "album_name": "Album",
      "track_popularity": 81,
      "final_score": 0.904
    }
  ]
}
```

---

### `POST /recommend/features`
Search by audio features. Any unspecified features default to dataset averages.

**Request:**
```json
{ "energy": 0.9, "tempo": 150, "danceability": 0.8 }
```

**Response:**
```json
{
  "input_features": { "danceability": 0.8, "energy": 0.9, ... },
  "recommendations": [ ... ]
}
```

---

## 📊 Model Evaluation

| Metric | Score |
|---|---|
| Best K | 20 |
| Best Metric | Cosine |
| Intra-list Similarity | 0.84 |
| Catalog Coverage | 61.6% |
| Hit Rate (same artist) | 33% |

---

## 🛠️ Tech Stack

| Layer | Tech |
|---|---|
| ML Model | Scikit-learn (KNN) |
| Data Processing | Pandas, NumPy |
| Fuzzy Search | RapidFuzz |
| Backend | Flask |
| Frontend | HTML, CSS, Vanilla JS |
| Deployment | Render |

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/Aryansingh7117/tune_finder.git
cd tune_finder

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---

## 📈 Future Improvements

- [ ] Add collaborative filtering using user listening history
- [ ] Expand dataset to 10k+ songs via Spotify API
- [ ] Add genre filtering
- [ ] Build a playlist generator

---

## 👤 Author

**Aryan Singh**
- GitHub: [@Aryansingh7117](https://github.com/Aryansingh7117)

---

> Built as a portfolio project to learn ML engineering end-to-end — from data processing and model training to REST API design and deployment.
