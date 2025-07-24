# Movie Finder App

An interactive Gradio web application that allows users to:
1. Enter a keyword to search movie titles from a dataset
2. Select a matching title from a dynamic dropdown
3. View a list of similar movies in a tidy, scrollable DataFrame

Built with **Gradio**, **pandas**, and a **Hybrid Movie Recommendation Engine**

the dual-model movie recommender built using:
- **KNN over count vectorized metadata**
- **Transformer-based semantic embedding model**

This system merges content-based filtering with semantic similarity via Hugging Face's feature extraction pipeline, offering high-quality, explainable movie recommendations.

---

## 📦 Features

- Keyword-based title search using pandas filtering
- Dynamic dropdown of matching titles
- Similar movie recommendations powered by dual-model movie recommender system

---
## ⚙️ Overview

The recommender combines:
1. **KNN Model**:
   - Trained on vectorized movie genres and overviews
   - Enriched with numeric features like vote average and popularity
   - Scaled and indexed via `sklearn.NearestNeighbors`
   ```python
      X_scaled = scaler.fit_transform(KNN_features_db)
      knn_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
      knn_model.fit(X_scaled)
      distances, indices = knn_model.kneighbors(X_scaled)

2. **Transformer Model**:
   - Sentence embeddings extracted from `all-MiniLM-L6-v2` via `transformers.pipeline("feature-extraction")`
   - Embeddings cached and reused across sessions
   - Cosine similarity used to compare semantic meaning
   ```python
      feature_extractor = pipeline("feature-extraction",
                             model="sentence-transformers/all-MiniLM-L6-v2",
                             device_map="auto")
      embeddings = series.progress_apply(lambda x: feature_extractor(x)[0][0])
      embeddings_movies = np.vstack(embeddings)
      m_embedding = np.array(embeddings[movie_index]).reshape(1, -1)
      similarity_scores = cosine_similarity(m_embedding, embeddings)

## 🧪 Feature Engineering Pipeline
Extensive feature extraction was performed to support dual-model recommendations:
- **Genres Vectorization**: Using `CountVectorizer` to encode genre tags as sparse binary vectors.
- **Overview Vectorization**: Applied `CountVectorizer` to movie overviews for content-based similarity.
- Combined into a unified representation:
  ```python
  KNN_features_db = pd.concat([
      vector_word_df_generes,
      vector_word_df_overview,
      clean_data[['vote_average', 'popularity']]
  ], axis=1)

## 🚀 Getting Started

###  Clone the repo
```bash
git clone https://github.com/your-username/movie-finder-gradio.git
cd movie-finder-gradio
---