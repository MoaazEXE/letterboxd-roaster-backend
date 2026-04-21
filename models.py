import pickle
import pandas as pd

#load models
with open('models/svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)
with open('models/cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)
with open('models/movies_df.pkl', 'rb') as f:
    movies_df = pickle.load(f)

#persona mapping 
persona_map={ "casual":52960,
             "enthusiast":72315,
              "critic":80974} 

movies_df_lower = movies_df.copy()
movies_df_lower['title_lower'] = movies_df_lower['title'].str.lower()

#get_candidates(title) function
def get_candidates(title, n=20):
    title=title.lower()

    if title not in movies_df['title_lower'].values:
        raise ValueError("Movie not found")
    
    idx = movies_df_lower[movies_df_lower['title_lower'] == title].index[0]

    top_indices = cosine_sim['indices'][idx][:n]
    top_scores = cosine_sim['scores'][idx][:n]

    results=[]
    
    for movie_idx, score in zip(top_indices, top_scores):
        movie = movies_df.iloc[movie_idx]
        results.append((movie['title'], movie['movieId']))

    return results

#rerank_with_svd(candidates, user_id) function
def rerank_with_svd(candidates, user_id):
    results=[]

    for title, movie_id in candidates:
        pred=svd_model.predict(user_id, movie_id)
        results.append((title, movie_id, pred.est))
    
    results=sorted(results, key=lambda x: x[2], reverse=True)
    return results[:5]

#testing
print("models.py running")
if __name__ == "__main__":
    movie = "Interstellar"
    user_id = persona_map["enthusiast"]

    candidates = get_candidates(movie)
    print("Candidates (top 5):", candidates[:5])

    top5 = rerank_with_svd(candidates, user_id)
    print("Top 5:", top5)