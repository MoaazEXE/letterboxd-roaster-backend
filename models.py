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

movies_df['title_lower']=movies_df['title'].str.lower()

#get_candidates(title) function
def get_candidates(title, n=20):
    title=title.lower()

    if title not in movies_df['title_lower'].values:
        raise ValueError("Movie not found")
    
    idx = movies_df[movies_df['title_lower'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    results=[]
    
    for i in sim_scores:
        movie=movies_df.iloc[i[0]]
        results.append((movie['title'], movie['movieId']))
    
    return results

#rerank_with_svd(candidates, user_id) function
def rerank_with_svd(candidates, user_id):
    results=[]

    for tile, movie_id in candidates:
        pred=svd_model.predict(user_id, movie_id)
        results.append((title, movie_id, pred.est))
    
    results=sorted(results, key=lambda x: x[2], reverse=True)
    return results[:5]
