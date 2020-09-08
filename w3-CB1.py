# Created on Sun Sep 15 2019
# author: 임일

import pandas as pd
import numpy as np

#Import the meat data file and keep only necessary columns
movies = pd.read_csv(r'C:/RecoSys/Data/movies_metadata.csv', low_memory=False)
movies = movies[['id', 'title', 'overview']]

#Import TfIdfVectorizer from the scikit-learn library
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stopwords
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
movies['overview'] = movies['overview'].fillna('')

#Construct the required TF-IDF matrix from the overview features
tfidf_matrix = tfidf.fit_transform(movies['overview'])

#Import linear_kernel to compute the dot product
from sklearn.metrics.pairwise import linear_kernel

#Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse mapping of idices and movie titles, and drop duplicate titles
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

#Function that takes in movie title as input and gives recommendations
def content_recommender(title, cosine_sim=cosine_sim, movies=movies, indices=indices):
    #Obtain the index of the movie that matches the title
    idx = indices[title]
    
    #Get the pairwise similarity scores of all movies with that movie
    #And convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    #Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    #Get the scores of the 10 most similar movies. Ignore the first movie (self)
    sim_scores = sim_scores[1:11]
    
    #Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    #Return the top 10 most similar movies
    return movies['title'].iloc[movie_indices]

#Get recommendations for a movie
print(content_recommender('The Dark Knight Rises'))

