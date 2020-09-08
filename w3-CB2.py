# Created on Sun Sep 15 2019
# author: 임일

import pandas as pd
import numpy as np

#Load the keywords and credits files
movies = pd.read_csv(r'C:/RecoSys/Data/movies_metadata.csv')
movies = movies[['id', 'title']]
credits = pd.read_csv(r'C:/RecoSys/Data/credits.csv')
keywords = pd.read_csv(r'C:/RecoSys/Data/keywords.csv')
credits.head()
keywords.head()

#Function to convert all non-integer IDs to NaN
def clean_ids(x):
    try:
        return int(x)
    except:
        return np.nan

#Clean the ids of df
movies['id'] = movies['id'].apply(clean_ids)

#Filter all rows that have a null ID
movies = movies[movies['id'].notnull()]

#Convert IDs into integer
movies['id'] = movies['id'].astype('int')
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')

#Merge keywords and credits into your main metadata dataframe
movies = movies.merge(credits, on='id')
movies = movies.merge(keywords, on='id')

#Display the head of the merged movies
movies.head()

#Convert the stringified objects into the native python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords']
for feature in features:
    movies[feature] = movies[feature].apply(literal_eval)
    
#Print the first cast member of the first movie
movies.iloc[0]['crew'][0]

#Extract the direct's name. If director is not listed, return NaN
def get_director(x):
    for crew_member in x:
        if crew_member['job'] == 'Director':
            return crew_member['name']
        return np.nan
    
#Define the new director feature
movies['director'] = movies['crew'].apply(get_director)

#Return the list top 3 elements
def generate_list(x):
    if isinstance(x, list):
        names = [item['name'] for item in x]
        #Check if more than 3 elements exist. If yes, return only first three.
        #If not, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names
    #Return empty list in case of missing/malformed data
    return []

#Apply the generate_list function to cast and keywords
movies['cast'] = movies['cast'].apply(generate_list)
movies['keywords'] = movies['keywords'].apply(generate_list)

#Print the new features of the first 5 movies along with title
movies[['title', 'cast', 'director', 'keywords']].head(5)

#Removes spaces and converts to lowercase
def sanitize(x):
    if isinstance(x, list):
        #Strip spaces and convert to lowercase
        return [str.lower(i.replace(" ",""))for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance (x, str):
            return str.lower(x.replace(" ",""))
        else:
            return ''
        
#Apply the generate_list function to cast, keywords, and director
for feature in ['cast', 'director', 'keywords']:
    movies[feature] = movies[feature].apply(sanitize)
    
#Function that creates a soup out of the desired metadata
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director']

#Create the new soup feature
movies['soup'] = movies.apply(create_soup, axis=1)

#Display the soup of the first movie
movies.iloc[0]['soup']

#Import CountVectorizer from the scikit-learn library
from sklearn.feature_extraction.text import CountVectorizer

#Define a new CountVectorizer object and create vectors for the soup
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(movies['soup'])

#Import cosine_similarity function
from sklearn.metrics.pairwise import cosine_similarity

#Compute the cosine similarity score (equivalent to dot product for tf-idf vectors)
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

#Reset index of your movies and construct reverse mapping again
movies = movies.reset_index()
indices2 = pd.Series(movies.index, index=movies['title'])


def content_recommender2(title, cosine_sim=cosine_sim2, movies=movies, indices=indices2):
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
print(content_recommender2('The Dark Knight Rises', cosine_sim2, movies, indices2))
