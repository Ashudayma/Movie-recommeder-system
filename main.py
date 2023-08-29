# i have comment-down everything that i have done in this project

import pandas as pd

# accessing the dataset
movies = pd.read_csv("https://raw.githubusercontent.com/Ashudayma/Movie-recommeder-system/main/tmdb_5000_movies.csv")
credits = pd.read_csv("https://raw.githubusercontent.com/Ashudayma/Movie-recommeder-system/main/tmdb_5000_credits.csv")

# merging the dataset
movies = movies.merge(credits, on='title')

# adding the important column only on which we recommend the movies
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

#this is a func which will be applied on the columns
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

# droping the null value
movies.dropna(inplace=True)

# applying the convert fun on genres column
movies['genres'] = movies['genres'].apply(convert)

# applying the convert function on keyword column
movies['keywords'] = movies['keywords'].apply(convert)

import ast

ast.literal_eval(
    '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L


# applying the convert function on cast column
movies['cast'] = movies['cast'].apply(convert)
movies.head()

movies['cast'] = movies['cast'].apply(lambda x: x[0:3])


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


movies['crew'] = movies['crew'].apply(fetch_director)

movies.sample(5)


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1


# applying the collapse function
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

# now merging the imp columns so that we can recommend the movie on the basis of one column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# now droping all the rest columns which are not required
new = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])

new['tags'] = new['tags'].apply(lambda x: " ".join(x))

# importing the libraries
from sklearn.feature_extraction.text import CountVectorizer

# now here we are doing the text vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')

# using the function of the libraries
vector = cv.fit_transform(new['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

# now here comes the most important part of the project, which is checking the similarity through cosine similarity b/w vectors
similarity = cosine_similarity(vector)

new[new['title'] == 'The Lego Movie'].index[0]

def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)

# checking the testcase
print(recommend('Spider-Man'))

# making the tag column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# droping the columns after adding to tags
new = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])

new['tags'] = new['tags'].apply(lambda x: " ".join(x))

# here i have display the project on the server and the rest code is in main.py file
import pickle

pickle.dump(new, open('movies.pkl', 'wb'))

print(new['title'].values)

pickle.dump(new.to_dict(), open('movie.pkl', 'wb'))

pickle.dump(similarity, open('similarity.pkl', 'wb'))
