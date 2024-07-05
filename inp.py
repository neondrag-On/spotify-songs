import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.metrics.pairwise import cosine_similarity

# Load the movie dataset
#movies_data = pd.read_csv('Popular_Spotify_Songs.csv')
movies_data = pd.read_csv('Popular_Spotify_Songs.csv', encoding='latin-1')


selected_features = ['artist(s)_name', 'in_spotify_playlists', 'streams', 'acousticness_%', 'danceability_%']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('').astype(str)

# Combine selected features into one
combined_features = movies_data[selected_features].apply(lambda x: ' '.join(str(xi) for xi in x), axis=1)

# Convert text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)

# Cosine Similarity - getting similarity scores
similarity = cosine_similarity(feature_vector)


st.title("Song Recommendation App")
st.text("Get Recommendation with in seconds")



# Get the mo name from the user
movie_name = st.text_input("Enter your favorite song name: ")

# Find the close match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name, movies_data['track_name'], n=1)
if find_close_match:
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data['track_name'] == close_match].index[0]

    # Getting list of similar movies
    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    # Sorting the movies based on their similarity score
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    st.balloons()
    st.write('Songs suggested for you :')

    # Display suggested movies
    for i, movie in enumerate(sorted_similar_movies[:10], 1):
        index = movie[0]
        title_from_index = movies_data.loc[index, 'track_name']
        tagline_from_index = movies_data.loc[index, 'artist(s)_name']
        url = movies_data.loc[index, 'released_year']
        st.write(f"{i}. {title_from_index} by {tagline_from_index} (In Playlists: {url})")
        
        #st.write(f"{i}. {title_from_index} : {tagline_from_index} ( {url} )" )
        
