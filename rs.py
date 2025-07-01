import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
# read in the csv using the read_csv command in pandas
df = pd.read_csv(r'anime-dataset-2023.csv')
columns_to_drop = ['anime_id', 'English name', 'Other name', 'Synopsis', 'Aired', 'Premiered', 'Status',
       'Producers', 'Licensors', 'Studios', 'Source', 'Duration', 'Rating',
       'Rank', 'Popularity', 'Favorites', 'Scored By', 'Members', 'Image URL']
df = df.drop(columns=columns_to_drop )
# convert 'episodes' and 'score' -- string -> integer
df['Episodes'] = pd.to_numeric(df['Episodes'], errors='coerce')
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
# divide the features into numerical and categorical
number_features = ['Episodes', 'Score']
categorical_features = ['Genres', 'Type']
# drop any N/A values
df = df.dropna(subset=number_features)
df = df.dropna(subset=categorical_features)
# declare the scaler and scale the numerical features
# scaling ensures that all values are within a certain range
scaler = StandardScaler()
scaled_numbers = scaler.fit_transform(df[number_features])
# hot one-encode type
# converts categorical 'type' into binary
# essentially, assigns 'type' a number
type_encoded = pd.get_dummies(df['Type'])
# fill any and all N/A or NaN values to be empty
df['Genres'] = df['Genres'].fillna('')
# transforms the strings of genres into a list of individual genres
# .apply(lambda x:) applies a certain function into every value (x) in the genre column
# g.strip for g in... strips any extra spaces
# x.lower() makes the string into lowercase
# .split(',') splits the genres using a comma
# if x else [] checks if x is not empty/null
genres = df['Genres'].apply(lambda x: [g.strip() for g in x.lower().split(',')] if x else [])
# encode the genres using the multilabelbinarizer
# turns each string of genres into a binary vector
# is able to handle multiple labels unlike hot-one encoding
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(genres)
# features is a variable that stores a stack of arrays
# hstack stacks arrays horrizontally (increaing columns, maintaining rows)
# --- rows: animes --- columns: scaled_numbers, type_encoded, genred_encoded
# use .values for type_encoded since they are binary
features = np.hstack([scaled_numbers, type_encoded.values, genres_encoded])
# function for the recommendation system!
def recommendation_system(name, df, choice_type, n, min_episodes = 0, max_episodes = float('inf')):
    df = df.copy()
    # find the row in which the anime name that was inputted is
    # ensures it is lowercase
    idx_list = df.index[df['Name'].str.lower() == name.lower()].tolist()
    # if the anime name was not found, return error message
    if not idx_list:
        return (f'{name} not found.')
    # if there are multiple matches, use the first one
    idx = idx_list[0]
    # choose what type of anime they want recommended
    if choice_type != 'any':
        recommend_pool = df[df['Type'] == choice_type]
        if choice_type == 'TV':
            recommend_pool = recommend_pool[(recommend_pool['Episodes'] >= min_episodes) & (recommend_pool['Episodes'] <= max_episodes)]
    else:
        recommend_pool = df.copy()
    
    # variable for target row
    target = df.loc[idx]
    # the following is the data that is extracted
    # clean the genres -> strip of extra spaces, make them lowercase, and split them with commas
    target_genres = set(g.strip() for g in target['Genres'].lower().split(',') if g.strip())
    target_type = target['Type']
    target_episodes = target['Episodes']
    target_rating = target['Score']
    # comparing genre function
    # takes a row and compares it to the target row
    def genre_similarity(row):
        if pd.isna(row['Genres']):
            return 0
        # similar to previous line, it cleans and extracts the genre(s) from a certain row
        genres = set(g.strip() for g in row['Genres'].lower().split(',') if g.strip())
        # returns similar genres
        return int(len(target_genres.intersection(genres)))
    # finding similarities / differences
    # apply 'genre_similarity' function to find number of shared genres
    recommend_pool['genre_sim'] = recommend_pool.apply(genre_similarity, axis=1)
    # returns integer 1 if type matches, 0 if not
    recommend_pool['type_sim'] = (recommend_pool['Type'] == target_type).astype(int)
    # finds differnce in number of episodes
    # the lower the number, the better // similar episode count
    recommend_pool['episodes_diff'] = (recommend_pool['Episodes'] - target_episodes).abs()
    # finds difference in rating
    # just like with episodes, the lower the number, the better // we want a similar rating
    recommend_pool['rating_diff'] = (recommend_pool['Score'] - target_rating).abs()

    # applying weights to each feature
    # i want genre to be prioritized when searching for similar anime, so that has a higher weight
    recommend_pool['weight_score'] = (
        recommend_pool['genre_sim'] * 1000 +
        recommend_pool['type_sim'] * 500 -
        recommend_pool['episodes_diff'] * 10 -
        recommend_pool['rating_diff'] * 5
    )
    # variable that stores the score in descending order
    # also returns top n results // in this case n = 5
    filtered = recommend_pool[(recommend_pool.index!= idx)]
    recommendations = filtered.sort_values('weight_score', ascending=False).head(n)

    return recommendations[['Name', 'Genres', 'Type', 'Episodes', 'Score']]

# UI
# import libraries
import streamlit as st
import base64
import os
# set the background of web app to a gif
def set_background(image_path):
    if not os.path.isfile(image_path):
        st.error(f"Background image not found at: {image_path}")
        return

    ext = os.path.splitext(image_path)[-1][1:]
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/{ext};base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background(r"C:\Users\britn\Downloads\Sailor Moon🌙 #sailormoon.gif")
# setting up the side bar
with st.sidebar:
    st.subheader('filter your recommendations:')
    # input number of recommendations
    number_recommendations = st.number_input('**how many recommendations would you like?**', min_value=1, max_value=10)
    # dropdown menu to select media type
    anime_type = st.selectbox('**what media type would you like?**', options=['any', 'TV', 'Movie','OVA', 'ONA', 'Special'])
    # initialize episode range
    episode_range = {'any': (0, float('inf')),
                     '1-11': (1,11),
                     '12-24': (12,24),
                     '25-48': (25,48),
                     '49-70': (49,70),
                     '71+': (71, float('inf'))}
    range_label = None
    min_episodes, max_episodes = 0, float('inf')
    # if 'TV' is selected, dropdown box pops up to select number of episodes
    if anime_type == 'TV':
        selected_range = st.selectbox('**select number of episodes**', list(episode_range))
        min_episodes, max_episodes = episode_range[selected_range]


# main page details
st.title('anime recommender ⋆.˚⟡ ࣪ ˖')
st.subheader('**don\'t know what to watch? we can recommend some anime for you!**')
st.write('**enter an anime title and we\'ll show you some similar ones :)**')
anime = st.text_input('**enter anime title:**')
# if button is clicked:
if st.button('**get recommendations!**'):
    # if field is filled:
    if anime:
        # variable to store recommendation results
        results = recommendation_system(anime, df, anime_type, n = number_recommendations, min_episodes=min_episodes, max_episodes=max_episodes)
        # if the result is a string ('anime was not found') then display the error message
        if isinstance(results, str):
            st.error(results)
        # if the result is not a string- a dataframe- then print out the recommended anime
        # we want the result to be a dataframe, not a string!!
        elif not results.empty:
            st.subheader('recommended anime:')
            st.dataframe(results)
        # however, if the result is an empty dataframe
        # print out an error message
        else:
            st.error('**no recommendations found...**')
