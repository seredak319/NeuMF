from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load models
mf_model = tf.keras.models.load_model('models/MF_model.h5')
neumf_model = tf.keras.models.load_model('models/NeuMF_model.h5')

# Load mappings and data
cleaned_links = pd.read_csv('data/cleaned_links.csv')
imdb_id_mapping = pd.read_csv('data/imdb_id_mapping.csv')
user_id_mapping = pd.read_csv('data/user_id_mapping.csv')
full_ratings = pd.read_csv('data/full_ratings.csv')
movies_metadata = pd.read_csv('data/cleaned_movies_metadata.csv')

num_users = user_id_mapping['new_user_id'].max() + 1
num_items = imdb_id_mapping['new_imdb_id'].max() + 1

def map_user_id(user_id):
    if user_id in user_id_mapping['userId'].values:
        return user_id_mapping[user_id_mapping['userId'] == user_id]['new_user_id'].values[0]
    else:
        # Handle new user ID
        new_id = user_id_mapping['new_user_id'].max() + 1
        user_id_mapping.loc[len(user_id_mapping)] = [user_id, new_id]
        user_id_mapping.to_csv('data/user_id_mapping.csv', index=False)
        return new_id

def get_unseen_movies(user_id, rated_movies):
    all_movies = set(full_ratings['movieId'].unique())
    seen_movies = set(rated_movies[(rated_movies['userId'] == user_id) & (rated_movies['rating'] == 1)]['movieId'].values)
    return list(all_movies - seen_movies)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('userId')

    print("@user_id = ", user_id)

    model_type = data.get('modelType', 'mf')  # 'mf' or 'neumf'

    # if not user_id:
    #     return jsonify({"error": "Invalid input"}), 400

    user_id = map_user_id(user_id)

    print("@ user_id = ", user_id)

    if user_id >= num_users:
        return jsonify({"error": "User ID is out of range"}), 400

    unseen_movies = get_unseen_movies(user_id, full_ratings)



    unseen_movies = [mid for mid in unseen_movies if mid < num_items]
    print("@unseen_movies = ", unseen_movies)
    if not unseen_movies:
        return jsonify({"error": "No unseen movies found"}), 400

    inputs = {
        'userId': np.array([user_id] * len(unseen_movies)),
        'movieId': np.array(unseen_movies)
    }

    if model_type == 'neumf':
        print("@ NeuMF")
        predictions = neumf_model.predict(inputs)
    else:
        print("@ MF")
        predictions = mf_model.predict(inputs)

    print("finished")
    print(predictions)

    top_indices = predictions.flatten().argsort()[-10:][::-1]
    # print(top_indices)
    top_movie_ids = [unseen_movies[i] for i in top_indices]
    # print(top_movie_ids)
    top_predictions = [predictions[i][0] for i in top_indices]
    print("@top_predictions ",top_predictions)
    top_imdb_ids = [imdb_id_mapping[imdb_id_mapping['new_imdb_id'] == mid]['imdb_id'].values[0] for mid in top_movie_ids]
    print("@top_imdb_ids ", top_imdb_ids)
    top_titles = [movies_metadata[movies_metadata['imdb_id'] == imdb_id]['title'].values[0] for imdb_id in top_imdb_ids]
    print("@top_titles ", top_titles)

    response = {
        "userId": str(user_id),
        "top_recommendations": [
            {
                "imdb_id": str(top_imdb_ids[i]),
                "title": str(top_titles[i]),
                "prediction": str(float(top_predictions[i]))  # Convert to percentage
            } for i in range(len(top_movie_ids))
        ]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
