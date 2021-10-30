import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class RecommendationDataset(Dataset):
    def __init__(self, df):
        self.dataset = df
        self.max_num_users = df['user_id'].max()
        self.max_num_movies = df['movie_id'].max()
        self.max_num_occupation = df['occupation'].max()

        self.movie_genre_cols = ["unknown", "Action", "Adventure", "Animation",
                                 "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                                 "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                                 "Thriller", "War", "Western"]

    def __getitem__(self, idx):
        movie_id = int(self.dataset.iloc[idx]['movie_id'] - 1)
        movie_genres_tensor = self.dataset.iloc[idx][self.movie_genre_cols].tolist()
        movie_genres_tensor = torch.Tensor(movie_genres_tensor)

        user_id = int(self.dataset.iloc[idx]['user_id'] - 1)
        user_age = torch.Tensor([self.dataset.iloc[idx]["age"]])
        user_gender = torch.Tensor([self.dataset.iloc[idx]["gender"]])
        user_occupation = int(self.dataset.iloc[idx]["occupation"] - 1)
        user_occupation_tensor = F.one_hot(torch.as_tensor(user_occupation), num_classes=self.max_num_occupation)

        other_features = torch.cat((movie_genres_tensor, user_occupation_tensor, user_gender, user_age))

        target = torch.as_tensor(self.dataset.iloc[idx]["rating"], dtype=torch.float32)

        return movie_id, user_id, other_features, target

    def __len__(self):
        return len(self.dataset)


def load_movielens_dataset(ratings_path, movie_path, user_path):
    ratings_data_columns = ["user_id", "movie_id", "rating", "timestamp"]
    rating_df = pd.read_csv(ratings_path, sep="\t", names=ratings_data_columns, encoding='latin-1')

    movie_data_columns = ["movie_id", "movie_title", "release_date", "video_release_date",
                          "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
                          "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                          "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                          "Thriller", "War", "Western"]
    movies_df = pd.read_csv(movie_path, sep="|", names=movie_data_columns, encoding='latin-1')

    users_data_columns = ["user_id", "age", "gender", "occupation", "zip_code"
                          ]
    users_df = pd.read_csv(user_path, sep="|", names=users_data_columns, encoding='latin-1')

    df = pd.merge(pd.merge(movies_df, rating_df), users_df)

    # Drop some columns to simplfy the data although features might be useful. For instance movie title can be encoded using BERT or an RNN
    df.drop(columns=["timestamp", "IMDb_URL", "movie_title", "release_date", "video_release_date", "zip_code"],
            inplace=True)
    return df.sample(frac=1)
