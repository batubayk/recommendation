import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader

from helpers.data import RecommendationDataset, load_movielens_dataset
from helpers.model import RecNetwork

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def train_val_test_loop(model, optimizer, criterion, data_loader, device, print_interval, mode="train"):
    total_loss = 0
    for batch_idx, batch in enumerate(data_loader):
        movie_id, user_id, other_features, target = batch
        if torch.cuda.is_available():
            movie_id = movie_id.to(device)
            user_id = user_id.to(device)
            other_features = other_features.to(device)
            target = target.to(device)
        if mode == "train":
            model.zero_grad()
            y = model(user_id, movie_id, other_features)
            loss = criterion(y, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        else:
            with torch.no_grad():
                y = model(user_id, movie_id, other_features)
                loss = criterion(y, target)
                total_loss += loss.item()

        if (batch_idx + 1) % print_interval == 0:
            print("Mode:{} \t Batch: {}/{} \t Loss: {} \t Avg Loss: {}".format(mode, batch_idx + 1,
                                                                               len(data_loader), loss.item(),
                                                                               total_loss / (batch_idx + 1)))

    return total_loss / (batch_idx + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--ratings_path', default="data/ml-100k/u.data", type=str)
    parser.add_argument('--movie_path', default="data/ml-100k/u.item", type=str)
    parser.add_argument('--user_path', default="data/ml-100k/u.user", type=str)
    parser.add_argument('--model_path', default="rec_model.pt", type=str)

    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=int)

    parser.add_argument('--print_batch_interval', default=10, type=int)

    args = parser.parse_args()

    # Constants
    print(args)
    RATINGS_PATH = Path(args.ratings_path)
    MOVIE_PATH = Path(args.movie_path)
    USER_PATH = Path(args.user_path)
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    EMBEDDING_SIZE = args.embedding_size
    HIDDEN_SIZE = args.hidden_size
    LEARNING_RATE = args.learning_rate
    PRINT_BATCH_INTERVAL = args.print_batch_interval
    MODEL_PATH = args.model_path
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    movielens_df = load_movielens_dataset(RATINGS_PATH, MOVIE_PATH, USER_PATH)

    # Label encode categorical feats
    label_object = {}
    categorical_columns = ['gender', 'occupation']
    for col in categorical_columns:
        labelencoder = LabelEncoder()
        labelencoder.fit(movielens_df[col])
        movielens_df[col] = labelencoder.fit_transform(movielens_df[col])
        label_object[col] = labelencoder

    # Scale continuous feats
    scaler = MinMaxScaler()
    scaler.fit(movielens_df[["age"]])
    movielens_df["age"] = scaler.transform(movielens_df[["age"]])

    # Train, val, test split
    train, test = train_test_split(movielens_df, test_size=0.1)
    train, val = train_test_split(train, test_size=0.1)

    train_data_loader = DataLoader(RecommendationDataset(train), batch_size=BATCH_SIZE)
    val_data_loader = DataLoader(RecommendationDataset(val), batch_size=BATCH_SIZE)
    test_data_loader = DataLoader(RecommendationDataset(test), batch_size=BATCH_SIZE)

    # Model init
    shapes = dict()
    shapes['users'] = movielens_df['user_id'].max()
    shapes['movies'] = movielens_df['movie_id'].max()
    shapes['others'] = movielens_df['occupation'].max() + 22

    model = RecNetwork(shapes, embedding_dim=EMBEDDING_SIZE, hidden_dim=HIDDEN_SIZE)
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training
    if torch.cuda.is_available():
        model = model.to(DEVICE)

    for epoch in range(NUM_EPOCHS):
        train_val_test_loop(model, optimizer, criterion, val_data_loader, DEVICE, PRINT_BATCH_INTERVAL)

        # Validation
        train_val_test_loop(model, optimizer, criterion, val_data_loader, DEVICE, PRINT_BATCH_INTERVAL, mode="val")

    # Test
    train_val_test_loop(model, optimizer, criterion, test_data_loader, DEVICE, PRINT_BATCH_INTERVAL, mode="test")

    # Save the model
    torch.save(model.state_dict(), MODEL_PATH)
