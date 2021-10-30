import random
from pathlib import Path
import torch
import numpy as np
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

# Constants
RATINGS_PATH = Path("data/ml-100k/u.data")
MOVIE_PATH = Path("data/ml-100k/u.item")
USER_PATH = Path("data/ml-100k/u.user")
NUM_EPOCHS = 10
BATCH_SIZE = 4
EMBEDDING_SIZE = 32
HIDDEN_SIZE = 32
LEARNING_RATE = 0.002
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
shapes['others'] = movielens_df['occupation'].max() + 21

model = RecNetwork(shapes, embedding_dim=EMBEDDING_SIZE, hidden_dim=HIDDEN_SIZE)
print(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('The number of parameters of model is', num_params)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if torch.cuda.is_available():
    model = model.to(DEVICE)

for epoch in range(NUM_EPOCHS):
    for batch_idx,batch in enumerate(train_data_loader):
        movie_id, user_id, other_features, target = batch
        if torch.cuda.is_available():
            movie_id_tensor = movie_id_tensor.to(DEVICE)
            user_id_tensor = user_id_tensor.to(DEVICE)
            other_features = other_features.to(DEVICE)
            target = target.to(DEVICE)
        model.zero_grad()
        y = model(user_id,movie_id,other_features)
        loss = criterion(y,target)
        loss.backward()
        optimizer.step()

    # Validation
    for batch_idx,batch in enumerate(val_data_loader):
        with torch.no_grad():
            movie_id, user_id, other_features, target = batch
            if torch.cuda.is_available():
                movie_id_tensor = movie_id_tensor.to(DEVICE)
                user_id_tensor = user_id_tensor.to(DEVICE)
                other_features = other_features.to(DEVICE)
                target = target.to(DEVICE)
            y = model(user_id,movie_id,other_features)
            loss = criterion(y,target)
