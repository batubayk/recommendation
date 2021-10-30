import torch
import torch.nn as nn
import torch.nn.functional as F


class RecNetwork(nn.Module):
    def __init__(self, shapes, embedding_dim, hidden_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=shapes['users'], embedding_dim=embedding_dim)
        self.movie_embedding = nn.Embedding(num_embeddings=shapes['movies'], embedding_dim=embedding_dim)
        self.hidden = nn.Linear(shapes['others'], hidden_dim)
        self.output_layer = nn.Linear(2 * embedding_dim + hidden_dim, 1)

    def forward(self, user_inp, movie_inp, others_inp):
        users_embed = self.user_embedding(user_inp)
        movies_embed = self.movie_embedding(movie_inp)
        others_out = F.leaky_relu(self.hidden(others_inp))
        x = torch.cat((users_embed, movies_embed, others_out),dim=1)
        x = self.output_layer(x)
        return torch.flatten(x)
