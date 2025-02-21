import math
import torch
import torch.nn as nn
device = torch.device("cuda")


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std


class Transformer(nn.Transformer):
    def __init__(self, d_model, nhead, nhid, nlayers, in_features=1, buckets_size=100):
        super(Transformer, self).__init__(d_model=d_model, nhead=nhead, dim_feedforward=nhid, dropout=0,
                                          num_encoder_layers=nlayers)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.d_model = d_model
        self.nhead = nhead
        self.nhid = nhid
        self.nlayers = nlayers
        self.embed_size = d_model // 2
        self.linear_x = nn.Linear(in_features, d_model)
        self.decoder = nn.Linear(d_model + self.embed_size, buckets_size)
        self.embedding = nn.Embedding(buckets_size + 1,self.embed_size)

    def _generate_mask(self, size):
        matrix = torch.zeros((size, size), dtype=torch.float32)
        matrix[:size -1, :size - 1] = 1
        matrix[size -1] = 1
        matrix = matrix.masked_fill(matrix == 0, float('-inf')).masked_fill(matrix == 1, 0)
        return matrix.to(device)


    def forward(self, X,num_clusters):
        # convert features to higher dimension
        cluster_input = torch.full_like(X[:1], -1, dtype=torch.float, device=device)
        X = torch.cat((X, cluster_input),dim=0)
        train = (self.linear_x(X)) # Shape S + 1,B, d_model

        src_mask = self._generate_mask(train.shape[0]) # Generate mask
        output = self.encoder(train, mask=src_mask) # S + 1, B, E

        num_clusters = num_clusters.squeeze(0)
        cluster_conditional = self.embedding(num_clusters) # shape B, E /2
        cluster_conditional = cluster_conditional.expand(train.shape[0], -1, -1) # shape S, B, E/2
        output = torch.cat([output, cluster_conditional], dim=-1) # Shape S, B, E + E /2
        output = self.decoder(output)
        return output[:-1], output[-1:]