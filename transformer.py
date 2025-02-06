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

        self.linear_x = nn.Sequential(Normalize(.5, math.sqrt(1 / 12)), nn.Linear(in_features, d_model))
        self.linear_num_clusters = nn.Linear(in_features, d_model)
        self.decoder = nn.Linear(d_model, buckets_size)

    def _generate_mask(self, size):
        matrix = torch.zeros((size, size), dtype=torch.float32)
        matrix[:size -1, :size - 1] = 1
        matrix[size -1] = 1
        matrix = matrix.masked_fill(matrix == 0, float('-inf')).masked_fill(matrix == 1, 0)
        return matrix.to(device)


    def forward(self, X):
        train = (self.linear_x(X))
        cluster_input = torch.full((1,X.shape[1], X.shape[2]), -1, dtype=torch.float, device=device)
        cluster_embedding = self.linear_num_clusters(cluster_input)
        train  = torch.cat((train, cluster_embedding) , dim=0)
        src_mask = self._generate_mask(train.shape[0])
        output = self.encoder(train, mask=src_mask)
        output = self.decoder(output)
        return output[:train.shape[0] -1], output[-1:]





