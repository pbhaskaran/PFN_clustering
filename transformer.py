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
        self.linear_num_clusters = nn.Linear(1, d_model)
        self.linear_desired_clusters = nn.Linear(1, d_model)
        self.decoder = nn.Linear(d_model, buckets_size)

    def _generate_mask(self, size):
        matrix = torch.zeros((size, size), dtype=torch.float32)
        matrix[:size -1, :size - 1] = 1
        matrix[size -1] = 1
        matrix = matrix.masked_fill(matrix == 0, float('-inf')).masked_fill(matrix == 1, 0)
        return matrix.to(device)


    def forward(self, X, desired_clusters=None):
        train = (self.linear_x(X))
        cluster_input = torch.full((1, X.shape[1], 1) , -1, device=device, dtype=torch.float32)
        if desired_clusters is None:
            cluster_embedding = self.linear_num_clusters(cluster_input)
            train  = torch.cat((train, cluster_embedding) , dim=0)
            src_mask = self._generate_mask(train.shape[0])
        else:
            desired_cluster_embedding = self.linear_desired_clusters(desired_clusters).unsqueeze(0)
            train  = torch.cat((train, desired_cluster_embedding) , dim=0)
            src_mask = self._generate_desired_mask(train.shape[0])
        output = self.encoder(train, mask=src_mask)
        output = self.decoder(output)
        return output[:train.shape[0] -1], output[-1:]

    def _generate_desired_mask(self, size):
        matrix = torch.ones((size, size), dtype=torch.float32)
        matrix[size -1, :] = 0
        matrix = matrix.masked_fill(matrix == 0, float('-inf')).masked_fill(matrix == 1, 0)
        return matrix.to(device)


