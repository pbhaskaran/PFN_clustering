import math
import torch
import torch.nn as nn


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
        #self.linear_x = nn.Linear(in_features, d_model) # todo do we need to normalize here
        self.linear_y = nn.Linear(1, d_model) # todo might need to normalize this as well
        self.decoder = nn.Linear(d_model, buckets_size)

    def _generate_mask(self, size, input_pair_length):
        matrix = torch.zeros((size, size), dtype=torch.float32)
        n = input_pair_length
        matrix[:size, :input_pair_length] = 1
        # Set the remaining rows with ones until column n and set the diagonal to 1
        for i in range(n, size):
            matrix[i, i] = 1  # Set the diagonal entry for this row
        matrix = matrix.masked_fill(matrix == 0, float('-inf')).masked_fill(matrix == 1, 0)
        return matrix

    def forward(self, X_train, y_train, X_test, single_eval_pos, has_mask=True):
        y_train = y_train.unsqueeze(-1)
        train = (self.linear_x(X_train) + self.linear_y(y_train))
        X_test = self.linear_x(X_test)
        train = torch.cat((train, X_test), dim=0)
        if not has_mask:
            train = X_test

        if has_mask:
            device = train.device
            src_mask = self._generate_mask(train.shape[0], single_eval_pos).to(device)
        else:
            src_mask = None
        output = self.encoder(train, mask=src_mask)

        # shape (S, B, bucket_size) but we need only need outputs from the queries (m)
        output = self.decoder(output)
        # shape (m, B, bucket_size)
        output = output[single_eval_pos:]
        return output





