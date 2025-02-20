import torch
import utils
device = torch.device("cuda")
import prior
import sys
import time
import transformer
import torch.nn as nn

def train(model, criterion, num_epochs, optimizer, scheduler, batch_size,seq_len, num_features, **kwargs):
    model.train()
    trains = []
    start_time = time.time()
    random_seed = 0
    X_val, y_val,X_true,y_true,batch_clusters_val = prior.sample_dirichlet_clusters(batch_size=2 * batch_size,seq_len=seq_len, num_features=num_features, random_seed=random_seed, **kwargs)
    val_mask = (torch.zeros(batch_clusters_val.shape)).long().to(device)

    for e in range(num_epochs):
        model.zero_grad()
        X, y, X_true, y_true, batch_clusters = prior.sample_dirichlet_clusters(batch_size=batch_size,seq_len=seq_len, num_features=num_features,random_seed=random_seed, **kwargs)
        mask = (torch.rand(batch_clusters.shape) > 0.1).long().to(device)
        batch_clusters_masked = mask * batch_clusters
        output,batch_cluster_output = model(X, batch_clusters_masked)
        targets = y
        targets_batch_clusters = batch_clusters

        output = output.view(-1, output.shape[2])
        batch_cluster_output = batch_cluster_output.view(-1, batch_cluster_output.shape[2])
        targets = targets.reshape(-1).type(torch.LongTensor).to(device)
        targets_batch_clusters = targets_batch_clusters.reshape(-1).type(torch.LongTensor).to(device)
        targets_batch_clusters -=1 ## super janky
        loss_output = criterion(output, targets)
        loss_clusters = criterion(batch_cluster_output, targets_batch_clusters)
        loss = loss_output + loss_clusters
        loss.backward()
        optimizer.step()
        scheduler.step()
        trains.append(loss.item())
        if e % 1000 == 0:
            model.eval()
            val_output, val_cluster_output = model(X_val, val_mask)
            val_output = val_output.view(-1, val_output.shape[2])
            val_targets = y_val.reshape(-1).type(torch.LongTensor).to(device)
            val_loss = criterion(val_output, val_targets)
            print('| epoch {:3d} | lr {} || ''validation loss {:5.3f}'.format(
                e, scheduler.get_last_lr()[0], val_loss))
            model.train()
        if e != 0 and e % 2000 == 0:
            torch.save(model.state_dict(), f"check_point_half_dim_dirichlet_01.pt")
        random_seed += 1
    end_time = time.time()
    print(f"training completed in {end_time - start_time:.2f} seconds")
    return trains


if __name__ == '__main__':
    print(f"Using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("conditional")
    device = torch.device("cuda")
    d_model, nhead, nhid, nlayers = 256, 4, 512, 4
    num_epochs = 20000
    lr = 0.001
    num_outputs = 10
    batch_size = 150
    in_features = 2
    seq_len = 200
    noise = False
    warm_up_epochs = 5
    model = transformer.Transformer(d_model, nhead, nhid, nlayers, in_features=in_features,
                                    buckets_size=num_outputs).to(device)
    print(f"total params:{sum(p.numel() for p in model.parameters())}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = utils.get_cosine_schedule_with_warmup(optimizer, warm_up_epochs, num_epochs)
    model.criterion = criterion
    trains = train(model, criterion, num_epochs, optimizer, scheduler, batch_size,seq_len, in_features,
                        num_classes=num_outputs,std_variation=True)
    torch.save(model.state_dict(), "saved_model_half_dim_dirichlet_01.pt")
