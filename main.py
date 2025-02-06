import torch
import utils
device = torch.device("cuda")
import prior
import sys
import time
import transformer
import torch.nn as nn

def train(model, criterion, num_epochs, optimizer, scheduler, batch_size, num_features, **kwargs):
    model.train()
    trains = []
    start_time = time.time()
    X_val, y_val, y_val_noisy,batch_clusters_val = prior.sample_clusters(batch_size=2 * batch_size, num_features=num_features, **kwargs)
    random_state = 0
    for e in range(num_epochs):
        model.zero_grad()
        X, y, y_noisy,batch_clusters = prior.sample_clusters(batch_size=batch_size, num_features=num_features, **kwargs)
        output,batch_cluster_output = model(X)
        targets = y
        targets_batch_clusters = batch_clusters
        output = output.view(-1, output.shape[2])
        batch_cluster_output = batch_cluster_output.view(-1, batch_cluster_output.shape[2])
        targets = targets.reshape(-1).type(torch.LongTensor).to(device)
        targets_batch_clusters = targets_batch_clusters.reshape(-1).type(torch.LongTensor).to(device)
        targets_batch_clusters -=1 ## super janky
        loss_output = criterion(output, targets)
        loss_clusters = criterion(batch_cluster_output, targets_batch_clusters)
        loss = 0.7 * loss_output + 0.3 * loss_clusters
        loss.backward()
        optimizer.step()
        scheduler.step()
        trains.append(loss.item())
        if e % 500 == 0:
            model.eval()
            val_output, val_cluster_output = model(X_val)
            val_output = val_output.view(-1, val_output.shape[2])
            val_targets = y_val.reshape(-1).type(torch.LongTensor).to(device)
            val_loss = criterion(val_output, val_targets)
            print('| epoch {:3d} | lr {} || ''validation loss {:5.3f}'.format(
                e, scheduler.get_last_lr()[0], val_loss))
            model.train()
        if e != 0 and e % 2000 == 0:
            torch.save(model.state_dict(), f"checkpoint.pt")
    end_time = time.time()
    print(f"training completed in {end_time - start_time:.2f} seconds")
    return trains


if __name__ == '__main__':
    print(f"Using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("total clustering main 2")
    device = torch.device("cuda")
    d_model, nhead, nhid, nlayers = 256, 4, 512, 4
    num_epochs = 25000
    lr = 0.001
    num_outputs = 10
    batch_size = 500
    in_features = 5
    noise = False
    warm_up_epochs = 5
    model = transformer.Transformer(d_model, nhead, nhid, nlayers, in_features=in_features,
                                    buckets_size=num_outputs).to(device)
    print(f"total params:{sum(p.numel() for p in model.parameters())}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = utils.get_cosine_schedule_with_warmup(optimizer, warm_up_epochs, num_epochs)
    model.criterion = criterion
    trains = train(model, criterion, num_epochs, optimizer, scheduler, batch_size, in_features,
                        num_classes=num_outputs,std_variation=True)
    torch.save(model.state_dict(), "saved_model_five_features.pt")
