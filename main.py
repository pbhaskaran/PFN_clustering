import torch
import utils
device = torch.device("cuda")
import prior
import sys
import time
#
# def train(model, criterion, num_epochs, optimizer, scheduler, batch_size, seq_len, num_features, **kwargs):
#     model.train()
#     trains = []
#     for e in range(num_epochs):
#         model.zero_grad()
#         X, y, y_noisy = prior.sample_clusters(batch_size=batch_size, seq_len=seq_len, num_features=num_features, **kwargs)
#         X, y_test, single_eval_pos = utils.compute_split(X, y, y_noisy)
#         output = model(X,single_eval_pos)
#         targets = y_test
#         output = output.view(-1, output.shape[2])
#         targets = targets.reshape(-1).type(torch.LongTensor).to(device)
#         loss = criterion(output, targets)
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         trains.append(loss.item())
#         if e % 10 == 0:
#             print('| epoch {:3d} | lr {} || ''loss {:5.3f}'.format(
#                 e, scheduler.get_last_lr()[0], loss))
#     return trains



def train(model, criterion, num_epochs, optimizer, scheduler, batch_size, seq_len, num_features,num_classes, **kwargs):
    model.train()
    trains = []
    start_time = time.time()
    for e in range(num_epochs):
        model.zero_grad()
        X, y, y_noisy = prior.sample_clusters(batch_size=batch_size, seq_len=seq_len, num_features=num_features, **kwargs)
        X, y_test, single_eval_pos = utils.compute_split(X, y, y_noisy)
        output = model(X,single_eval_pos)
        targets = y_test
        output = output.view(-1, output.shape[2])
        permutations = utils.permute(num_classes = num_classes)
        loss = float('inf')
        for permutation in permutations:
            target = utils.map_labels(permutation, targets)
            curr_loss = criterion(output, target)
            if curr_loss < loss:
                loss = curr_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        trains.append(loss.item())
        if e % 50 == 0:
            print('| epoch {:3d} | lr {} || ''loss {:5.3f}'.format(e, scheduler.get_last_lr()[0], loss))
    end_time = time.time()
    print(f"training completed in {end_time - start_time:.2f} seconds")
    return trains


if __name__ == '__main__':
    prior.sample_clusters()
    print("done")
