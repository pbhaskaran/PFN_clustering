import torch
import utils
device = torch.device("cuda")
import prior

def train(model, criterion, num_epochs, optimizer, scheduler, batch_size, seq_len, num_features, **kwargs):
    model.train()
    trains = []
    for e in range(num_epochs):
        model.zero_grad()
        X, y, y_noisy = prior.sample_clusters(batch_size=batch_size, seq_len=seq_len, num_features=num_features, **kwargs)
        X_train, y_train, X_test, y_test, single_eval_pos = utils.compute_split(X, y, y_noisy)
        output = model(X_train, y_train, X_test, single_eval_pos)
        targets = y_test
        output = output.view(-1, output.shape[2])
        targets = targets.reshape(-1).type(torch.LongTensor).to(device)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        trains.append(loss.item())
        if e % 10 == 0:
            print('| epoch {:3d} | lr {} || ''loss {:5.3f}'.format(
                e, scheduler.get_last_lr()[0], loss))
    return trains


if __name__ == '__main__':
    prior.sample_clusters()
    print("done")
