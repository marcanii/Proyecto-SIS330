from ActorNetwork import ActorNetwork, MyModelActorCNN
import numpy as np
import os
import torch

class Dataset(torch.utils.data.Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, ix):
    img = X[ix]
    return torch.from_numpy(img).float(), torch.tensor(self.y[ix]).long()

def laod_dataset(data_dir):
    X = []
    Y = []
    label_dict = {'parar': 0, 'atras': 1, 'adelante': 2, 'izquierda': 3, 'derecha': 4, 'giroIzq': 5, 'giroDer': 6}

    for filename in os.listdir(data_dir):
        if filename.endswith('.npy'):
            label = filename.split('_')[0]
            #print(label)
            img = np.load(os.path.join(data_dir, filename))
            img = np.expand_dims(img, axis=0)
            X.append(img)
            Y.append(label_dict[label])
    
    return np.array(X), np.array(Y)

def fit(model, dataloader, epochs=10, lr=0.0003, device='cpu'):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs+1):
        model.train()
        train_loss, train_acc = [], []
        #bar = tqdm(dataloader['train'])
        for batch in dataloader['train']:
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
            train_acc.append(acc)
            #bar.set_description(f"loss {np.mean(train_loss):.5f} acc {np.mean(train_acc):.5f}")

        if epoch % 10 == 0:
            model.save_checkpoint()
            print(f"Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f} acc {np.mean(train_acc):.5f}")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #classes = ['parar', 'atras', 'adelante', 'izquierda', 'derecha', 'giroIzq', 'giroDer']
    actor = ActorNetwork(7, 0.0003, False, False)
    #actor.load_checkpoint()
    data_dir = 'source/PPO/dataset'
    X, y = laod_dataset(data_dir)
    dataset = Dataset(X, y)

    dataset = {
        'train': Dataset(X, y),
    }
    dataloader = {
        'train': torch.utils.data.DataLoader(dataset['train'], batch_size=8, shuffle=True, pin_memory=True),
    }
    #imgs, labels = next(iter(dataloader['train']))
    #print(imgs.shape)
    #print(dataset['train'][2])
    fit(actor, dataloader, epochs=100, lr=0.0003, device=device)
