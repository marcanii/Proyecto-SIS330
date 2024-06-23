from ActorNetwork import ActorNetwork
import numpy as np
import torch

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = ['parar', 'atras', 'adelante', 'izquierda', 'derecha', 'giroIzq', 'giroDer']
    actor = ActorNetwork(7, 0.0003, False)
    actor.load_checkpoint()

    X_test = np.load('source/PPO/dataset/atras_20.npy')

    X_test = np.expand_dims(X_test, axis=0)
    X_test = np.expand_dims(X_test, axis=0)
    #print(X_test.shape)
    X_test = torch.from_numpy(X_test).float()

    y_hat = actor(X_test)
    print(classes[torch.argmax(y_hat).item()])
