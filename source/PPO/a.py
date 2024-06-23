from ActorNetwork import ActorNetwork, DenseNetActor121, DenseNetActor201
import numpy as np
import torch
import time 

if __name__ == '__main__':
    classes = ['parar', 'atras', 'adelante', 'izquierda', 'derecha', 'giroIzq', 'giroDer']
    actor = DenseNetActor201(7, 0.0003, True)
    actor.load_checkpoint()
    inicio = time.time()
    x = np.load('source/PPO/dataset/adelante_3.npy')
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x / 2.0).float().to(actor.device)
    y = actor(x)
    print(classes[y.argmax().item()])
    print("Tiempo en segundos: ", time.time() - inicio)
