import numpy as np
from Agent import Agent
import csv
import time

def append_scores(file_path, new_scores):
    existing_scores = []
    last_episode = 0

    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Omitir la fila de encabezado
            existing_scores = [(int(row[0]), float(row[1])) for row in reader]
            last_episode = existing_scores[-1][0] + 1 if existing_scores else 0
    except FileNotFoundError:
        pass

    all_scores = existing_scores + [(episode + last_episode, score) for episode, score in enumerate(new_scores)]

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Score'])
        writer.writerows(all_scores)

if __name__ == "__main__":
    agent = Agent()
    if agent.loadModels():
        print("Modelos cargados correctamente...")
    else:
        print("Error al cargar los modelos...")
    N = 4
    n_games = 10

    best_score = -np.inf
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, _, _ = agent.observation()
        done = False
        score = 0
        while not done:
            init = time.time()
            action, prob, val = agent.chooseAction(observation)
            observation_, reward, done = agent.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation_, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                print("Aprendiendo...")
                learn_iters += 1
            observation = observation_
            print("Tiempo:", time.time() - init)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
        code = agent.saveModels()
        print("Guardar modelos:", code)

    x = [i+1 for i in range(len(score_history))]
    print(x)
    print(score_history)
    append_scores('/home/jetson/Proyecto-SIS330/Jetson/score_history.csv', score_history)

    print("Historial de puntuaciones guardado en score_history.csv")
