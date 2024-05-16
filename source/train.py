import numbers as np
from PPO.Agent import Agent
from Environment import Environment

if __name__ == '__main__':
    N = 5
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    env = Environment()
    agent = Agent(n_actions=11, cuda=True, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)

    n_games = 10

    best_score = -1000
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.observation()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
