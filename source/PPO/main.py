import torchvision.models as models
from ActorNetwork import MyModelActorCNN, MobileNetActor
import torch

if __name__ == '__main__':
    #actor = models.mobilenet_v3_small(pretrained=True)
    actor = MobileNetActor(8, 0.0003, False)
    #print(actor)
    out = actor(torch.randn(1, 2, 60, 108))
    print(out)
    # N = 5
    # batch_size = 5
    # n_epochs = 4
    # alpha = 0.0003
    # agent = Agent(n_actions=5, input_dims=3*59*105,
    #               batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)

    # n_games = 10

    # best_score = -1000
    # score_history = []

    # learn_iters = 0
    # avg_score = 0
    # n_steps = 0

    # for i in range(n_games):
    #     observation = env.cameraOn()
    #     done = False
    #     score = 0
    #     while not done:
    #         action, prob, val = agent.choose_action(observation)
    #         observation_, reward, done, info = env.step(action)
    #         n_steps += 1
    #         score += reward
    #         agent.remember(observation, action, prob, val, reward, done)
    #         if n_steps % N == 0:
    #             agent.learn()
    #             learn_iters += 1
    #         observation = observation_
    #     score_history.append(score)
    #     avg_score = np.mean(score_history[-100:])

    #     if avg_score > best_score:
    #         best_score = avg_score

    #     print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
    #             'time_steps', n_steps, 'learning_steps', learn_iters)
    # x = [i+1 for i in range(len(score_history))]

    # agente = Agent(5, 3*59*105)
    # observation = torch.randn(1, 3, 59, 105)
    # action, probs, value = agente.choose_action(observation)
    # print(action, probs, value)