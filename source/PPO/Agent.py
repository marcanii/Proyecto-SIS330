from PPO.ActorNetwork import ActorNetwork, DenseNetActor121, DenseNetActor201
from PPO.CriticNetwork import CriticNetwork
from PPO.PPOMemory import PPOMemory
import torch
import numpy as np
from PPO.AdvancedRewardCalculator import AdvancedRewardCalculator

class Agent:
    def __init__(self, n_actions, cuda, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                policy_clip=0.2, batch_size=4, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.previous_reward = 0

        self.actor = DenseNetActor201(n_actions, alpha, cuda)
        self.critic = CriticNetwork(alpha, cuda)
        self.memory = PPOMemory(batch_size)
        self.advancedRewardCalculator = AdvancedRewardCalculator()
    
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
    
    def choose_action(self, observation):
        state = observation.to(self.actor.device)
        #state_critic = state.view(1, -1)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        #print("StateShape: ", state.shape, "DistShape: ", dist.probs.shape, "ActionShape: ", action.shape)
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)
            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
        self.memory.clear_memory()

    def calculateReward(self, masks):
        reward, done = self.advancedRewardCalculator.calculate_reward(masks)
        return reward, done
        # done = False
        # total_area = np.sum(masks)
        # camino = np.sum(masks == 2)
        # obs = np.sum(masks == 1)
        # #print("Camino: ", camino, "Obs: ", obs, "TotalArea: ", total_area)
        # # Recompensa base por área de camino
        # reward_camino_base = 0.25 * camino

        # # Recompensa adicional por porcentaje de área de camino
        # area_camino_porcentaje = camino / total_area if total_area > 0 else 0
        # #print(area_camino_porcentaje,"=", camino, "/", total_area)
        # if area_camino_porcentaje >= 0.5:
        #     reward_camino_porcentaje = 0.75
        # elif area_camino_porcentaje >= 0.4:
        #     reward_camino_porcentaje = 0.60
        # elif area_camino_porcentaje >= 0.3:
        #     reward_camino_porcentaje = 0.45
        # elif area_camino_porcentaje >= 0.2:
        #     reward_camino_porcentaje = 0.30
        # else:
        #     reward_camino_porcentaje = 0

        # # Penalización por obstáculos
        # reward_obs = -0.1 * obs
        # #print("RewardObs: ", reward_obs)
        # # Recompensa por finalización
        # if obs > camino:
        #     done = True
        # elif camino >= total_area:
        #     done = True
        #     reward = 10

        # reward = reward_camino_base * reward_camino_porcentaje + reward_obs

        # return int(reward), done