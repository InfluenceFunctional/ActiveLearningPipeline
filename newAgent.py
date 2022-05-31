import gym
import torch
import numpy as np
from torch import nn
import random
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from RLmodels import MLP
from replay_buffer import ParameterUpdateReplayMemory
from types import SimpleNamespace

"""
Implementation of Double DQN for gym environments with discrete action space.
"""

class Agent():
    def __init__(self, config) -> None:
        self.device = config.device
        #RL Params
        self.epsilon_end = config.eps_min
        self.epsilon_decay = config.eps_decay
        self.epsilon = config.eps_start

        #Model Params
        self.action_size = config.action_size
        self.hidden_size = config.hidden_size
        self.state_length = config.state_length


        self.state = None

        self._create_models()

        # Optimizer and Training Params
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.tau = config.tau
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), self.alpha)
        self.memory = ParameterUpdateReplayMemory(config.memory_limit, device=self.device)
        self.update_type = config.update_type
        self.dqn_epochs = config.update_repeats
        self.batch_size = config.dqn_batch_size

    def updateState(self, state):
        self.state=torch.tensor(state)

    def _create_models(self):
        """Creates the Online and Target DQNs

        """
        # Query network (and target network for DQN)
        self.policy_net = MLP(self.state_length, self.action_size, self.hidden_size).to(self.device)
        self.target_net = MLP(self.state_length, self.action_size, self.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # sync parameters.
        for param in self.target_net.parameters():
            param.requires_grad = False

    def update_target_network(self):
        if self.update_type == 'hard':
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def select_action(self):
        with torch.no_grad():
            q_values = self.policy_net(self.state.to(self.device))

        if random.random() > self.epsilon:
            action = np.argmax(q_values.cpu().numpy())

        else:
            action = np.random.randint(0, self.action_size)

        return action

    def train(self):
        for _ in range(self.dqn_epochs):
            states, actions, next_states, rewards, is_done = self.memory.sample(self.batch_size)

            q_values = self.policy_net(states)

            next_q_values = self.policy_net(next_states)
            next_q_state_values = self.target_net(next_states)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

            expected_q_value = rewards + self.gamma * next_q_value * (1 - is_done*1)

            loss = (q_value - expected_q_value.detach()).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.policy_error = np.append(self.policy_error, loss.detach().cpu().numpy())

def main():
    config = SimpleNamespace(**{"gamma":0.99,
                "lr":1e-3,
                "min_memory_for_replay":500,
                "eps_start":1,
                "eps_decay":0.995,
                "eps_min":0.01,
                "update_step":1,
                "batch_size":64,
                "update_repeats":10,
                "seed":42,
                "max_memory_size":50000,
                "hidden_dim":64,
                "env_name":'CartPole-v1',
                "num_episodes": 2000,
                "dqn_batch_size": 64,
                "action_size": 2,
                "state_length": 4,
                "hidden_size": 64,
                "gamma": 0.95,
                "alpha": 0.001,
                "memory_limit": 50000,
                "min_memory": 500,
                "tau": 0.1,
                "update_type": 'soft',
                "device": 'cuda'})
    agent = Agent(config)
    env = gym.make(config.env_name)
    torch.manual_seed(config.seed)
    env.seed(config.seed)

    trial_episode_scores = []
    avg_grad_value = None
    debug_log = True
    i = 0
    for i_episode in range(config.num_episodes):
        episode_score = 0
        state = env.reset()
        agent.updateState(state)
        done = False

        while not done:
            i += 1
            action = agent.select_action()
            new_state, reward, done, _ = env.step(action)
            episode_score += reward

            # save state, action, reward sequence
            agent.memory.push(state, action, new_state, reward, done)
            state = new_state
            agent.updateState(state)
        if len(agent.memory) > config.min_memory_for_replay and (i_episode % config.update_step == 0):
            i+=30
            agent.policy_error = []
            agent.train()
            agent.update_target_network()
            avg_grad_value = torch.mean(torch.stack([torch.mean(abs(param.grad)) for param in agent.policy_net.parameters()]))

        avg_param_value = torch.mean(torch.stack([torch.mean(abs(param)) for param in agent.policy_net.parameters()]))

        agent.update_epsilon()

        ## Logging
        trial_episode_scores += [episode_score]
        last_100_avg = np.mean(trial_episode_scores[-100:])
        if debug_log:
            if avg_grad_value:
                print(f'E {i_episode} scored {episode_score}, avg {last_100_avg:.2f}, avg param {avg_param_value:.2f}', end = ', ')
                print(f'avg_grad {avg_grad_value:.2f} avg_loss {np.mean(agent.policy_error):.2f}')
            else:
                print(f'E {i_episode} scored {episode_score}, avg {last_100_avg:.2f}, avg param {avg_param_value:.2f}')

    print(i)

if __name__ == '__main__':
    import cProfile as profile
    import pstats
    prof = profile.Profile()
    prof.enable()
    main()
    stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
    prof.disable()
    # print profiling output
    stats.print_stats(30) # top 10 rows'''

