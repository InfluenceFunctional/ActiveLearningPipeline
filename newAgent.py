import gym
import torch
import numpy as np
from torch import nn
import os
import random
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from RLmodels import MLP
from replay_buffer import ReplayMemory
from oracle import Oracle
from utils import printRecord, get_n_params
from comet_ml import Experiment
"""
Implementation of Double DQN for gym environments with discrete action space.
"""

class DQN():
    def __init__(self, config) -> None:
        # General Properties
        self.config = config
        self.device = config.device
        self.load = False
        self.exp_name = config.exp_name
        self.log_rl_to_console = config.rl.log_rl_to_console
        # RL Properties
        self.epsilon_end = config.rl.eps_end
        self.epsilon_decay = config.rl.eps_decay
        self.epsilon = config.rl.eps_start
        self.gamma = config.rl.gamma
        self.tau = config.rl.tau
        self.dqn_train_frequency = config.rl.dqn_train_frequency
        self.memory = ReplayMemory(config.rl.buffer_size, device=self.device)
        self.dqn_epochs = config.rl.dqn_epochs
        self.batch_size = config.rl.dqn_batch_size

        # Agent Properties
        self.episode = 0
        self.policy_error = []
        self.step = 0
        self.state = None

        # Optimizer and Training Properties
        self.alpha = config.rl.alpha
        self.max_grad_norm = config.rl.max_grad_norm
        self.optimizer_param = {
            "opt_choice": config.querier.opt,
            "momentum": config.querier.momentum,
            "ckpt_path": "./ckpts/",
            "exp_name_toload": config.querier.model_ckpt,
            "exp_name": self.exp_name,
            "snapshot": 0,
            "load_opt": self.load,
            "lr_dqn": self.alpha
        }
        self.lr_step = config.rl.lr_step
        self.lr_gamma = config.rl.lr_gamma

    def _load_models(self, file_name="policy_agent"):
        """Load trained policy agent for experiments. Needs to know file_name. File expected in
        project working directory.
        """

        try:  # reload model
            policy_checkpoint = torch.load(f"{file_name}.pt")
            self.model.load_state_dict(policy_checkpoint["policy_state_dict"])

        except:
            raise ValueError("No agent checkpoint found.")


    def _create_models(self):
        """Placeholder for _create_models"""
        raise NotImplementedError("You need to implement a _create_models method for this child class")

    def updateState(self):
        """Placeholder for updateState"""
        raise NotImplementedError("You need to implement a updateState method for this child class")


    def _create_and_load_optimizer(
        self,
        opt_choice,
        momentum,
        ckpt_path,
        exp_name_toload,
        exp_name,
        snapshot,
        load_opt,
        wd=0.001,
        lr_dqn=0.001,
    ):
        opt_kwargs = {"lr": lr_dqn, "weight_decay": wd, "momentum": momentum}

        if opt_choice == "SGD":
            self.optimizer = torch.optim.SGD(
                self.policy_net.parameters(), **opt_kwargs
            )
        elif opt_choice =="Adam":
            self.optimizer = torch.optim.Adam(
                self.policy_net.parameters(), lr_dqn
            )
        elif opt_choice == "RMSprop":
            self.optimizer = torch.optim.RMSprop(
                self.policy_net.parameters(), lr_dqn
            )

        name = exp_name_toload if load_opt and len(exp_name_toload) > 0 else exp_name
        opt_policy_path = os.path.join(ckpt_path, name, "opt_policy_" + str(snapshot))

        if load_opt:
            print("(Opt load) Loading policy optimizer")
            self.optimizer.load_state_dict(torch.load(opt_policy_path))

        print("Policy optimizer created")

    def push_to_buffer(self, state, action, next_state, reward, terminal):
        """Saves a transition."""
        self.memory.push(state, action, next_state, reward, terminal)

    def update_target_network(self):
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

            #loss = (q_value - expected_q_value.detach()).pow(2).mean()
            loss = F.smooth_l1_loss(q_value, expected_q_value)
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.policy_error = np.append(self.policy_error, loss.detach().cpu().numpy())

    def save_models(self, file_name="policy_agent"):
        """Save trained policy agent for experiments. Needs to know file_name. File expected in
        project working directory.
        """
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            f"{file_name}.pt",
        )
class SelectionPolicyAgent(DQN):
    def __init__(self, config):
        super().__init__(config)
        self.trial_episode_scores = []
        self.avg_grad_value = None
        #State Specific Params
        self.singleton_state_variables = (
            5  # [test loss, test std, n proxy models, cluster cutoff and elapsed time]
        )
        self.state_length = int(
            config.querier.model_state_size * 5 + self.singleton_state_variables
        )  # This depends on size of dataset

        #Model Specific Params
        self.action_size = config.rl.action_size
        self.hidden_size = config.rl.hidden_size

        #Create (and load) Model and Optimizer
        self._create_models()
        if self.load:
            self._load_models()

        self._create_and_load_optimizer(**self.optimizer_param)
        self.scheduler = StepLR(self.optimizer, step_size=self.lr_step, gamma=self.lr_gamma)

    def _create_models(self):
        """Creates the Online and Target DQNs
        """
        # Query network (and target network for DQN)
        self.policy_net = MLP(self.state_length, self.action_size, self.hidden_size).to(self.device).double()
        self.target_net = MLP(self.state_length, self.action_size, self.hidden_size).to(self.device).double()
        self.target_net.load_state_dict(self.policy_net.state_dict())  # sync parameters.
        for param in self.target_net.parameters():
            param.requires_grad = False
        printRecord("Policy network has " + str(get_n_params(self.policy_net)) + " parameters.")

        # print("DQN Models created!")

    #def select_action(self):
    #    return self.action_to_map(super().select_action())

    def updateState(self, state, model):
        """
        update the model state and store it for later sampling
        :param state:
        :return:
        """
        model_state_dict = state
        previous_model_state = self.state
        # things to put into the model state
        # test loss and standard deviation between models
        self.state = torch.stack(
            (
                torch.tensor(model_state_dict["test loss"]),
                torch.tensor(model_state_dict["test std"]),
            )
        )

        # sample energies
        self.state = torch.cat(
            (self.state, torch.tensor(model_state_dict["best energies"]))
        )

        # sample uncertainties
        self.state = torch.cat(
            (self.state, torch.Tensor(model_state_dict["best uncertanties"]))
        )

        # internal dist, dataset dist, random set dist
        self.state = torch.cat(
            (self.state, torch.tensor(model_state_dict["best samples internal diff"]))
        )
        self.state = torch.cat(
            (self.state, torch.tensor(model_state_dict["best samples dataset diff"]))
        )
        self.state = torch.cat(
            (self.state, torch.tensor(model_state_dict["best samples random set diff"]))
        )

        # n proxy models,         # clustering cutoff,         # progress fraction
        singletons = torch.stack(
            (
                torch.tensor(model_state_dict["n proxy models"]),
                torch.tensor(model_state_dict["clustering cutoff"]),
                torch.tensor(model_state_dict["iter"] / model_state_dict["budget"]),
            )
        )

        self.state = torch.cat((self.state, singletons))
        self.state = self.state.to(self.device)

        self.proxyModel = model  # this should already be on correct device - passed directly from the main program

        # get data to compute distances
        # model state samples
        self.modelStateSamples = model_state_dict["best samples"]
        # training dataset
        self.trainingSamples = np.load(
            "datasets/" + self.config.dataset.oracle + ".npy", allow_pickle=True
        ).item()
        self.trainingSamples = self.trainingSamples["samples"]
        # large random sample
        numSamples = min(
            int(1e4), self.config.dataset.dict_size ** self.config.dataset.max_length // 100
        )  # either 1e4, or 1% of the sample space, whichever is smaller
        dataoracle = Oracle(self.config)
        self.randomSamples = dataoracle.initializeDataset(
            save=False, returnData=True, customSize=numSamples
        )  # get large random dataset
        self.randomSamples = self.randomSamples["samples"]

        return previous_model_state, self.state

    def action_to_map(self, action_id):
        action = torch.zeros(self.action_size, dtype=int)
        action[action_id] = 1
        return action

    def evaluateProxyModel(self, output="Average"):  # just evaluate the proxy
        return self.proxyModel.evaluate(self.randomSamples, output=output)

    def handleTraining(self):
        if len(self.memory) >= self.config.rl.min_memory and (self.episode % self.config.rl.dqn_train_frequency == 0):
            self.train()
            self.update_target_network()
            self.avg_grad_value = torch.mean(torch.stack([torch.mean(abs(param.grad)) for param in self.policy_net.parameters()]))

        elif len(self.memory) <= self.config.rl.min_memory:
            Warning('Replay Buffer not populated enough')


    def handleEndofEpisode(self, logger):

        ## Update epsilon and learning rate
        self.scheduler.step()
        self.update_epsilon()

        episode_score = self.evaluateProxyModel().mean()

        ## Logging
        avg_param_value = torch.mean(torch.stack([torch.mean(abs(param)) for param in self.policy_net.parameters()]))
        self.trial_episode_scores += [episode_score]
        last_100_avg = np.mean(self.trial_episode_scores[-100:])
        if self.log_rl_to_console:
            if self.avg_grad_value:
                print(f'E {self.episode} scored {episode_score:.2f}, avg {last_100_avg:.2f}, avg param {avg_param_value:.2f}', end = ', ')
                print(f'avg_grad {self.avg_grad_value:.2f} avg_loss {np.mean(self.policy_error):.2f}')
            else:
                print(f'E {self.episode} scored {episode_score:.2f}, avg {last_100_avg:.2f}, avg param {avg_param_value:.2f}')
        if (self.episode % self.config.rl.eval_interval == 0):
            mean_eval_score = self.evaluateProxyModel().mean()
            logger.log_metric(name='Mean Eval Score', value=mean_eval_score, step=self.episode)

        logger.log_metric(name='Learning Rate', value=self.scheduler.get_last_lr(), step=self.episode)
        logger.log_metric(name='Episode Score', value=episode_score, step=self.episode)
        logger.log_metric(name='Moving Score Average (100 eps)', value=last_100_avg, step=self.episode)
        logger.log_metric(name='Mean Gradient Value', value=self.avg_grad_value, step=self.episode)
        logger.log_metric(name='Mean Weight Value', value=avg_param_value, step=self.episode)
        logger.log_metric(name='Mean Training Replay Loss', value=np.mean(self.policy_error), step=self.episode)


class BasicAgent(DQN):

    def __init__(self, config) -> None:
        super().__init__(config)

        #Model Specific Params
        self.action_size = config.rl.action_size
        self.hidden_size = config.rl.hidden_size
        self.state_length = config.rl.state_length

        #Create (and load) Model and Optimizer
        self._create_models()
        if self.load:
            self._load_models()

        self._create_and_load_optimizer(**self.optimizer_param)
        self.scheduler = StepLR(self.optimizer, step_size=self.lr_step, gamma=self.lr_gamma)

    def _create_models(self):
        """Creates the Online and Target DQNs

        """
        # Query network (and target network for DQN)
        self.policy_net = MLP(self.state_length, self.action_size, self.hidden_size).to(self.device)
        self.target_net = MLP(self.state_length, self.action_size, self.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # sync parameters.
        for param in self.target_net.parameters():
            param.requires_grad = False

    def updateState(self, state):
        self.state=torch.tensor(state)

    def evaluate(self, env):
        """Get average reward using the current policy"""
        rewards = []
        for i, x in enumerate(range(20)):
            state = env.reset()
            done = False
            episode_score = 0
            while not done:
                if i==0:
                    env.render()
                    time.sleep(.001)
                with torch.no_grad():
                    q_values = self.policy_net(torch.Tensor(state).to(self.device))
                    action = np.argmax(q_values.cpu().numpy())
                state, reward, done, _ = env.step(action)
                episode_score += reward
            rewards += [episode_score]
        return np.mean(rewards)

def main(config):
    agent = BasicAgent(config)
    env = gym.make(config.rl.env_name)
    torch.manual_seed(config.rl.seed)
    env.seed(config.rl.seed)
    logger = Experiment(project_name=config.rl.comet.project, display_summary_level=0,)
    if config.rl.comet.tags:
        if isinstance(config.rl.comet.tags, list):
            logger.add_tags(config.rl.comet.tags)
        else:
            logger.add_tag(config.rl.comet.tags)
    hyperparams = vars(config.al)
    hyperparams.pop('comet')
    logger.log_parameters(vars(config.al))
    trial_episode_scores = []
    avg_grad_value = None
    debug_log = True

    for i_episode in range(config.rl.episodes):
        episode_score = 0
        state = env.reset()
        agent.updateState(state)
        done = False
        episode_length = 0
        while not done:
            episode_length += 1
            action = agent.select_action()
            new_state, reward, done, _ = env.step(action)
            episode_score += reward

            # save state, action, reward sequence
            agent.memory.push(state, action, new_state, reward, done)
            state = new_state
            agent.updateState(state)
        if len(agent.memory) >= config.rl.min_memory and (i_episode % config.rl.dqn_train_frequency == 0) and i_episode >= config.rl.learning_start:
            agent.policy_error = []
            agent.train()
            agent.update_target_network()
            avg_grad_value = torch.mean(torch.stack([torch.mean(abs(param.grad)) for param in agent.policy_net.parameters()]))

        avg_param_value = torch.mean(torch.stack([torch.mean(abs(param)) for param in agent.policy_net.parameters()]))

        # Update epsilon and learning rate
        agent.scheduler.step()
        agent.update_epsilon()

        ## Logging
        trial_episode_scores += [episode_score]
        last_100_avg = np.mean(trial_episode_scores[-100:])
        if debug_log:
            if avg_grad_value:
                print(f'E {i_episode} scored {episode_score:.2f}, ep_length {episode_length}, avg {last_100_avg:.2f}, avg param {avg_param_value:.2f}', end = ', ')
                print(f'avg_grad {avg_grad_value:.2f} avg_loss {np.mean(agent.policy_error):.2f}')
            else:
                print(f'E {i_episode} scored {episode_score:.2f}, ep_length {episode_length}, avg {last_100_avg:.2f}, avg param {avg_param_value:.2f}')
        if (i_episode % config.rl.eval_interval == 0):
            mean_eval_score = agent.evaluate(env)
            logger.log_metric(name='Mean Eval Score', value=mean_eval_score, step=i_episode)

        logger.log_metric(name='Learning Rate', value=agent.scheduler.get_last_lr(), step=i_episode)
        logger.log_metric(name='Episode Duration', value=episode_length, step=i_episode)
        logger.log_metric(name='Episode Score', value=episode_score, step=i_episode)
        logger.log_metric(name='Moving Score Average (100 eps)', value=last_100_avg, step=i_episode)
        logger.log_metric(name='Mean Gradient Value', value=avg_grad_value, step=i_episode)
        logger.log_metric(name='Mean Weight Value', value=avg_param_value, step=i_episode)
        logger.log_metric(name='Mean Training Replay Loss', value=np.mean(agent.policy_error), step=i_episode)


def train_toy_agent(config):
    import cProfile as profile
    import pstats
    prof = profile.Profile()
    prof.enable()
    main(config)
    stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
    prof.disable()
    # print profiling output
    stats.print_stats(30) # top 10 rows'''

