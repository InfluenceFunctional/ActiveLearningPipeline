import gym
import torch
import numpy as np
from torch import nn
import os
import random
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

        # RL Properties
        self.epsilon_end = config.al.eps_end
        self.epsilon_decay = config.al.eps_decay
        self.epsilon = config.al.eps_start
        self.gamma = config.al.gamma
        self.tau = config.al.tau
        self.dqn_train_frequency = config.al.dqn_train_frequency
        self.memory = ReplayMemory(config.al.memory_limit, device=self.device)
        self.update_type = config.al.update_type
        self.dqn_epochs = config.al.dqn_epochs
        self.batch_size = config.al.dqn_batch_size

        # Agent Properties
        self.episode = 0
        self.policy_error = []
        self.step = 0
        self.state = None

        # Optimizer and Training Properties
        self.alpha = config.al.alpha
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
        self.lr_step = config.al.lr_step
        self.lr_gamma = config.al.lr_gamma

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

        #Model Specific Params
        self.action_size = config.al.action_size
        self.hidden_size = config.al.hidden_size
        self.state_length = config.al.state_length

        #State Specific Params
        self.singleton_state_variables = (
            5  # [test loss, test std, n proxy models, cluster cutoff and elapsed time]
        )
        self.state_dataset_size = int(
            config.querier.model_state_size * 5 + self.singleton_state_variables
        )  # This depends on size of dataset
        self.model_state = None

        # Create Model and Optimizer
        self._create_models()
        if self.load:
            self._load_models()
        self._create_and_load_optimizer(**self.optimizer_param)

    def _create_models(self):
        """Creates the Online and Target DQNs
        """
        # Query network (and target network for DQN)
        self.policy_net = MLP(self.state_length, self.action_size, self.hidden_size).to(self.device)
        self.target_net = MLP(self.state_length, self.action_size, self.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # sync parameters.
        for param in self.target_net.parameters():
            param.requires_grad = False
        printRecord("Policy network has " + str(get_n_params(self.policy_net)) + " parameters.")

        # print("DQN Models created!")
    def updateState(self, model_state, model):
        """
        update the model state and store it for later sampling
        :param model_state:
        :return:
        """
        model_state_dict = model_state
        previous_model_state = self.model_state
        # things to put into the model state
        # test loss and standard deviation between models
        self.model_state = torch.stack(
            (
                torch.tensor(model_state_dict["test loss"]),
                torch.tensor(model_state_dict["test std"]),
            )
        )

        # sample energies
        self.model_state = torch.cat(
            (self.model_state, torch.tensor(model_state_dict["best energies"]))
        )

        # sample uncertainties
        self.model_state = torch.cat(
            (self.model_state, torch.Tensor(model_state_dict["best uncertanties"]))
        )

        # internal dist, dataset dist, random set dist
        self.model_state = torch.cat(
            (self.model_state, torch.tensor(model_state_dict["best samples internal diff"]))
        )
        self.model_state = torch.cat(
            (self.model_state, torch.tensor(model_state_dict["best samples dataset diff"]))
        )
        self.model_state = torch.cat(
            (self.model_state, torch.tensor(model_state_dict["best samples random set diff"]))
        )

        # n proxy models,         # clustering cutoff,         # progress fraction
        singletons = torch.stack(
            (
                torch.tensor(model_state_dict["n proxy models"]),
                torch.tensor(model_state_dict["clustering cutoff"]),
                torch.tensor(model_state_dict["iter"] / model_state_dict["budget"]),
            )
        )

        self.model_state = torch.cat((self.model_state, singletons))
        self.model_state = self.model_state.to(self.device)

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

        return previous_model_state, self.model_state

    def evaluate(self, sample, output="Average"):  # just evaluate the proxy
        return self.proxyModel.evaluate(sample, output=output)


class BasicAgent(DQN):

    def __init__(self, config) -> None:
        super().__init__(config)

        #Model Specific Params
        self.action_size = config.al.action_size
        self.hidden_size = config.al.hidden_size
        self.state_length = config.al.state_length

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

def main(config):
    agent = BasicAgent(config)
    env = gym.make(config.al.env_name)
    torch.manual_seed(config.al.seed)
    env.seed(config.al.seed)
    logger = Experiment(project_name=config.al.comet.project, display_summary_level=0,)
    if config.al.comet.tags:
        if isinstance(config.al.comet.tags, list):
            logger.add_tags(config.al.comet.tags)
        else:
            logger.add_tag(config.al.comet.tags)
    hyperparams = vars(config.al)
    hyperparams.pop('comet')
    logger.log_parameters(vars(config.al))
    trial_episode_scores = []
    avg_grad_value = None
    debug_log = True
    i = 0
    for i_episode in range(config.al.episodes):
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
        if len(agent.memory) > config.al.min_memory and (i_episode % config.al.dqn_train_frequency == 0):
            i+=30
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
                print(f'E {i_episode} scored {episode_score}, avg {last_100_avg:.2f}, avg param {avg_param_value:.2f}', end = ', ')
                print(f'avg_grad {avg_grad_value:.2f} avg_loss {np.mean(agent.policy_error):.2f}')
            else:
                print(f'E {i_episode} scored {episode_score}, avg {last_100_avg:.2f}, avg param {avg_param_value:.2f}')

        logger.log_metric(name='Learning Rate', value=agent.scheduler.get_last_lr(), step=i_episode)
        logger.log_metric(name='Episode Score', value=episode_score, step=i_episode)
        logger.log_metric(name='Moving Score Average (100 eps)', value=last_100_avg, step=i_episode)
        logger.log_metric(name='Mean Gradient Value', value=avg_grad_value, step=i_episode)
        logger.log_metric(name='Mean Weight Value', value=avg_param_value, step=i_episode)
        logger.log_metric(name='Mean Training Replay Loss', value=np.mean(agent.policy_error), step=i_episode)

    print(i)

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

