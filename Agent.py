# This code is the modified version of code from
# ksenia-konyushkova/intelligent_annotation_dialogs/exp1_IAD_RL.ipynb

import warnings
from scipy.sparse.construct import rand
from RLmodels import QueryNetworkDQN, ParameterUpdateDQN, MLP
import numpy as np
import os
import random
import torch
from tqdm import tqdm
import math
import torch.functional as F

#from torch.utils.tensorboard import SummaryWriter
from utils import *
from replay_buffer import QuerySelectionReplayMemory, ParameterUpdateReplayMemory
from oracle import Oracle
from datetime import datetime


class DQN:
    """The DQN class that learns a RL policy.


    Attributes:
        policy_net: An object of class QueryNetworkDQN that is used for q-value prediction
        target_net: An object of class QueryNetworkDQN that is a lagging copy of estimator

    """

    def __init__(self, config):
        """Inits the DQN object.

        Args:
            experiment_dir: A string with parth to the folder where to save the agent and training data.
            lr: A float with a learning rate for Adam optimiser.
            batch_size: An integer indicating the size of a batch to be sampled from replay buffer for estimator update.
            target_copy_factor: A float used for updates of target_estimator,
                with a rule (1-target_copy_factor)*target_estimator weights
                + target_copy_factor*estimator

        """

        torch.manual_seed(config.seeds.model)
        #self.writer = SummaryWriter(log_dir=f"C:/Users/Danny/Desktop/ActiveLearningPipeline/logs/exp{datetime.now().strftime('D%Y-%m-%dT%H-%M-%S')}")
        self.config = config
        self.device = config.device
        self.target_sync_interval = config.al.target_sync_interval
        self.episode = 0
        self.policy_error = []
        self.step = 0

        # Magic Hyperparameters for Greedy Sampling in Action Selection
        self.epsilon_start = config.al.eps_start
        self.epsilon_end = config.al.eps_end
        self.epsilon_decay = config.al.eps_decay
        self.alpha = config.al.alpha
        self.gamma = config.al.gamma
        self.tau = config.al.tau


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

    def updateState(self):
        raise NotImplementedError("You need to implement a updateState method for this child class")
        pass

    def update_target_network(self):
        """Default Target Network Update is interval based"""
        if self.episode % self.target_sync_interval == 0:
            print(f"Updating Target Network Parameters.")
            self.target_net.load_state_dict(self.policy_net.state_dict())

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

    def getAction(self):
        with torch.no_grad():
            q_values = self.policy_net(self.state)

        if random.random() > self.epsilon or self.config.al.mode == "deploy":
            action = np.argmax(q_values.cpu().numpy())

        else:
            action = np.random.randint(0, self.action_size)

        return action

    def count_parameters(
        net: torch.nn.Module,
    ) -> int:  # TODO - delete - MK didn't work for whatever reason
        model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def action_to_map(self, action_id):
        action = torch.zeros(self.action_size, dtype=int)
        action[action_id] = 1
        return action


    def train(self, BATCH_SIZE=32, GAMMA=0.99, dqn_epochs=20, comet=None, iteration=None):
        """Train a q-function estimator on a minibatch.

        Train estimator on minibatch, partially copy
        optimised parameters to target_estimator.
        We use double DQN that means that estimator is
        used to select the best action, but target_estimator
        predicts the q-value.

        :(ReplayMemory) memory: Experience replay buffer
        :param Transition: definition of the experience replay tuple
        :param BATCH_SIZE: (int) Batch size to sample from the experience replay
        :param GAMMA: (float) Discount factor
        :param dqn_epochs: (int) Number of epochs to train the DQN
        """
        # Code adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

        for _ in range(dqn_epochs):

            # Get Batch
            state_batch, action_batch, next_states_batch, reward_batch, terminal_batch  = self.memory.sample(BATCH_SIZE)

            # Get Predicted Q-values at s_t
            q_values = self.policy_net(state_batch)

            # Get Target q-value function value for the action at s_t+1
            # that yields the highest discounted return
            next_q_values = self.policy_net(next_states_batch)
            next_q_state_values = self.target_net(next_states_batch)
            next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
            q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

            # Compute the Target Q values (No future return if terminal).
            # Use Bellman Equation which essentially states that sum of r_t+1 and the max_q_value at time t+1
            # is the target/expected value of the Q-function at time t.
            expected_q_value = reward_batch + GAMMA * next_q_value * (1 - terminal_batch*1)

            # Compute MSE loss Comparing Q(s) obtained from Online Policy to
            # target Q value (Q'(s)) obtained from Target Network + Bellman Equation
            #loss = torch.nn.functional.mse_loss(q_value, expected_q_value)
            loss = (q_value - expected_q_value.detach()).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            self.policy_error = np.append(self.policy_error, loss.detach().cpu().numpy())
            # Optional Comet Logging
            if comet:
                comet.log_metric('RL Error', loss, step=self.episode)


class ParameterUpdateAgent(DQN):
    def __init__(self, config):
        self.load = False if config.querier.model_ckpt is None else True
        self.exp_name = "learned_"
        self.action_size = config.al.action_state_size
        super().__init__(config)
        self.memory = ParameterUpdateReplayMemory(self.config.al.memory_limit)
        self.singleton_state_variables = (
            5  # [test loss, test std, n proxy models, cluster cutoff and elapsed time]
        )
        self.state_dataset_size = int(
            config.querier.model_state_size * 5 + self.singleton_state_variables
        )  # This depends on size of dataset V
        self.model_state_latent_dimension = config.al.q_network_width  # latent dim of model state
        self.model_state = None

        self._create_models()
        if self.load:
            self._load_models()

        self._create_and_load_optimizer(**self.optimizer_param)

    def _create_models(self):
        """Creates the Online and Target DQNs

        """
        # Query network (and target network for DQN)
        self.policy_net = ParameterUpdateDQN(
            model_state_length=self.state_dataset_size,
            model_state_latent_dimension=self.model_state_latent_dimension,
            action_size=self.action_size,
            bias_average=1,  # TODO Figure out the query network bias trick from 2018 paper.
        ).to(self.device)
        self.target_net = ParameterUpdateDQN(
            model_state_length=self.state_dataset_size,
            model_state_latent_dimension=self.model_state_latent_dimension,
            action_size=self.action_size,
            bias_average=1,  # TODO Figure out the query network bias trick from 2018 paper. #MK what are we biasing
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # sync parameters.
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



    # def test_rl(self):
    # Oracle
    # Given an input, x, gives the right answer y. Will use Toy Models already in oracle.py.

    # Proxy
    # Given an input, x, gives an estimate y'

    # Querier

    # Sampler
    # Selects x's from a list of x's obtained using MCMC of the Proxy Model. Scores Samples based on uncertainty and energy

    # RL Agent
    # Given a proxy model state, z which is a vector, output an action vector which will change the Uncertainty/Enery ratio for sample scoring


class BasicAgent(DQN):
    def __init__(self, config):

        self.policy_error = []
        self.load = False
        self.exp_name = "test"
        self.action_size = config.al.action_size
        self.hidden_size = config.al.hidden_size
        self.state_length = config.al.state_length
        self.agent_memory_limit = config.al.memory_limit
        super().__init__(config)
        self.epsilon = self.epsilon_start
        self.memory = ParameterUpdateReplayMemory(self.config.al.memory_limit, device=self.device)
        self.state = None
        self._create_models()
        if self.load:
            self._load_models()
        self._create_and_load_optimizer(**self.optimizer_param)
        self.activation = {}
        #self.policy_net.fc2.register_forward_hook(self.get_activation('fc2'))
        self.update_type = 'soft'

    def update_target_network(self):
        if self.update_type == 'hard':
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)


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

    def updateState(self, state):

        self.state=torch.tensor(state).to(self.device)

    def get_activation(self,name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook


def plot_individual_trial(trial):
    plt.plot(trial)
    plt.ylabel('Steps in Episode')
    plt.xlabel('Episode')
    plt.title('Double DQN CartPole v-0 Steps in Select Trial')
    plt.show()


def train_toy_agent(config):

    import gym
    #torch.autograd.set_detect_anomaly(True)
    replay_batch_size = config.al.dqn_batch_size
    avg_grad_value = None
    min_memory_for_replay = config.al.min_memory
    agent = BasicAgent(config)
    env = gym.make(config.al.gym_env)
    env.seed(42)
    torch.manual_seed(42)
    MAX_STEPS_PER_EPISODE = 200
    trial_episode_scores = []

    update_step = 1
    dqn_epochs = 10

    for i_episode in range(config.al.episodes):
        episode_score = 0
        state = env.reset()
        done = False

        agent.updateState(state)
        agent.policy_error = []

        #print(f'Start---X-coord:{state[0]}, Velocity:{state[1]}')
        #for t in range(MAX_STEPS_PER_EPISODE):
        while not done:
            #env.render()
            action = agent.getAction()
            next_state, reward, done, _ = env.step(action)
            episode_score += reward
            agent.updateState(next_state)
            #print(f'[{t}]---X-coord:{next_state[0]}, Velocity:{next_state[1]}, Done:{done}')
            agent.push_to_buffer(state, action, next_state, reward, done)

        # Train Network every
        if len(agent.memory) > min_memory_for_replay and (i_episode % update_step == 0):
            agent.train(BATCH_SIZE=replay_batch_size,  dqn_epochs=dqn_epochs, iteration=i_episode)
            agent.update_target_network()
            avg_grad_value = torch.mean(torch.stack([torch.mean(abs(param.grad)) for param in agent.policy_net.parameters()]))

        avg_param_value = torch.mean(torch.stack([torch.mean(abs(param)) for param in agent.policy_net.parameters()]))

        agent.update_epsilon()
        trial_episode_scores += [episode_score]
        last_100_avg = np.mean(trial_episode_scores[-100:])
        if avg_grad_value:
            print(f'E {i_episode} scored {episode_score}, avg {last_100_avg:.2f}, avg param {avg_param_value:.2f}', end = ', ')
            print(f'avg_grad {avg_grad_value:.2f} avg_loss {np.mean(agent.policy_error):.2f}')
        else:
            print(f'E {i_episode} scored {episode_score}, avg {last_100_avg:.2f}, avg param {avg_param_value:.2f}')

        if len(trial_episode_scores) >= 100 and last_100_avg >= 195.0:
            print (f'Trial 1 solved in {i_episode-100} episodes!')
            break

    env.close()
    #plot_individual_trial(trial_episode_scores)
    #plot_individual_trial(agent.policy_error)
