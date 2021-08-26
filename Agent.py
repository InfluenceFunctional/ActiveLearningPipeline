# This code is the modified version of code from
# ksenia-konyushkova/intelligent_annotation_dialogs/exp1_IAD_RL.ipynb

from RLAptamer.models.query_network import QueryNetworkDQN
import numpy as np
import os
import random
import torch
import math
import torch.functional as F


class DQN:
    """The DQN class that learns a RL policy.


    Attributes:
        policy_net: An object of class QueryNetworkDQN that is used for q-value prediction
        target_net: An object of class QueryNetworkDQN that is a lagging copy of estimator

    """

    def __init__(self, exp_name: str, lr: float, target_copy_factor: float):
        """Inits the DQN object.

        Args:
            experiment_dir: A string with parth to the folder where to save the agent and training data.
            lr: A float with a learning rate for Adam optimiser.
            batch_size: An integer indicating the size of a batch to be sampled from replay buffer for estimator update.
            target_copy_factor: A float used for updates of target_estimator,
                with a rule (1-target_copy_factor)*target_estimator weights
                + target_copy_factor*estimator

        """
        self.exp_name = exp_name
        self.load = False
        self.state_dataset_size = 30  # This depends on size of dataset V
        self.action_state_length = 3
        self.device = "cuda:0"

        # Magic Hyperparameters for Greedy Sampling in Action Selection
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 10
        self.rl_pool = 100  # Size of Unlabelled Dataset aka Number of actions

        self.optimizer_param = {
            "opt_choice": "SGD",
            "momentum": 0.95,
            "ckpt_path": "./checkpoints",
            "exp_name_toload": None,
            "exp_name": self.exp_name,
            "snapshot": None,
            "load_opt": self.load,
        }

        self.opt_choice = "SGD"
        self.momentum = 0.95
        if self.load:
            self._load_models()
        else:
            self._create_models()

        self._create_and_load_optimizer(**self.optimizer_param)

    def _load_models(self):
        """Load model weights.
        :param load_weights: (bool) True if segmentation network is loaded from pretrained weights in 'exp_name_toload'
        :param exp_name_toload: (str) Folder where to find pre-trained segmentation network's weights.
        :param snapshot: (str) Name of the checkpoint.
        :param exp_name: (str) Experiment name.
        :param ckpt_path: (str) Checkpoint name.
        :param checkpointer: (bool) If True, load weights from the same folder.
        :param exp_name_toload_rl: (str) Folder where to find trained weights for the query network (DQN). Used to test
        query network.
        :param test: (bool) If True  and there exists a checkpoint in 'exp_name_toload_rl', we will load checkpoint for trained query network (DQN).
        """

    def _create_models(self):
        """Creates the Online and Target DQNs

        """
        # Query network (and target network for DQN)
        self.policy_net = QueryNetworkDQN(
            model_state_length=self.state_dataset_size,
            action_state_length=self.action_state_length,
            bias_average=1,  # TODO Figure out the query network bias trick from 2018 paper.
        ).to(device=self.device)
        self.target_net = QueryNetworkDQN(
            model_state_length=self.state_dataset_size,
            action_state_length=self.action_state_length,
            bias_average=1,  # TODO Figure out the query network bias trick from 2018 paper.
        ).to(device=self.device)

        print("Policy network has " + str(self.count_parameters(self.policy_net)) + " parameters.")

        print("DQN Models created!")

    def count_parameters(net: torch.nn.Module) -> int:
        model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def create_and_load_optimizer(
        self,
        opt_choice,
        momentum,
        ckpt_path,
        exp_name_toload,
        exp_name,
        snapshot,
        load_opt,
        wd=0.001,
        lr_dqn=0.0001,
    ):
        opt_kwargs = {"lr": lr_dqn, "weight_decay": wd, "momentum": momentum}

        if opt_choice == "SGD":
            self.optimizer = torch.optim.SGD(
                params=filter(lambda p: p.requires_grad, self.policy_net.parameters()), **opt_kwargs
            )
        elif opt_choice == "RMSprop":
            self.optimizer = torch.optim.RMSprop(
                params=filter(lambda p: p.requires_grad, self.policy_net.parameters()), lr=lr_dqn
            )

        name = exp_name_toload if load_opt and len(exp_name_toload) > 0 else exp_name
        opt_policy_path = os.path.join(ckpt_path, name, "opt_policy_" + snapshot)

        if load_opt:
            print("(Opt load) Loading policy optimizer")
            self.optimizer.load_state_dict(torch.load(opt_policy_path))

        print("Policy optimizer created")

    # TODO Not sure if this should return an int or a Tensor...
    def select_action(
        self,
        model_state: torch.Tensor,
        action_state: torch.Tensor,
        steps_done: int,
        test: bool = False,
    ) -> torch.Tensor:
        """Get the best action in a state.

        This function returns the best action according to
        Q-function estimator: the action with the highest
        expected return in a given classification state
        among all available action with given action states.

        :param model_state: (torch.Variable) Torch tensor containing the model state representation.
        :param action_state: (torch.Variable) Torch tensor containing the action state representations.
        :param steps_done: (int) Number of aptamers labeled so far.
        :param test: (bool) Whether we are testing the DQN or training it. Disables greedy-epsilon when True.

        :return: Action (index of Sequence to Label)
        """

        self.policy_net.eval()
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1.0 * steps_done / self.EPS_DECAY
        )
        q_val_ = []
        if sample > eps_threshold or test:
            print("Action selected with DQN!")
            with torch.no_grad():
                # Get Q-values for every action
                q_val_ = [
                    self.policy_net(model_state, action_i_state) for action_i_state in action_state
                ]

                action = torch.argmax(torch.stack(q_val_))
                del q_val_
        else:
            action = torch.Variable(
                torch.Tensor(
                    [np.random.choice(range(self.rl_pool), action_state.size()[0], replace=True)]
                )
                .type(torch.LongTensor)
                .view(-1)
            ).to(self.device)

        return action

    def train(self, memory_batch, BATCH_SIZE=32, GAMMA=0.999, dqn_epochs=1):
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
        if len(memory_batch) < BATCH_SIZE:
            return
        print("Optimize model...")
        print(len(memory_batch))
        self.policy_net.train()
        loss_item = 0
        for ep in range(dqn_epochs):
            self.optimizer.zero_grad()
            transitions = memory_batch.sample(BATCH_SIZE)
            expected_q_values = []
            for transition in transitions:
                # Predict q-value function value for all available actions in transition
                action_i = self.select_action(
                    transition.model_state, transition.action_state, test=False
                )
                with torch.autograd.set_detect_anomaly(True):
                    self.policy_net.train()
                    q_policy = self.policy_net(
                        transition.model_state.detach(), transition.action_state[action_i].detach()
                    )
                    q_target = self.target_net(
                        transition.model_state.detach(), transition.action_state[action_i].detach()
                    )

                    # Compute the expected Q values
                    expected_q_values = (q_target * GAMMA) + transition.reward

                    # Compute MSE loss
                    loss = F.mse_loss(q_policy, expected_q_values)
                    loss_item += loss.item()
                    loss.backward()
            self.optimizer.step()

            del loss
            del transitions
