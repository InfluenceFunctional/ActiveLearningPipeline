from NewActiveLearner import ActiveLearning
import torch
import numpy as np
from oracle import Oracle


class AptamerEnvironment:
    """
    Reinforcement Learning environment in the style of OpenAI Gym.
    Contains methods:
        Reset:
        Step:
        _get_obs:

    """

    def __init__(self, config, logger, workdir):
        self.config = config
        self.logger = logger
        self.device = config.device
        self.workdir = workdir
        self.reset()
        # self.terminal = 1 if self.pipeIter == (self.config.al.n_iter - 1)
        # self.activeLearner.saveOutputs()  # save pipeline outputs

    def reset(self):
        self.activeLearner = ActiveLearning(self.config, self.logger, workdir=self.workdir)
        self.pipeIter = 0
        self.done = 0
        return self._get_obs(), self.activeLearner.model

    def step(self, action):
        """Returns State and Reward"""
        self.activeLearner.iterate(action, self.done)
        self.pipeIter += 1
        self.done = 1 if self.pipeIter == (self.config.al.n_iter - 1) else 0
        return (
            self._get_obs(),
            self.activeLearner.model_state_reward,
            self.done,
            self.activeLearner.model,
        )

    def _get_obs(self):
        return self._calculateModelState(self.activeLearner.stateDict)

    def _calculateModelState(self, model_state_dict):
        """
        update the model state and store it for later sampling
        :param state:
        :return:
        """
        # things to put into the model state
        # test loss and standard deviation between models
        state = torch.stack(
            (
                torch.tensor(model_state_dict["test loss"]),
                torch.tensor(model_state_dict["test std"]),
            )
        )

        # sample energies
        state = torch.cat((state, torch.tensor(model_state_dict["best energies"])))

        # sample uncertainties
        state = torch.cat((state, torch.Tensor(model_state_dict["best uncertanties"])))

        # internal dist, dataset dist, random set dist
        state = torch.cat((state, torch.tensor(model_state_dict["best samples internal diff"])))
        state = torch.cat((state, torch.tensor(model_state_dict["best samples dataset diff"])))
        state = torch.cat((state, torch.tensor(model_state_dict["best samples random set diff"])))

        # n proxy models,         # clustering cutoff,         # progress fraction
        singletons = torch.stack(
            (
                torch.tensor(model_state_dict["n proxy models"]),
                torch.tensor(model_state_dict["clustering cutoff"]),
                torch.tensor(model_state_dict["iter"] / model_state_dict["budget"]),
            )
        )

        state = torch.cat((state, singletons)).to(self.device)
        return state
