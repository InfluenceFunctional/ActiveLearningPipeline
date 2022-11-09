from NewActiveLearner import ActiveLearning
import torch
import os


class AptamerEnvironment:
    """
    Reinforcement Learning environment in the style of OpenAI Gym.
    Contains methods:
        Reset:
        Step:
        _get_obs:

    """

    def __init__(self, config, logger=None, rundir="/Runs"):
        self.config = config
        self.logger = logger
        self.device = config.device
        self.rundir = rundir
        self.data_to_log = []  # ["Model_State_Score"]
        # Model_State_Score, Model_State_Calculation, Proxy_Evaluation_Loss
        # Query_Energies, Proxy_Train_Loss, Sample_Energies
        # self.reset(0)
        # self.terminal = 1 if self.pipeIter == (self.config.al.n_iter - 1)
        # self.activeLearner.saveOutputs()  # save pipeline outputs

    def reset(self, episode):
        episode_path = f"{self.rundir}/episode{episode}/"
        if not os.path.exists(episode_path):
            os.mkdir(episode_path)
            os.mkdir(f"{episode_path}/datasets")
            os.mkdir(f"{episode_path}/ckpts")
        self.config.workdir = f"{self.rundir}/episode{episode}/"
        self.activeLearner = ActiveLearning(
            self.config, logger=self.logger, data_to_log=self.data_to_log, episode=episode
        )
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
            self._get_info(),
        )

    def _get_obs(self):
        return self._calculateModelState(self.activeLearner.stateDict)

    def _get_info(self):
        return {
            "model_state_cumul_score": self.activeLearner.model_state_cumulative_score,
            "dataset_cumul_score": self.activeLearner.dataset_cumulative_score,
        }

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
                # torch.tensor(model_state_dict["n proxy models"]),
                # torch.tensor(model_state_dict["clustering cutoff"]),
                torch.tensor(model_state_dict["iter"] / model_state_dict["budget"]),
            )
        )

        state = torch.cat((state, singletons)).to(self.device)
        return state
