# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
from collections import deque, namedtuple
import random
import torch

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10

# Definition needed to store memory replay in pickle
Query_Transition = namedtuple(
    "Transition",
    ("model_state", "action_state", "next_model_state", "next_action_state", "reward", "terminal"),
)

Parameter_Transition = namedtuple(
    "Transition",
    ("model_state", "action", "next_model_state", "reward", "terminal"),
)


class QuerySelectionReplayMemory(object):
    """
    Class that encapsulates the experience replay buffer, the push and sampling method
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(
        self, model_state, action_state, next_model_state, next_action_state, reward, terminal
    ):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = None
        self.memory[self.position] = Query_Transition(
            model_state, action_state, next_model_state, next_action_state, reward, terminal
        )
        self.position = (self.position + 1) % self.capacity

        del model_state
        del action_state
        del next_model_state
        del next_action_state
        del terminal
        del reward

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayMemory(object):
    """
    Class that encapsulates the experience replay buffer, the push and sampling method
    """

    def __init__(self, length, device="cpu"):
        self.memory = deque(maxlen=length)
        self.device = device

    def __len__(self):
        return len(self.memory)

    def push(self, state, action, next_state, reward, terminal):
        self.memory.append(
            {
                "state": torch.tensor(state),
                "action": action,
                "next_state": torch.tensor(next_state),
                "reward": reward,
                "terminal": terminal,
            }
        )

        del state
        del action
        del next_state
        del terminal
        del reward

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        batch = random.sample(self.memory, batch_size)

        return (
            torch.stack([step["state"] for step in batch]).to(self.device),
            torch.tensor([step["action"] for step in batch]).to(self.device),
            torch.stack([step["next_state"] for step in batch]).to(self.device),
            torch.tensor([step["reward"] for step in batch]).float().to(self.device),
            torch.tensor([step["terminal"] for step in batch]).to(self.device),
        )

    def reset(self):
        self.memory.clear()
