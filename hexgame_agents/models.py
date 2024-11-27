import torch
from typing import Tuple, List

from contextlib import contextmanager

import hexgame_agents.game_utils as gu

from ourhexgame.ourhexenv import OurHexGame
import numpy as np


from attrs import define, field
from functools import wraps

from torchvision.transforms import Normalize
from hexgame_agents.protocols import Agent
import pickle


normalize_tf = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


def board_into_tensor(board: np.ndarray) -> torch.Tensor:
    """Convert the board into a tensor.

    Args:
        board (np.ndarray): The board.

    Returns:
        torch.Tensor: The board as a tensor.
    """
    # Generate three channels, one for each player
    # plus a channel of ones to represent the empty cells.
    # 0: empty, 1: player_1, 2: player_2
    assert board.shape == gu.BOARD_SHAPE, f"Board shape is {board.shape}, expected {gu.BOARD_SHAPE}"
    player_1 = np.where(board == 1, 1, 0)
    player_2 = np.where(board == 2, 1, 0)
    empty = np.where(board == 0, 1, 0)
    board = np.stack([empty, player_1, player_2], axis=0)
    board_tensor = torch.from_numpy(board).to(dtype=torch.float32)
    return normalize_tf(board_tensor)


def board_adapter(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # check the first argument
        if args[0].shape == gu.BOARD_SHAPE:
            args = (board_into_tensor(args[0]),) + args[1:]
        return func(self, *args, **kwargs)
    return wrapper


class CategoricalMasked(torch.distributions.Categorical):

    def __init__(self, probs: torch.Tensor, mask: torch.Tensor):
        self.batch, self.nb_action = probs.size()
        self.tensor_mask = torch.stack([mask.bool()]*self.batch, dim=0)
        self.all_zeros = torch.zeros_like(probs)
        probs = torch.where(self.tensor_mask, probs, self.all_zeros)
        super(CategoricalMasked, self).__init__(probs=probs)

    def entropy(self):
        # Elementwise multiplication
        p_log_p = torch.mul(self.logits, self.probs)
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.tensor_mask,
            p_log_p,
            self.all_zeros
        )
        # entropy is simply the sum of the negative log probabilities
        # along the last dimension
        return -p_log_p.sum(dim=-1)


class BaseNN(torch.nn.Module):
    """Base class for neural networks.
    """

    def __init__(self):
        super(BaseNN, self).__init__()
    
    @contextmanager
    def eval_mode(self):
        """Context manager to set the model to evaluation mode.
        """
        try:
            self.eval()
            with torch.no_grad():
                yield
        finally:
            self.train()


    def load(self, model_path: str, mode: str = "eval"):
        """Load the model from a file.

        Args:
            model_path (str): Path to the model file.
            mode (str): Mode to set the model. Either "eval" or "train".
        """
        self.load_state_dict(torch.load(model_path))
        if mode == "eval":
            self.eval()
        else:
            self.train()


    def save(self, model_path: str):
        """Save the model to a file. Use the state_dict instead
        of pickle to avoid compatibility issues.
        """
        torch.save(self.state_dict(), model_path)



class CNNApproximator(BaseNN):
    """A Convolutional Neural network approximator.
    3 CNN layers, followed by two fully connected layers.
    """
    def __init__(self,
            in_channels: int = 3,
            board_shape: Tuple[int, int] = gu.BOARD_SHAPE,
            first_fc_units: int = 128,
        ):
        super(CNNApproximator, self).__init__()
        self.board_shape = board_shape
        self.height, self.width = board_shape
        self.first_cnn_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.second_cnn_block = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, first_fc_units),
            torch.nn.ReLU(),
        )
        self.output_dim = first_fc_units


    def forward(self, x):
        if len(x.shape) == 3:
            # Handle the case when the input is a single board
            x = x.unsqueeze(0)
        x = self.first_cnn_block(x)
        x = self.second_cnn_block(x)
        # flatten the tensor
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ActorCriticNN(BaseNN):

    def __init__(self):
        super(ActorCriticNN, self).__init__()
        self.actor_cnn = CNNApproximator()
        self.critic_cnn = CNNApproximator()
        self.actor = torch.nn.Sequential(
            self.actor_cnn,
            torch.nn.Linear(self.actor_cnn.output_dim, gu.ACTION_SPACE),
            torch.nn.Softmax(dim=-1),
        )
        self.critic = torch.nn.Sequential(
            self.critic_cnn,
            torch.nn.Linear(self.critic_cnn.output_dim, 1),
        )

    @board_adapter
    def act(self, state: np.ndarray, mask: torch.Tensor):
        probs = self.actor(state)
        distribution = CategoricalMasked(probs, mask)
        action = distribution.sample()
        action_log_prob = distribution.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_log_prob.detach(), state_val.detach()
    
    @board_adapter
    def evaluate(self, state: np.ndarray, action: torch.Tensor):
        probs = self.actor(state)
        distribution = torch.distributions.Categorical(probs)
        action_log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        state_val = self.critic(state)
        return action_log_prob, state_val, entropy


@define
class PPOActionRecord:
    state: torch.Tensor
    action: torch.Tensor
    action_log_prob: torch.Tensor
    state_val: torch.Tensor
    reward: float
    done: bool = False
    

@define
class PPOAgent(Agent):
    env: OurHexGame
    nn: ActorCriticNN
    device: str = "cpu"

    def to(self, device: str):
        self.device = device
        self.nn.to(device)
        return self
    
    @board_adapter
    def _select_action(self, state, mask: np.ndarray, nn: ActorCriticNN):
        with nn.eval_mode():
            state = state.to(self.device)
            mask_t = torch.from_numpy(mask).to(
                dtype=torch.float32).to(self.device)
            action, action_logprob, state_val = nn.act(state, mask_t)
        return action, action_logprob, state_val, state
    
    def select_action(self, observation, reward, termination, truncation, info) -> int:
        board = observation["observation"]
        mask = info["action_mask"]
        action, *_ = self._select_action(board, mask, nn=self.nn)
        return action.item()
    
    @classmethod
    def from_file(cls, model_path: str, env: OurHexGame):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        actor_critic = ActorCriticNN()
        with open(model_path, "rb") as fp:
            checkpoint_dict = pickle.load(fp)
        state_dict = checkpoint_dict["nn_state_dict"]
        actor_critic.load_state_dict(state_dict)
        _instance = cls(env, actor_critic, device=device)
        _instance.to(device)
        return _instance
    

@define(slots=False)
class TrainablePPOAgent(PPOAgent):
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    clip_eps: float = 0.2
    value_loss_weight: float = 0.5
    entropy_weight: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    optimizer: torch.optim.Optimizer = field(init=False)
    loss: torch.nn.Module = torch.nn.MSELoss()
    buffer: List[PPOActionRecord] = field(factory=list)

    def __attrs_post_init__(self):
        self.optimizer = torch.optim.Adam([
            {'params': self.nn.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.nn.critic.parameters(), 'lr': self.lr_critic}
        ])
        self.old_nn = ActorCriticNN()
        self.old_nn.load_state_dict(self.nn.state_dict())
        self.to(self.device)

    def select_action(self, observation, reward, termination, truncation, info) -> int:
        board = observation["observation"]
        mask = info["action_mask"]
        action, action_logprob, state_val, state = self._select_action(board, mask, nn=self.old_nn)
        record = PPOActionRecord(state, action, action_logprob, state_val, 0.0)
        self.buffer.append(record)
        return action.item()
    
    def get_discounted_rewards(self) -> torch.Tensor:
        """Get MC discounted rewards.
        """
        rewards = []
        discounted_reward = 0
        _rewards = [record.reward for record in self.buffer]
        _dones = [record.done for record in self.buffer]
        for reward, is_terminal in zip(reversed(_rewards), reversed(_dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        return torch.tensor(rewards, dtype=torch.float32)
    
    def optimize_policy(self, epochs: int):
        if not self.buffer:
            return
        _states = torch.stack([record.state for record in self.buffer])
        _actions = torch.stack([record.action for record in self.buffer])
        _old_logprobs = torch.stack([record.action_log_prob for record in self.buffer])
        _old_state_values = torch.stack([record.state_val for record in self.buffer])
        old_states = torch.squeeze(_states).detach().to(self.device)
        old_actions = torch.squeeze(_actions).detach().to(self.device)
        old_logprobs = torch.squeeze(_old_logprobs).detach().to(self.device)
        old_state_values = torch.squeeze(_old_state_values).detach().to(self.device)

        # get rewards and normalize them
        rewards = self.get_discounted_rewards().to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        for epoch in range(epochs):
            self._epoch_iter(epoch, old_states, old_actions, old_logprobs, rewards, advantages)

        # Copy new weights into old policy
        self.old_nn.load_state_dict(self.nn.state_dict())

        self.buffer.clear()

    def _epoch_iter(
            self,
            epoch: int,
            states: torch.Tensor,
            actions: torch.Tensor,
            old_log_probs: torch.Tensor,
            rewards: torch.Tensor,
            advantages: torch.Tensor
        ):
        # Evaluating actions and values
        logprobs, state_values, dist_entropy = self.nn.evaluate(
            states, actions)

        # match state_values tensor dimensions with rewards tensor
        state_values = torch.squeeze(state_values)

        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = torch.exp(logprobs - old_log_probs.detach())

        # Finding Surrogate Loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.clip_eps,
                            1+self.clip_eps) * advantages

        # final loss of clipped objective PPO
        loss = -torch.min(surr1, surr2) + 0.5 * \
            self.loss(state_values, rewards) - 0.01 * dist_entropy

        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
    
    
    def save(self, model_path: str):
        self.nn.save(model_path)
    
    def load(self, model_path: str):
        self.nn.load(model_path)
    
    def act(self, state: np.ndarray, mask: torch.Tensor):
        return self.nn.act(state, mask=mask)
    
    def evaluate(self, state: np.ndarray, action: torch.Tensor):
        return self.nn.evaluate(state, action)
    
    def to(self, device: str):
        self.device = device
        self.nn.to(device)
        self.old_nn.to(device)
        return self