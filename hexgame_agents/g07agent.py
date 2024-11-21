from abc import ABC, abstractmethod
from typing import Any

from attrs import define, field
from ourhexgame.ourhexenv import OurHexGame


@define
class Agent(ABC):
    env: OurHexGame = field()

    def from_file(self, file) -> "Agent":
        pass


    def to_file(self, file: str) -> None:
        pass


    @abstractmethod
    def select_action(observation, reward, termination, truncation, info) -> int:
        pass


@define
class SparseRewardAgent(Agent):

    def select_action(observation, reward, termination, truncation, info) -> int:
        pass


@define
class DenseRewardAgent(Agent):

    def select_action(observation, reward, termination, truncation, info) -> int:
        pass


class G07Agent(Agent):
    """Wrapper class for the two agents.
    """
    def __init__(self, env: OurHexGame) -> "G07Agent":
        if env.sparse_flag:
            self.agent = SparseRewardAgent(env)
        else:
            self.agent = DenseRewardAgent(env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)