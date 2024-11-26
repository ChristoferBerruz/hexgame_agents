from typing import Any

from attrs import define
from ourhexgame.ourhexenv import OurHexGame

from hexgame_agents.protocols import Agent


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