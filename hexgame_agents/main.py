import click
from hexgame_agents.models import TrainablePPOAgent, ActorCriticNN, PPOAgent

from ourhexgame.ourhexenv import OurHexGame
import torch

import random

import time
import ray
from ray import tune
from ray import train
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from pathlib import Path

from typing import Tuple


@click.group()
def cli():
    pass


class SelfPlayTrainable(ray.tune.Trainable):

    def setup(self, config: dict):
        self.lr_actor = config["lr_actor"]
        self.lr_critic = config["lr_critic"]
        self.swap_rate = config["swap_rate"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = OurHexGame(sparse_flag=self.config["sparse_flag"], render_mode=None)
        self.optimize_policy_epochs = config["optimize_policy_epochs"]
        self.target_agent = TrainablePPOAgent(
            env,
            ActorCriticNN(),
            device=self.device,
            lr_actor=self.lr_actor,
            lr_critic=self.lr_critic
        )
        self.oponent_agent = TrainablePPOAgent(
            env,
            ActorCriticNN(),
            device=self.device,
            lr_actor=self.lr_actor,
            lr_critic=self.lr_critic
        )
        self.target_agent_score = 1400
        self.oponent_agent_score = 1400

    def step(self):
        # Every trainable step is a full episode
        running_reward = 0
        for _ in range(self.config["games_per_step"]):
            # The swap rate is the probability of swapping the agents
            if random.random() < self.swap_rate:
                player_1_agent = self.target_agent
                player_2_agent = self.oponent_agent
                player_name_to_rating = {
                    "player_1": self.target_agent_score,
                    "player_2": self.oponent_agent_score
                }
                player_name_to_agent_name = {
                    "player_1": "target",
                    "player_2": "oponent"
                }
            else:
                player_1_agent = self.oponent_agent
                player_2_agent = self.target_agent
                player_name_to_rating = {
                    "player_1": self.oponent_agent_score,
                    "player_2": self.target_agent_score
                }
                player_name_to_agent_name = {
                    "player_1": "oponent",
                    "player_2": "target"
                }
            player_1_cumulative_reward, player_2_cumulative_reward, winner = self.play_episode(
                player_1_agent, player_2_agent)
            player_1_agent.optimize_policy(self.optimize_policy_epochs)
            player_2_agent.optimize_policy(self.optimize_policy_epochs)
            # get only the reward from the perspective of the target agent
            if player_name_to_agent_name["player_1"] == "target":
                running_reward += player_1_cumulative_reward
            else:
                running_reward += player_2_cumulative_reward
            # Calculate the new ELO rating for the agents
            if winner == "player_1":
                self.target_agent_score = self.calculate_elo_rating(
                    self.target_agent_score,
                    player_name_to_rating["player_2"],
                    1
                )
                self.oponent_agent_score = self.calculate_elo_rating(
                    self.oponent_agent_score,
                    player_name_to_rating["player_1"],
                    0
                )
            elif winner == "player_2":
                self.target_agent_score = self.calculate_elo_rating(
                    self.target_agent_score,
                    player_name_to_rating["player_2"],
                    0
                )
                self.oponent_agent_score = self.calculate_elo_rating(
                    self.oponent_agent_score,
                    player_name_to_rating["player_1"],
                    1
                )
        return {"average_reward": running_reward/self.config["games_per_step"], 
                "target_agent_score": self.target_agent_score,
                }
    
    def calculate_elo_rating(self, current_rating: float, opponent_rating: float, score: float, k: int = 32) -> float:
        expected_score = 1 / (1 + 10 ** ((opponent_rating - current_rating) / 400))
        return current_rating + k * (score - expected_score)
            

    def play_episode(
            self,
            player_1_agent: TrainablePPOAgent,
            player_2_agent: TrainablePPOAgent
        ) -> Tuple[int, int, str]:
        """Play an episode of the game.

        Args:
            player_1_agent (Agent): player 1
            player_2_agent (Agent): player 2

        Returns:
            Tuple[int, int, str]: (player1_cumulative_reward, player2_cumulative_reward, winner)
        """
        env = self.target_agent.env
        env.reset()
        episode_cumulative_reward_per_agent = {
            "player_1": 0,
            "player_2": 0
        }
        agent_names_to_agents = {
            "player_1": player_1_agent,
            "player_2": player_2_agent
        }
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            episode_cumulative_reward_per_agent[agent_name] += reward
            agent = agent_names_to_agents[agent_name]
            if termination or truncation:
                action = None
            else:
                action = agent.select_action(
                    observation,
                    reward,
                    termination,
                    truncation,
                    info
                )
            env.step(action)

        if episode_cumulative_reward_per_agent["player_1"] > episode_cumulative_reward_per_agent["player_2"]:
            winner = "player_1"
        else:
            winner = "player_2"
        return (episode_cumulative_reward_per_agent["player_1"],
                episode_cumulative_reward_per_agent["player_2"],
                winner)
    
    def save_checkpoint(self, checkpoint_dir: str):
        checkpoint_data = {
            "nn_state_dict": self.target_agent.nn.state_dict(),
            "old_nn_state_dict": self.target_agent.old_nn.state_dict(),
            "optimizer_state_dict": self.target_agent.optimizer.state_dict(),
            "oponent_nn_state_dict": self.oponent_agent.nn.state_dict(),
            "oponent_old_nn_state_dict": self.oponent_agent.old_nn.state_dict(),
            "oponent_optimizer_state_dict": self.oponent_agent.optimizer.state_dict(),
            "target_agent_score": self.target_agent_score,
            "oponent_agent_score": self.oponent_agent_score
        }
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "wb") as fp:
            pickle.dump(checkpoint_data, fp)

        return Checkpoint.from_directory(checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir: str):
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            checkpoint_state = pickle.load(fp)
        self.target_agent.nn.load_state_dict(checkpoint_state["nn_state_dict"])
        self.target_agent.old_nn.load_state_dict(
            checkpoint_state["old_nn_state_dict"])
        self.target_agent.optimizer.load_state_dict(
            checkpoint_state["optimizer_state_dict"])
        self.oponent_agent.nn.load_state_dict(checkpoint_state["oponent_nn_state_dict"])
        self.oponent_agent.old_nn.load_state_dict(
            checkpoint_state["oponent_old_nn_state_dict"])
        self.oponent_agent.optimizer.load_state_dict(
            checkpoint_state["oponent_optimizer_state_dict"])
        self.target_agent_score = checkpoint_state["target_agent_score"]
        self.oponent_agent_score = checkpoint_state["oponent_agent_score"]


@cli.command()
@click.option("--sparse/--no-sparse", is_flag=True, default=False)
@click.option("--gpu/--no-gpu", is_flag=True, default=False)
@click.option(
    "--lr-actor",
    type=float,
    default=None
)
@click.option(
    "--lr-critic",
    type=float,
    default=None
)
@click.option(
    "--swap-rate",
    type=float,
    default=None
)
@click.option(
    "--optimize-policy-epochs",
    type=int,
    default=None
)
@click.option(
    "--num-samples",
    type=int,
    default=10
)
def train_agent(
    sparse: bool,
    gpu: bool,
    lr_actor: float,
    lr_critic: float,
    swap_rate: float,
    optimize_policy_epochs: int,
    num_samples: int):
    if not gpu:
        # Currently there is a bug in WSL2 that prevents Ray tune from auto-detecting
        # whether the current device is a GPU or not.
        # A quick and dirty fix is to directly inform RAY that the current device is a CPU.
        ray.init(num_gpus=0)
        print("--no-gpu flag detected. Forcing Ray to run on CPU.")
    config = {
        "lr_critic": lr_critic or tune.loguniform(1e-3, 1e-1),
        "lr_actor": lr_actor or tune.loguniform(1e-4, 1e-2),
        "swap_rate": swap_rate or tune.uniform(0.1, 0.5),
        "games_per_step": 10,
        "optimize_policy_epochs": optimize_policy_epochs or tune.choice([1, 3, 5, 10]),
        "sparse_flag": sparse,
    }
    scheduler = ASHAScheduler(
        max_t=10000,
        metric="average_reward",
        mode="max",
        grace_period=2,
        reduction_factor=2
    )
    # TODO: Figure out a way to better schedule the cpus
    # to avoid running out of memory.
    result = tune.run(
        SelfPlayTrainable,
        config=config,
        num_samples=num_samples,
        resources_per_trial={"cpu": 2},
        scheduler=scheduler,
        checkpoint_config=train.CheckpointConfig(checkpoint_frequency=1)
    )

    best_trial = result.get_best_trial("average_reward", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(
        f"Best trial average reward: {best_trial.last_result['average_reward']}")

@cli.command()
def re_train():
    pass


@cli.command()
@click.option(
    "--checkpoint-file",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True
)
@click.option("--sparse/--no-sparse", is_flag=True, default=False)
def test_agent(
    checkpoint_file: str,
    sparse: bool,
):
    env = OurHexGame(sparse_flag=sparse, render_mode="human")
    agent: PPOAgent = PPOAgent.from_file(checkpoint_file, env=env)
    env.reset()
    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        elif agent_name == "player_1":
            action_mask = info["action_mask"]
            action = env.action_space(agent_name).sample(action_mask)
        else:
            action = agent.select_action(
                observation,
                reward,
                termination,
                truncation,
                info
            )
        env.step(action)
        time.sleep(0.5)
    print("Game over. Freezing window for visual check.")
    time.sleep(10)


def main():
    cli()

if __name__ == "__main__":
    main()