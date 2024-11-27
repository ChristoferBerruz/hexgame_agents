import click
from hexgame_agents.models import TrainablePPOAgent, ActorCriticNN, PPOAgent

from ourhexgame.ourhexenv import OurHexGame
import torch

import random

import time
import ray
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from pathlib import Path
import tempfile

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

    def step(self):
        # Every trainable step is a full episode
        running_reward = 0
        for _ in range(self.config["games_per_step"]):
            # The swap rate is the probability of swapping the agents
            if random.random() < self.swap_rate:
                player_1_agent = self.target_agent
                player_2_agent = self.oponent_agent
            else:
                player_1_agent = self.oponent_agent
                player_2_agent = self.target_agent
            player_1_cumulative_reward, player_2_cumulative_reward, winner = self.play_episode(
                player_1_agent, player_2_agent)
            player_1_agent.optimize_policy(self.optimize_policy_epochs)
            player_2_agent.optimize_policy(self.optimize_policy_epochs)
            running_reward += player_1_cumulative_reward + player_2_cumulative_reward
            running_reward /= 2
        return {"average_reward": running_reward/self.config["games_per_step"]}
            

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
            if agent.buffer:
                agent.buffer[-1].reward = reward
                agent.buffer[-1].done = termination or truncation
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

        return (episode_cumulative_reward_per_agent["player_1"],
                episode_cumulative_reward_per_agent["player_2"],
                env.winner)
    
    def save_checkpoint(self):
        checkpoint_data = {
            "nn_state_dict": self.target_agent.nn.state_dict(),
            "old_nn_state_dict": self.target_agent.old_nn.state_dict(),
            "optimizer_state_dict": self.target_agent.optimizer.state_dict(),
            "oponent_nn_state_dict": self.oponent_agent.nn.state_dict(),
            "oponent_old_nn_state_dict": self.oponent_agent.old_nn.state_dict(),
            "oponent_optimizer_state_dict": self.oponent_agent.optimizer.state_dict()
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            return Checkpoint.from_directory(checkpoint_dir)

    def load_checkpoint(self):
        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
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


@cli.command()
@click.option("--sparse/--no-sparse", is_flag=True, default=False)
@click.option("--gpu/--no-gpu", is_flag=True, default=False)
def train_agent(sparse: bool, gpu: bool):
    if not gpu:
        # Currently there is a bug in WSL2 that prevents Ray tune from auto-detecting
        # whether the current device is a GPU or not.
        # A quick and dirty fix is to directly inform RAY that the current device is a CPU.
        ray.init(num_gpus=0)
        print("--no-gpu flag detected. Forcing Ray to run on CPU.")
    num_samples = 10
    config = {
        "lr_critic": tune.loguniform(1e-4, 1e-1),
        "lr_actor": tune.loguniform(1e-4, 1e-1),
        "swap_rate": tune.uniform(0.1, 0.9),
        "games_per_step": 10,
        "optimize_policy_epochs": 80,
        "sparse_flag": sparse,
    }
    scheduler = ASHAScheduler(
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
    )

    best_trial = result.get_best_trial("average_reward", "max", "avg")
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
    print("Winner: ", env.winner)

def main():
    cli()

if __name__ == "__main__":
    main()