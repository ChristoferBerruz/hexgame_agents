import click
from hexgame_agents.models import TrainablePPOAgent, ActorCriticNN, PPOAgent

from ourhexgame.ourhexenv import OurHexGame
import torch

import time
import ray
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from pathlib import Path
import tempfile


@click.group()
def cli():
    pass

def train_general(sparse_flag: bool, config):
    env = OurHexGame(sparse_flag=sparse_flag, render_mode=None)
    nn = ActorCriticNN()
    lr_actor = config["lr_actor"]
    lr_critic = config["lr_critic"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TrainablePPOAgent(env, nn, device=device, lr_actor=lr_actor, lr_critic=lr_critic)
    total_games = 1_000
    every_how_many = 100
    current_iter = 0

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_game = checkpoint_state["game"]
            agent.nn.load_state_dict(checkpoint_state["nn_state_dict"])
            agent.old_nn.load_state_dict(checkpoint_state["old_nn_state_dict"])
            agent.optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_game = 0

    for game in range(start_game, total_games):
        env.reset()
        turns_per_agents = {
            "player_1": 0,
            "player_2": 0
        }
        cumulative_reward = 0
        n_steps = 0
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if agent_name == "player_2" and agent.buffer:
                agent.buffer[-1].reward = reward
                agent.buffer[-1].done = termination or truncation
                cumulative_reward += reward
            
            if termination or truncation:
                action = None
            elif agent_name == "player_1":
                action_mask = info["action_mask"]
                action = env.action_space(agent_name).sample(action_mask)
            else:
                if (current_iter + 1) % every_how_many == 0:
                    agent.optimize_policy(epochs=80)
                action = agent.select_action(
                    observation,
                    reward,
                    termination,
                    truncation,
                    info
                )
                current_iter += 1
                n_steps += 1

            env.step(action)
            turns_per_agents[agent_name] += 1
        if game % 50 == 0:
            checkpoint_data = {
                "game": game,
                "nn_state_dict": agent.nn.state_dict(),
                "old_nn_state_dict": agent.old_nn.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict()
            }
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(
                    {"average_reward": cumulative_reward / n_steps},
                    checkpoint=checkpoint,
                )
    env.close()

def train_dense(config):
    train_general(sparse_flag=False, config=config)

def train_sparse(config):
    train_general(sparse_flag=True, config=config)


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
    max_num_epochs = 10
    num_samples = 10
    config = {
        "lr_critic": tune.loguniform(1e-4, 1e-1),
        "lr_actor": tune.loguniform(1e-4, 1e-1),
    }
    scheduler = ASHAScheduler(
        metric="average_reward",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    if sparse:
        train_func = train_sparse
    else:
        train_func = train_dense
    result = tune.run(
        train_func,
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
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

def main():
    cli()

if __name__ == "__main__":
    main()