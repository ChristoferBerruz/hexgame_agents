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

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            time_step = checkpoint_state["time_step"]
            agent.nn.load_state_dict(checkpoint_state["nn_state_dict"])
            agent.old_nn.load_state_dict(checkpoint_state["old_nn_state_dict"])
            agent.optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        time_step = 0
    max_time_steps = 100_000_000
    running_reward = 0
    optimize_every_steps = 64
    checkpoint_every_steps = 500
    episodes_played = 0
    while time_step < max_time_steps:
        time_step += 1
        env.reset()
        episode_cumulative_reward = 0
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if agent_name == "player_2" and agent.buffer:
                agent.buffer[-1].reward = reward
                agent.buffer[-1].done = termination or truncation
                episode_cumulative_reward += reward
            
            if termination or truncation:
                action = None
            elif agent_name == "player_1":
                action_mask = info["action_mask"]
                action = env.action_space(agent_name).sample(action_mask)
            else:
                if time_step % optimize_every_steps == 0:
                    agent.optimize_policy(epochs=80)
                action = agent.select_action(
                    observation,
                    reward,
                    termination,
                    truncation,
                    info
                )

            env.step(action)
        episodes_played += 1
        running_reward += episode_cumulative_reward
        average_reward = running_reward / episodes_played
        if time_step % checkpoint_every_steps == 0:
            checkpoint_data = {
                "time_step": time_step,
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
                    {"average_reward": average_reward},
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
    num_samples = 10
    config = {
        "lr_critic": tune.loguniform(1e-4, 1e-1),
        "lr_actor": tune.loguniform(1e-4, 1e-1),
    }
    scheduler = ASHAScheduler(
        metric="average_reward",
        mode="max",
        grace_period=1,
        reduction_factor=2
    )
    if sparse:
        train_func = train_sparse
    else:
        train_func = train_dense
    # TODO: Figure out a way to better schedule the cpus
    # to avoid running out of memory.
    result = tune.run(
        train_func,
        config=config,
        num_samples=num_samples,
        resources_per_trial={"cpu": 8},
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

def main():
    cli()

if __name__ == "__main__":
    main()