import click
from hexgame_agents.models import TrainablePPOAgent, ActorCriticNN

from ourhexgame.ourhexenv import OurHexGame
import torch


@click.group()
def cli():
    pass


@cli.command()
def dense_train():
    env = OurHexGame(sparse_flag=False, render_mode=None)
    nn = ActorCriticNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TrainablePPOAgent(env, nn, device=device)
    total_games = 10_000
    every_how_many = 100
    current_iter = 0
    for game in range(total_games):
        env.reset()
        turns_per_agents = {
            "player_1": 0,
            "player_2": 0
        }
        cumulative_reward = 0
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            board = observation["observation"]
            mask = info["action_mask"]
            action_mask = info["action_mask"]
            if turns_per_agents[agent_name] > 0 and agent_name == "player_2":
                agent.buffer[-1].reward = reward
                agent.buffer[-1].done = termination or truncation
                cumulative_reward += reward
            
            if termination or truncation:
                action = None
            elif agent_name == "player_1":
                action = env.action_space(agent_name).sample(action_mask)
            else:
                if (current_iter + 1) % every_how_many == 0:
                    agent.optimize_policy(epochs=80)
                action = agent.select_action(board, mask)
                current_iter += 1

            env.step(action)
            turns_per_agents[agent_name] += 1
        if game % 50 == 0:
            print(f"Cumulative reward for agent in game {game} = {cumulative_reward}")
    env.close()

@cli.command()
def re_train():
    pass


def main():
    cli()

if __name__ == "__main__":
    main()