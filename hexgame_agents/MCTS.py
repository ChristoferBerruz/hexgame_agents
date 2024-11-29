from collections import Counter

from ourhexgame.ourhexenv import OurHexGame
import game_utils
import math
import random

class Node:
    def __init__(self, state, local_env, termination, parent=None):
        self.state = state
        self.is_terminal_state = termination
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.local_env = local_env

    def is_fully_expanded(self):
        return len(self.children) == len(self.get_legal_actions())

    def best_child(self, exploration_weight=1.0):
        #todo, compare this equation against UCB equation in slides
        return max(
            self.children,
            key=lambda child: child.value / (child.visits + 1e-6) +
                              exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
        )

    def add_child(self, child_state, child_env, child_termination):
        child_node = Node(child_state, child_env, child_termination, self)
        self.children.append(child_node)
        return child_node

    def is_terminal(self):
        return self.is_terminal_state

    def get_legal_actions(self):
        # all unoccupied tiles are legal actions
        actions = [index for index, value in enumerate(self.state[:-1]) if value == 0]
        # pie rule is a legal action if only one player_1 tile is on the board
        tile_freq_dict = Counter([tile for tile in self.state[:-1]])
        if tile_freq_dict[1] == 1:
            actions.append(len(self.state)-1)
        print('bruh')
        return actions


class MCTS:

    def __init__(self, root, term, env, num_iter=1000):
        self.root = root
        # we don't want to run MCTS on a terminal state
        self.termination = term
        self.env = env
        self.iterations = num_iter


    def determine_action(self):
        for _ in range(self.iterations):

            selected_node = self.select_node()

            self.expand_node(selected_node)

            print('yay')

            for child_node in selected_node.children:
                self.simulate_and_backpropagate_playout(child_node)

        return self.root.best_child(0)  # Best child without exploration


    def select_node(self) -> Node:
        """
        Traverse the tree downwards from the root to find the best node using UCB.
        """
        node = self.root
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()
        return node

    def expand_node(self, node):
        if not node.is_terminal():
            legal_actions = node.get_legal_actions()
            for action in legal_actions:
                # Make a copy env of the current state and take the action for the step

                env_copy = node.local_env.__copy__()
                env_copy.step(action)
                child_state = game_utils.flatten_observation(env_copy.observe(env_copy.agent_selection))

                child_node = node.add_child(child_state, env_copy, env.terminations)


    # def simulate_and_backpropagate_playout(self, child_node):
    #     # # Simulation
    #     # # todo: save a env object in the tree
    #
    #     current_state = node.state
    #     while not current_state.is_terminal():
    #         # rollout policy
    #         action = random.choice(current_state.get_legal_actions())
    #         # todo, actually step
    #         env.step(action)
    #         current_state = flattened_state(env.observe('player_1'))
    #
    #     # todo: accumulate reward list and pop from stack
    #     # Backpropagation
    #     return = env.rewards['player_1']
    #     while node:
    #         node.visits += 1
    #         node.value += reward
    #         # reward = -reward  # Switch perspective for opponent todo: maybe get rid of this?
    #         node = node.parent



env = OurHexGame()
env.reset()
observation, reward, termination, truncation, info = env.last()



#initialize tree
flattened_initial_state = game_utils.flatten_observation(observation)
root = Node(flattened_initial_state, env, termination)

mcts = MCTS(root, termination, env)

print("Best action's board:")
print(mcts.determine_action())
