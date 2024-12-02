from collections import Counter

from ourhexgame.ourhexenv import OurHexGame
import game_utils
import math
import random

class Node:
    def __init__(self, state, local_env: OurHexGame, terminations, parent=None):
        self.state = state
        self.terminations = terminations
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
        return True in self.terminations.values()

    def get_legal_actions(self):
        # all unoccupied tiles are legal actions
        actions = [index for index, value in enumerate(self.state[:-1]) if value == 0]
        # pie rule is a legal action if only one player_1 tile is on the board
        if self.local_env.is_pie_rule_usable and self.local_env.agent_selection == "player_2":
            actions.append(len(self.state)-1)
        # print('bruh')
        return actions

    def __str__(self):
        return str(self.value) + str(self.visits)



class MCTS:

    def __init__(self, root: Node, term, env, num_iter=1000):
        self.root = root
        # we don't want to run MCTS on a terminal state
        self.termination = term
        self.env = env
        self.iterations = num_iter



    def determine_action(self):
        for _ in range(self.iterations):
            # SELECT
            selected_node = self.select_node()
            # EXPAND
            if not selected_node.is_terminal():
                self.expand_node(selected_node)
                selected_node = random.choice(selected_node.children)
            # SIMULATE
            playout_return = self.simulation(selected_node)
            # BACKPROPAGATE
            self.backpropagate(selected_node, playout_return)

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
        '''
        Expand the selected node by creating leaf nodes for all legal_actions available from that state.
        '''
        if not node.is_terminal():
            legal_actions = node.get_legal_actions()
            for action in legal_actions:
                # Make a copy env of the current state and take the action for the step

                env_copy = node.local_env.__copy__()
                env_copy.step(action)
                child_state = game_utils.flatten_observation(env_copy.observe(env_copy.agent_selection))

                child_node = node.add_child(child_state, env_copy, env.terminations)



    def simulation(self, node):
        env_copy = node.local_env.__copy__()

        while True not in env.terminations.values():
            board_state = game_utils.flatten_observation(env.observe(env.agent_selection))
            legal_actions = [index for index, value in enumerate(board_state[:-1]) if value == 0]
            random_action = random.choice(legal_actions)

            env.step(random_action)

        player = self.root.local_env.agent_selection
        return 1 if env_copy.terminations[player] else 0


    def backpropagate(self, node, ret_val):
        while node:
            node.visits += 1
            node.value += ret_val
            reward = -ret_val  # Switch perspective for opponent
            node = node.parent





env = OurHexGame()
env.reset()
observation, reward, termination, truncation, info = env.last()



#initialize tree
flattened_initial_state = game_utils.flatten_observation(observation)
root = Node(flattened_initial_state, env, env.terminations)

mcts = MCTS(root, termination, env)

print("Best action's board:")
print(mcts.determine_action())
