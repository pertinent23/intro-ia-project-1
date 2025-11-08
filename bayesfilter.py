from math import comb

import numpy as np

from pacman_module.game import Agent, Directions, manhattanDistance, Configuration, Actions


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        if ghost == 'fearless':
            self.fear = 0.0
        elif ghost == 'terrified':
            self.fear = 3.0
        else:
            self.fear = 1.0

    def transition_matrix(self, walls, pacman_position):
        """Builds the transition matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            pacman_position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (k, l) | X_{t-1} = (i, j)) for
            the ghost to move from (i, j) to (k, l).
        """
        grid_width, grid_height = walls.width, walls.height
        T = np.zeros_like(grid_width, grid_height, grid_width, grid_height)

        for i in range(grid_width):
            for j in range(grid_height):
                if walls[i, j]:
                    continue

                ghost_configuration = Configuration((i, j), Directions.STOP)
                ghost_possible_actions = Actions.getPossibleActions(ghost_configuration, walls)
                ghost_actual_distance = manhattanDistance((i, j), pacman_position)
                actions_probability = {}

                for action in ghost_possible_actions:
                    ghost_next_position = Actions.getSuccessor((i,j), action)
                    ghost_next_distance = manhattanDistance(pacman_position, ghost_next_position)
                    if ghost_next_distance > ghost_actual_distance:
                        actions_probability[action] = 2**self.fear
                    else:
                        actions_probability[action] = 1.0

                sum_all_probability = sum(actions_probability.values())
                for action, probability in actions_probability.items():
                    k, l = Actions.getSuccessor((i,j), action)
                    T[i,j,k,l] = probability/sum_all_probability
        return T

    def observation_matrix(self, walls, evidence, pacman_position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            pacman_position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """

        n = 4
        p = 0.5
        binomial_probabilities = {}
        for k in range(n + 1):
            binomial_probabilities[k] = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

        grid_width, grid_height = walls.width, walls.height
        O = np.zeros((grid_width, grid_height))

        for i in range(grid_width):
            for j in range(grid_height):
                if walls[i][j]:
                    continue

                ghost_actual_distance = manhattanDistance((i, j), pacman_position)
                # En me basant sur la formule dans le README
                # evidence = ManhattanDistance ( Pacman , Ghost ) + z − n*p
                z = evidence - ghost_actual_distance + n*p

                if 0 <= z <= n:
                    O[i, j] = binomial_probabilities[int(z)]
                else:
                    O[i, j] = 0.0

        return O

    def update(self, walls, belief, evidence, position):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """

        T = self.transition_matrix(walls, position)
        O = self.observation_matrix(walls, evidence, position)

        pass

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(
                    walls,
                    beliefs[i],
                    evidences[i],
                    position,
                )

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()

    def _get_action(self, walls, beliefs, eaten, position):
        """
        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return Directions.STOP

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )
