import numpy as np

from pacman_module.game import Agent, Directions, manhattanDistance


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

    def transition_matrix(self, walls, position):
        """Builds the transition matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (k, l) | X_{t-1} = (i, j)) for
            the ghost to move from (i, j) to (k, l).
        """

        W, H = walls.width, walls.height
        T = np.zeros((W, H, W, H))

        # Set fear parameter based on ghost type
        if self.ghost == "afraid":
            fear = 1.0
        elif self.ghost == "terrified":
            fear = 3.0
        else:  # fearless
            fear = 0.0

        # For each possible previous position (i, j)
        for i in range(W):
            for j in range(H):
                # Skip if position is a wall
                if walls[i][j]:
                    continue

                # Get legal actions from position (i, j)
                legal_actions = []
                for action, (dx, dy) in [
                    (Directions.NORTH, (0, 1)),
                    (Directions.SOUTH, (0, -1)),
                    (Directions.EAST, (1, 0)),
                    (Directions.WEST, (-1, 0))
                ]:
                    next_x, next_y = i + dx, j + dy
                    # Check bounds and walls
                    if (0 <= next_x < W and 0 <= next_y < H and
                            not walls[next_x][next_y]):
                        legal_actions.append((action, next_x, next_y))

                # If no legal actions (shouldn't happen), stay in place
                if not legal_actions:
                    T[i, j, i, j] = 1.0
                    continue

                # Compute Manhattan distance from current ghost position to Pacman
                current_distance = manhattanDistance((i, j), position)

                # Compute action probabilities based on ghost policy
                action_weights = {}
                for action, next_x, next_y in legal_actions:
                    next_distance = manhattanDistance(
                        (next_x, next_y), position)
                    # Ghost favors moving away (distance increases or stays same)
                    if next_distance >= current_distance:
                        action_weights[(next_x, next_y)] = 2**fear
                    else:
                        action_weights[(next_x, next_y)] = 1.0

                # Normalize to get probabilities
                total_weight = sum(action_weights.values())
                for (next_x, next_y), weight in action_weights.items():
                    T[i, j, next_x, next_y] = weight / total_weight

        return T

    def observation_matrix(self, walls, evidence, position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """

        pass

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
