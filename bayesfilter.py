"""
INFO8006 - Project 1 - Bayes Filter
Student :
    Lucas Bauduin
    Franck Duval Heuba Batomen
    Morgan Phemba
"""

import numpy as np
from math import comb
from collections import deque

from pacman_module.game import (
    Agent,
    Directions,
    manhattanDistance,
    Actions,
    Configuration,
)
from pacman_module import util


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        # Déterminer le paramètre de "peur" basé sur le type de fantôme
        # Logique déduite de ghostAgents.py
        if ghost == 'afraid':
            self.fear = 1.0
        elif ghost == 'fearless':
            self.fear = 0.0
        elif ghost == 'terrified':
            self.fear = 3.0
        else:
            self.fear = 1.0  # Valeur par défaut

        # Pré-calculer les probabilités de la loi binomiale P(z=k)
        # pour z ~ Binom(n=4, p=0.5)
        # P(z=k) = comb(4, k) * (0.5)^k * (0.5)^(4-k) = comb(4, k) * (0.5)^4
        n = 4
        p = 0.5
        self.np_mean = n * p  # C'est 2.0
        self.bin_probs = {k: comb(n, k) * (p**n) for k in range(n + 1)}

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
        pac_pos = position

        # Itérer sur chaque position de départ possible (i, j)
        for i in range(W):
            for j in range(H):
                if walls[i][j]:
                    continue

                ghost_pos_prev = (i, j)
                dist_prev = manhattanDistance(ghost_pos_prev, pac_pos)

                # Obtenir les actions légales.
                # Dans ce mode,
                # la règle "pas de marche arrière" est désactivée.
                conf = Configuration(ghost_pos_prev, Directions.STOP)
                legal_actions = Actions.getPossibleActions(conf, walls)

                # Recréer la logique de AfraidGhost.getDistribution
                if Directions.STOP in legal_actions:
                    legal_actions.remove(Directions.STOP)

                action_probs = {}
                for a in legal_actions:
                    # Obtenir la position successeur
                    succ_pos = Actions.getSuccessor(ghost_pos_prev, a)
                    dist_succ = manhattanDistance(succ_pos, pac_pos)

                    # Assigner un poids plus élevé
                    # aux actions qui s'éloignent
                    if dist_succ >= dist_prev:
                        action_probs[a] = 2**self.fear
                    else:
                        action_probs[a] = 1

                # Normaliser les probabilités des actions
                total_weight = sum(action_probs.values())
                if total_weight > 0:
                    for a in action_probs:
                        action_probs[a] /= total_weight
                else:
                    # Si aucune action (sauf STOP) n'était possible
                    action_probs[Directions.STOP] = 1.0

                # Remplir la matrice de transition
                for a, prob in action_probs.items():
                    succ = Actions.getSuccessor(ghost_pos_prev, a)
                    # Coercer en coordonnées entières de la grille
                    # (le successeur peut être en float)
                    k, l = int(succ[0]), int(succ[1])

                    # S'assurer que le successeur est dans les limites
                    if 0 <= k < W and 0 <= l < H:
                        # Utiliser += au cas où
                        # plusieurs actions mènent à la même case
                        T[i, j, k, l] += prob

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

        W, H = walls.width, walls.height
        O = np.zeros((W, H))

        # Pour chaque position possible du fantôme
        for i in range(W):
            for j in range(H):
                # Si c'est un mur, la probabilité est 0
                if walls[i][j]:
                    continue

                ghost_pos = (i, j)
                # Calculer la vraie distance de Manhattan
                true_distance = manhattanDistance(ghost_pos, position)

                # Calculer P(evidence | ghost est en (i,j))
                # evidence = true_distance + z - np
                # donc z = evidence - true_distance + np
                z = evidence - true_distance + self.np_mean

                # z doit être un entier entre 0 et 4
                # pour avoir une probabilité non nulle
                # (car z ~ Binom(4, 0.5))
                if z in self.bin_probs:
                    O[i, j] = self.bin_probs[z]
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

        # S'assurer que le belief est un tableau numpy
        b_prev = np.array(belief, dtype=float, copy=True)

        # 1. Prediction
        # b_prime(x_t) = sum_{x_{t-1}} P(x_t | x_{t-1}) * b(x_{t-1})
        # T a la forme (W, H, W, H) et b_prev la forme (W, H)
        b_prime = np.einsum('ijkl,ij->kl', T, b_prev)

        # 2. Correction
        # b(x_t) ∝ P(e_t | x_t) * b_prime(x_t)
        b_t = O * b_prime

        # 3. Normalisation
        total_prob = float(np.sum(b_t))
        if total_prob > 0.0:
            b_t /= total_prob
            return b_t

        # Si la probabilité totale est nulle,
        # retourner une distribution uniforme
        W, H = walls.width, walls.height
        uniform = np.zeros((W, H), dtype=float)
        free_count = 0
        for i in range(W):
            for j in range(H):
                if not walls[i][j]:
                    uniform[i, j] = 1.0
                    free_count += 1

        if free_count > 0:
            uniform /= float(free_count)

        return uniform

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
        # Cache des coordonnées de la grille pour éviter de refaire le meshgrid
        self._grid_shape = None
        self._xs = None
        self._ys = None

    def _ensure_coord_cache(self, walls):
        """Pré-calcul de xs, ys si la taille de la grille change."""
        W, H = walls.width, walls.height
        if self._grid_shape != (W, H):
            xs, ys = np.meshgrid(np.arange(W), np.arange(H), indexing="ij")
            self._xs = xs
            self._ys = ys
            self._grid_shape = (W, H)

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

        # Cas dégénéré : rien à faire
        if beliefs is None or len(beliefs) == 0 or all(eaten):
            return Directions.STOP

        # Actions légales depuis la position actuelle
        conf = Configuration(position, Directions.STOP)
        legal_actions = Actions.getPossibleActions(conf, walls)

        # Éviter STOP si on a d'autres choix
        if Directions.STOP in legal_actions and len(legal_actions) > 1:
            legal_actions.remove(Directions.STOP)

        if not legal_actions:
            return Directions.STOP

        # Assurer le cache des coordonnées
        self._ensure_coord_cache(walls)
        xs, ys = self._xs, self._ys
        W, H = self._grid_shape

        # 1) Pré-calculer pour chaque fantôme une position "moyenne"
        ghost_means = []
        has_active_ghost = False

        for i, belief in enumerate(beliefs):
            if eaten[i]:
                ghost_means.append(None)
                continue

            b = np.asarray(belief, dtype=float)
            total = b.sum()
            if total <= 0.0:
                ghost_means.append(None)
                continue

            # Normalisation
            b /= total

            # Espérance des coordonnées
            ex = float((xs * b).sum())
            ey = float((ys * b).sum())
            ghost_means.append((ex, ey))
            has_active_ghost = True

        if not has_active_ghost:
            return Directions.STOP

        # 2) Choisir l'action qui minimise la distance à la position moyenne
        best_action = Directions.STOP
        best_score = float("inf")

        for action in legal_actions:
            succ = Actions.getSuccessor(position, action)
            x_s, y_s = int(succ[0]), int(succ[1])

            # Sécurité : ne pas sortir de la grille ou aller dans un mur
            if not (0 <= x_s < W and 0 <= y_s < H):
                continue
            if walls[x_s][y_s]:
                continue

            # Distance au fantôme le plus "proche" (en moyenne)
            min_dist = float("inf")
            for mean in ghost_means:
                if mean is None:
                    continue
                ex, ey = mean
                d = abs(x_s - ex) + abs(y_s - ey)
                if d < min_dist:
                    min_dist = d

            # Score de l'action : plus petit = mieux
            if min_dist < best_score:
                best_score = min_dist
                best_action = action

        return best_action

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
