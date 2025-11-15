import numpy as np
from math import comb
from collections import deque

from pacman_module.game import Agent, Directions, manhattanDistance, Actions, Configuration
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
                # Dans ce mode (avec beliefStates), la règle "pas de marche arrière"
                # est désactivée (vu dans pacman.py, GhostRules)
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

                    # Assigner un poids plus élevé aux actions qui s'éloignent
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
                    k, l = Actions.getSuccessor(ghost_pos_prev, a)

                    # S'assurer que le successeur est dans les limites
                    if 0 <= k < W and 0 <= l < H:
                        T[i, j, k, l] = prob

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

                # z doit être un entier entre 0 et 4 pour avoir une probabilité
                # non nulle (car z ~ Binom(4, 0.5))
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

        # Si la probabilité totale est nulle, retourner une distribution uniforme
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

    def _get_action(self, walls, beliefs, eaten, position):
        """Choisit une action pour Pacman en fonction des belief states.

        Arguments:
            walls: Grille W x H des murs.
            beliefs: Liste des belief states des fantômes (matrices W x H).
            eaten: Liste de booléens indiquant quels fantômes sont mangés.
            position: Position actuelle de Pacman (x, y).

        Returns:
            Une action légale (Directions.*).
        """
        # Si aucune croyance ou si tous les fantômes sont mangés, on ne bouge pas
        if not beliefs or all(eaten):
            return Directions.STOP

        merged = None
        for i, b in enumerate(beliefs):
            if eaten[i] or b is None:
                continue
            arr = np.array(b, dtype=float)
            if merged is None:
                merged = arr
            else:
                merged = np.maximum(merged, arr)

        # Si on n'a rien obtenu ou que tout est à zéro, on ne sait pas où aller
        if merged is None or np.sum(merged) == 0.0:
            return Directions.STOP

        target = tuple(np.unravel_index(np.argmax(merged), merged.shape))

        # Si Pacman est déjà sur la cible, on s'arrête
        if position == target:
            return Directions.STOP

        start = position
        visited = set([start])
        queue = util.Queue()
        # Dans la file: (position_courante, première_action_depuis_start)
        queue.push((start, None))

        while not queue.isEmpty():
            curr_pos, first_move = queue.pop()

            if curr_pos == target:
                return first_move or Directions.STOP

            conf = Configuration(curr_pos, Directions.STOP)
            legal_actions = Actions.getPossibleActions(conf, walls)

            for action in legal_actions:
                if action == Directions.STOP:
                    continue

                succ = Actions.getSuccessor(curr_pos, action)
                x, y = succ
                # On ignore les murs et les cases déjà visitées
                if walls[x][y] or succ in visited:
                    continue

                visited.add(succ)
                # Si on n'a pas encore choisi la première action,
                # celle que l'on joue maintenant devient la première du chemin.
                next_first = first_move if first_move is not None else action
                queue.push((succ, next_first))

        # Récupérer les actions légales depuis la position actuelle
        conf = Configuration(position, Directions.STOP)
        legal = Actions.getPossibleActions(conf, walls)

        # Retirer STOP s'il y a d'autres actions possibles
        if Directions.STOP in legal and len(legal) > 1:
            legal = [a for a in legal if a != Directions.STOP]

        best_action = Directions.STOP
        best_dist = manhattanDistance(position, target)

        for action in legal:
            succ = Actions.getSuccessor(position, action)
            dist = manhattanDistance(succ, target)
            if dist < best_dist:
                best_dist = dist
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
