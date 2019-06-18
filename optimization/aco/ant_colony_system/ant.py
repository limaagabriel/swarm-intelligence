import sys
import numpy as np
from optimization.aco.ant import BaseAnt


class Ant(BaseAnt):
    def __init__(self, fn, initializer):
        super().__init__(fn, initializer)

    def state_transition_rule(self, neighborhood, q0, alpha, beta, trails):
        if np.random.uniform(0, 1) <= q0:
            return self.__exploit(neighborhood, alpha, beta, trails)
        return self.__explore(neighborhood, alpha, beta, trails)

    def __get_targets(self, neighborhood, alpha, beta, trails):
        possible_trails = trails[self.current_state, neighborhood]
        attractiveness = self.fn.graph[self.current_state, neighborhood] ** -1

        return (possible_trails ** alpha) * (attractiveness ** beta)

    def __exploit(self, neighborhood, alpha, beta, trails):
        targets = self.__get_targets(neighborhood, alpha, beta, trails)
        return neighborhood[targets.argmax()]

    def __explore(self, neighborhood, alpha, beta, trails):
        targets = self.__get_targets(neighborhood, alpha, beta, trails)

        probabilities = targets / targets.sum()
        return np.random.choice(neighborhood, p=probabilities)

    def local_update_rule(self, t0=1):
        n = self.fn.num_states
        matrix = np.zeros((n, n))

        if not self.failed:
            src = self.position[:-1]
            dest = self.position[1:]

            matrix[src, dest] = t0
            matrix[dest, src] = t0
        return matrix

    def global_update_rule(self, *args, **kwargs):
        pass
