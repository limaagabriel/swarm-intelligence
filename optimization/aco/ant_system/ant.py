import numpy as np
from optimization.aco.ant import BaseAnt


class Ant(BaseAnt):
    def __init__(self, fn, initializer):
        super().__init__(fn, initializer)

    def state_transition_rule(self, neighborhood, alpha, beta, trails):
        possible_trails = trails[self.current_state, neighborhood]
        attractiveness = self.fn.graph[self.current_state, neighborhood] ** -1

        targets = (possible_trails ** alpha) * (attractiveness ** beta)
        probabilities = targets / targets.sum()
        return np.random.choice(neighborhood, p=probabilities)

    def local_update_rule(self, trails):
        return np.zeros_like(trails)

    def global_update_rule(self, q):
        n = self.fn.num_states
        pheromone = np.zeros((n, n))

        if not self.failed:
            step = q / self.fitness
            src = self.position[:-1]
            dest = self.position[1:]

            pheromone[src, dest] = step
            pheromone[dest, src] = step

        return pheromone
