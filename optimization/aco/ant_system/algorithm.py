import numpy as np
from optimization.aco.algorithm import ACO
from optimization.aco.ant_system.ant import Ant


class AntSystem(ACO):
    def __init__(self, n_ants, alpha, beta, rho, q):
        self.__alpha = alpha
        self.__beta = beta
        self.__rho = rho
        self.__q = q

        super().__init__(Ant, n_ants)

    def make_tours(self):
        for ant in self.swarm:
            ant.make_tour(self.__alpha, self.__beta, self.trails)
            self.update_best_solution()

    def local_pheromone_update(self, fn):
        pass

    def global_pheromone_update(self, fn):
        mp = 1.0e-6
        n = fn.num_states
        portions = np.zeros((len(self.swarm), n, n))
        for i, ant in enumerate(self.swarm):
            portions[i] = ant.global_update_rule(self.__q)

        a = (1.0 - self.__rho) * self.trails
        b = portions.sum(axis=0)

        self.trails = a + b
        self.trails[self.trails <= mp] = mp
