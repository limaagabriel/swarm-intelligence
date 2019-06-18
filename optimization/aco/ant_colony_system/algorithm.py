import numpy as np
from optimization.aco.algorithm import ACO
from optimization.aco.ant_colony_system.ant import Ant

# Implemented following the instructions provided in:
# Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem
# Authors: Marco Dorigo and Luca Maria Gambardella

# Default parameters were chosen based on:
# On Optimal Parameters for Ant Colony Optimization algorithms
# Authors: Dorian Gaertner and Keith Clark


class AntColonySystem(ACO):
    def __init__(self, n_ants, alpha=1, beta=6, rho=0.6, q0=0.8):
        self.__alpha = alpha
        self.__beta = beta
        self.__rho = rho
        self.__q0 = q0

        super().__init__(Ant, n_ants)

    def make_tours(self):
        for ant in self.swarm:
            ant.make_tour(self.__q0, self.__alpha, self.__beta, self.trails)
            self.update_best_solution()

    def local_pheromone_update(self, fn):
        mp = 1.0e-6
        n = fn.num_states
        portions = np.zeros((len(self.swarm), n, n))
        for i, ant in enumerate(self.swarm):
            portions[i] = ant.local_update_rule()

        a = (1.0 - self.__rho) * self.trails
        b = self.__rho * portions.sum(axis=0)

        self.trails = a + b
        self.trails[self.trails <= mp] = mp

    def global_pheromone_update(self, fn):
        a = (1 - self.__rho) * self.trails
        b = self.__rho * self.__make_delta_matrix()

        self.trails = a + b

    def __make_delta_matrix(self):
        src = self.best_position[:-1]
        dest = self.best_position[1:]

        matrix = np.zeros_like(self.trails)
        matrix[src, dest] = 1.0 / self.best_fitness
        matrix[dest, src] = 1.0 / self.best_fitness
        return matrix
