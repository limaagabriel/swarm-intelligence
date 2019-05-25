import numpy as np
from optimization.aco.ant import Ant
from optimization import SwarmOptimizationMethod


class AntSystem(SwarmOptimizationMethod):
    def __init__(self, num_ants, alpha, beta, rho, q):
        self.__alpha = alpha
        self.__beta = beta
        self.__rho = rho
        self.__q = q

        self.__trails = None
        super().__init__(Ant, num_ants)

    def __call__(self, fn, stop_criterion, tracker):
        it = 0
        self.__trails = np.ones_like(fn.graph)

        while not stop_criterion(iterations=it, evaluations=fn.evaluations):
            it = it + 1
            self.__random_travels()
            self.__update_trails(fn)
            tracker.track_by_iterations(self.best_fitness)

        return self.best_position, self.best_fitness

    def __random_travels(self):
        for ant in self.swarm:
            ant.make_travel(self.__alpha, self.__beta, self.__trails)
            self.update_best_solution()

    def __update_trails(self, fn):
        mp = 1.0e-6
        n = fn.num_states
        portions = np.zeros((len(self.swarm), n, n))
        for i, ant in enumerate(self.swarm):
            portions[i] = ant.get_pheromone(self.__q)

        a = (1.0 - self.__rho) * self.__trails
        b = portions.sum(axis=0)

        self.__trails = a + b
        self.__trails[self.__trails <= mp] = mp
