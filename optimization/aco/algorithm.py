import numpy as np
from util import foreach
from abc import abstractmethod
from optimization.aco.ant import BaseAnt
from optimization import SwarmOptimizationMethod


class ACO(SwarmOptimizationMethod):
    agent = BaseAnt

    def __init__(self, swarm_size, **kwargs):
        self.trails = None
        super().__init__(self.agent, swarm_size, **kwargs)

    def __call__(self, fn, stop_criterion, tracker):
        it = 0
        self.trails = np.ones_like(fn.graph)

        while not stop_criterion(iterations=it, evaluations=fn.evaluations):
            it = it + 1
            self.make_tours()
            self.local_pheromone_update(fn)
            self.global_pheromone_update(fn)
            foreach(lambda x: x.replace(), self.swarm)
            tracker.track_by_iterations(self.best_fitness)

        return self.best_position, self.best_fitness

    @abstractmethod
    def make_tours(self):
        pass

    @abstractmethod
    def local_pheromone_update(self, fn):
        pass

    @abstractmethod
    def global_pheromone_update(self, fn):
        pass
