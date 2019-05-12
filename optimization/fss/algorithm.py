import sys
import numpy as np

from util import foreach
from optimization.fss.fish import Fish
from optimization import SwarmOptimizationMethod


class FSS(SwarmOptimizationMethod):
    def __init__(self, n_fishes, step):
        super().__init__(Fish, n_fishes)
        self.__step = step

    def __call__(self, fn, stop_criterion, tracker):
        it = 0
        best_position = None
        best_fitness = sys.maxsize

        while not stop_criterion(iterations=it, evaluations=fn.evaluations):
            it = it + 1

            school_weight_1 = sum(map(lambda x: x.weight, self.swarm))
            foreach(lambda f: f.individual_step(self.__step), self.swarm)

            max_improvement = self.__find_max_improvement(self.swarm)
            foreach(lambda f: f.feed(max_improvement), self.swarm)
            school_weight_2 = sum(map(lambda x: x.weight, self.swarm))

            drift = self.__find_drift(self.swarm)
            foreach(lambda f: f.instinctive_step(drift), self.swarm)

            barycenter = self.__find_barycenter(self.swarm)
            success = school_weight_2 > school_weight_1
            foreach(lambda f: f.volitive_step(self.__step, barycenter, success), self.swarm)

            self.__step.update(stop_criterion, iterations=it, evaluations=fn.evaluations)
            best_fish = self.get_best_agent()

            if best_fish.fitness < best_fitness:
                best_fitness = best_fish.fitness
                best_position = best_fish.position

            tracker.track_by_iterations(best_fitness)
        return best_position, best_fitness

    @staticmethod
    def __find_max_improvement(school):
        best_fish = max(school, key=lambda f: np.abs(f.improvement))
        return np.abs(best_fish.improvement)

    @staticmethod
    def __find_drift(school):
        a = sum(map(lambda x: x.displacement * x.improvement, school))
        b = sum(map(lambda x: x.improvement, school))

        if b > 0:
            return a / b
        return np.zeros(school[0].position.shape)

    @staticmethod
    def __find_barycenter(school):
        a = sum(map(lambda x: x.position * x.weight, school))
        b = sum(map(lambda x: x.weight, school))

        return a / b
