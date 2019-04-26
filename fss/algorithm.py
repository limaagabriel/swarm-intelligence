import sys
import numpy as np

from util import foreach
from fss.fish import Fish


class FSS(object):
    def __init__(self, n_fishes, step):
        self.__n_fishes = n_fishes
        self.__step = step

    def optimize(self, fn, stop_criterion):
        it = 0
        improvements = 0
        best_position = None
        best_fitness = sys.maxsize
        school = self.__create_school(fn)

        while not stop_criterion(iterations=it, evaluations=fn.evaluations):
            it = it + 1

            school_weight_1 = sum(map(lambda x: x.weight, school))
            foreach(lambda f: f.individual_step(self.__step), school)

            max_improvement = self.__find_max_improvement(school)
            foreach(lambda f: f.feed(max_improvement), school)
            school_weight_2 = sum(map(lambda x: x.weight, school))

            drift = self.__find_drift(school)
            foreach(lambda f: f.instinctive_step(drift), school)

            barycenter = self.__find_barycenter(school)
            success = school_weight_2 > school_weight_1
            foreach(lambda f: f.volitive_step(self.__step, barycenter, success), school)

            self.__step.update(stop_criterion, iterations=it, evaluations=fn.evaluations)
            best_fish = min(school, key=lambda x: x.fitness)

            if best_fish.fitness < best_fitness:
                best_fitness = best_fish.fitness
                best_position = best_fish.position
                improvements = improvements + 1
                print('min updated {}'.format(best_fitness))

        print('fitness improved {} times'.format(improvements))
        return best_position, best_fitness, []

    def __create_school(self, fn):
        return [Fish(fn, 1) for _ in range(self.__n_fishes)]

    @staticmethod
    def __find_max_improvement(school):
        best_fish = max(school, key=lambda f: f.improvement)
        return best_fish.improvement

    @staticmethod
    def __find_drift(school):
        a = sum(map(lambda x: x.displacement * max(0, x.improvement), school))
        b = sum(map(lambda x: max(0, x.improvement), school))

        if b > 0:
            return a / b

        return np.zeros(school[0].position.shape)

    @staticmethod
    def __find_barycenter(school):
        a = sum(map(lambda x: x.position * x.weight, school))
        b = sum(map(lambda x: x.weight, school))

        return a / b
