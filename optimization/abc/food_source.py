import numpy as np


class FoodSource(object):
    def __init__(self, fn):
        self.__fn = fn
        self.__position = fn.random_region_scaling()
        self.__fitness = fn(self.__position)

        self.__probability = 0.0
        self.__trials = 0

    @property
    def position(self):
        return self.__position

    @property
    def fitness(self):
        return self.__fitness

    @property
    def probability(self):
        return self.__probability

    @probability.setter
    def probability(self, p):
        self.__probability = p

    @property
    def trials(self):
        return self.__trials

    def replace(self):
        self.__init__(self.__fn)

    def greedy_search(self, reference):
        v = self.__position.copy()
        phi = np.random.uniform(-1, 1)
        i = np.random.choice(range(self.__fn.dimensions))

        v[i] = v[i] + phi * (v[i] - reference[i])
        v_fitness = self.__fn(v)

        if v_fitness < self.__fitness:
            self.__trials = 0
            self.__position = v
            self.__fitness = v_fitness
        else:
            self.__trials = self.__trials + 1
