import numpy as np
from optimization import Agent


class FoodSource(Agent):
    def __init__(self, fn):
        super().__init__(fn)

        self.__probability = 0.0
        self.__trials = 0

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
        self.__init__(self.fn)

    def greedy_search(self, reference):
        v = self.position.copy()
        phi = np.random.uniform(-1, 1)
        i = np.random.choice(range(self.fn.dimensions))

        v[i] = v[i] + phi * (v[i] - reference[i])
        v[v > self.fn.max] = self.fn.max
        v[v < self.fn.min] = self.fn.min
        v_fitness = self.fn(v)

        if v_fitness < self.fitness:
            self.__trials = 0
            self.position = v
            self.fitness = v_fitness
        else:
            self.__trials = self.__trials + 1
