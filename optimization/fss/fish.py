import numpy as np
from optimization import Agent


class Fish(Agent):
    def __init__(self, fn, min_weight=1):
        super().__init__(fn)

        self.__improvement = 0
        self.__min_weight = min_weight
        self.__weight_scale = fn.dimensions
        self.__weight = self.__weight_scale / 2.0

        self.__displacement = np.zeros_like(self.position)

    @property
    def improvement(self):
        return self.__improvement

    @property
    def weight(self):
        return self.__weight

    @property
    def displacement(self):
        return self.__displacement

    def individual_step(self, step):
        random_movement = np.random.uniform(-1, 1, self.position.shape)
        n = self.position + random_movement * step('individual')
        n[n > self.fn.max] = self.fn.max
        n[n < self.fn.min] = self.fn.min

        fx = self.fn(self.position)
        fn = self.fn(n)

        self.__displacement = np.zeros_like(self.position)
        self.fitness = fx
        self.__improvement = fx - fn

        if self.__improvement > 0:
            self.__displacement = n - self.position
            self.position = n
            self.fitness = fn

    def feed(self, max_improvement):
        if max_improvement != 0:
            v = self.__improvement / max_improvement
            self.__weight = self.__weight + v
            self.__weight = max(self.__weight, self.__min_weight)

    def instinctive_step(self, drift):
        self.position = self.position + drift

    def volitive_step(self, step, barycenter, success):
        a = self.position - barycenter
        random_step = np.random.uniform(0, 1, self.position.shape)
        v = step('volitive') * (a / np.linalg.norm(a)) * random_step

        if success:
            self.position = self.position - v
        else:
            self.position = self.position + v
