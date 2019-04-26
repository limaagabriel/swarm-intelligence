import math
import numpy as np


class Fish(object):
    def __init__(self, fn, min_weight):
        self.__fn = fn

        self.__improvement = 0
        self.__min_weight = min_weight
        self.__weight_scale = fn.dimensions
        self.__weight = self.__weight_scale / 2.0

        self.__current_fitness = 0
        self.__position = fn.random_region_scaling()
        self.__displacement = np.zeros_like(self.position)

    @property
    def improvement(self):
        return self.__improvement

    @property
    def position(self):
        return self.__position

    @property
    def weight(self):
        return self.__weight

    @property
    def fitness(self):
        return self.__current_fitness

    @property
    def displacement(self):
        return self.__displacement

    def individual_step(self, step):
        random_movement = np.random.uniform(-1, 1, self.position.shape)
        n = self.__position + random_movement * step('individual')
        n[n > self.__fn.max] = self.__fn.max
        n[n < self.__fn.min] = self.__fn.min

        fx = self.__fn(self.position)
        fn = self.__fn(n)

        self.__displacement = np.zeros_like(self.position)
        self.__current_fitness = fx
        self.__improvement = fx - fn

        if self.__improvement > 0:
            self.__position = n
            self.__current_fitness = fn
            self.__displacement = n - self.position

    def feed(self, max_improvement):
        if max_improvement != 0:
            v = self.__improvement / max_improvement
            self.__weight = self.__weight + v
            self.__weight = max(self.__weight, self.__min_weight)

    def instinctive_step(self, drift):
        self.__position = self.__position + drift

    def volitive_step(self, step, barycenter, success):
        a = self.position - barycenter
        random_step = np.random.uniform(0, 1, self.position.shape)
        v = step('volitive') * (a / np.linalg.norm(a)) * random_step

        if success:
            self.__position = self.position - v
        else:
            self.__position = self.position + v
