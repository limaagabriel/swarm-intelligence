import uuid
import numpy as np


class Particle(object):
    def __init__(self, fn):
        self.__fn = fn
        self.__id = str(uuid.uuid4())

        mean = (fn.minf + fn.maxf) / 2.0
        std_dev = min(np.abs(fn.minf), np.abs(fn.maxf)) / 2.0

        self.__speed = np.random.normal(0.0, 1.0, fn.dim)
        self.__position = np.random.normal(mean, std_dev, fn.dim)
        self.__restrict_to_fn_boundaries()

        self.__fitness = fn(self.__position)

        self.social_reference = None
        self.cognitive_reference = self.__position.copy()

    @property
    def id(self):
        return self.__id

    @property
    def fitness(self):
        return self.__fitness

    @property
    def current_fitness(self):
        return self.__fn(self.__position)

    @property
    def position(self):
        return self.__position

    @property
    def speed(self):
        return self.__speed

    def update(self, a, b, c1, c2):
        r1 = np.random.random(self.__position.shape)
        r2 = np.random.random(self.__position.shape)

        inertia = a * self.__speed
        cognitive_component = b * c1 * r1 * (self.cognitive_reference - self.__position)
        social_component = b * c2 * r2 * (self.social_reference - self.__position)

        self.__speed = inertia + cognitive_component + social_component
        self.__position = self.__position + self.__speed

    def evaluate(self):
        if not self.__is_out_of_bounds():
            fitness = self.__fn(self.__position)

            if fitness <= self.__fitness:
                self.__fitness = fitness
                self.cognitive_reference = self.__position

    def distance_to(self, other):
        def squared_euclidean(a, b):
            return np.sum(np.square(a - b))

        return squared_euclidean(self.position, other.position)

    def __is_out_of_bounds(self):
        lower_bounds = self.__position < self.__fn.minf
        upper_bounds = self.__position > self.__fn.maxf

        return lower_bounds.any() or upper_bounds.any()

    def __restrict_to_fn_boundaries(self):
        self.__position[self.__position < self.__fn.minf] = self.__fn.minf
        self.__position[self.__position > self.__fn.maxf] = self.__fn.maxf

    def __eq__(self, other):
        return self.id == other.id
