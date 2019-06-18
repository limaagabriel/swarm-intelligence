import numpy as np
from optimization import Agent


class Particle(Agent):
    def __init__(self, fn, initializer, max_speed=1.0):
        super().__init__(fn, initializer)
        self.__max_speed = max_speed * (fn.max - fn.min)

        self.__speed = np.zeros(fn.dimensions)
        self.__restrict_to_fn_boundaries()

        self.social_reference = None
        self.cognitive_reference = self.position.copy()

    @property
    def current_fitness(self):
        return self.fn(self.position)

    @property
    def speed(self):
        return self.__speed

    def update(self, a, b, c1, c2):
        r1 = np.random.uniform(0.0, 1.0, self.fn.dimensions)
        r2 = np.random.uniform(0.0, 1.0, self.fn.dimensions)

        inertia = a * self.__speed
        cognitive_component = b * c1 * r1 * (self.cognitive_reference - self.position)
        social_component = b * c2 * r2 * (self.social_reference - self.position)

        self.__speed = inertia + cognitive_component + social_component
        self.__restrict_speed()
        self.position = self.position + self.__speed

    def evaluate(self):
        if not self.is_out_of_bounds():
            fitness = self.fn(self.position)

            if fitness < self.fitness:
                self.fitness = fitness
                self.cognitive_reference = self.position

    def __restrict_speed(self):
        self.__speed[self.__speed > self.__max_speed] = self.__max_speed
        self.__speed[self.__speed < -self.__max_speed] = -self.__max_speed

    def __restrict_to_fn_boundaries(self):
        self.position[self.position < self.fn.min] = self.fn.min
        self.position[self.position > self.fn.max] = self.fn.max
