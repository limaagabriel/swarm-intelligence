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
    def position(self):
        return self.__position

    def update(self, a, b, c1, c2):
        r1 = np.random.random()
        r2 = np.random.random()

        inertia = a * self.__speed
        cognitive_component = b * c1 * r1 * (self.cognitive_reference - self.__position)
        social_component = b * c2 * r2 * (self.social_reference - self.__position)

        self.__speed = inertia + cognitive_component + social_component
        self.__position = self.__position + self.__speed
        self.__restrict_to_fn_boundaries()

    def evaluate(self):
        fitness = self.__fn(self.__position)

        if fitness <= self.__fitness:
            self.__fitness = fitness
            self.cognitive_reference = self.__position

    def distance_to(self, other):
        def squared_euclidean(a, b):
            return np.sum(np.square(a - b))

        return squared_euclidean(self.position, other.position)

    def __restrict_to_fn_boundaries(self):
        self.__position[self.__position < self.__fn.minf] = self.__fn.minf
        self.__position[self.__position > self.__fn.maxf] = self.__fn.maxf

    def __eq__(self, other):
        return self.id == other.id


class PSO(object):
    def __init__(self, n_particles, social, cognitive, inertia, communication):
        self.__n_particles = n_particles

        self.__c1 = cognitive
        self.__c2 = social

        self.__inertia = inertia
        self.__communication = communication

    def optimize(self, fn, iterations=10000):
        it = 0
        fitness_evolution = []

        swarm = self.__create_swarm(fn)
        self.__communication.initialize(swarm)

        while it < iterations:
            it = it + 1
            a, b = self.__inertia(it, self.__c1, self.__c2)

            for particle in swarm:
                p = self.__communication(particle, swarm)
                particle.social_reference = p.cognitive_reference
                particle.update(a, b, self.__c1, self.__c2)

            for particle in swarm:
                particle.evaluate()

        p = self.__find_best(swarm)
        return p.position, p.fitness, fitness_evolution

    def __create_swarm(self, fn):
        mapper = lambda _: Particle(fn)
        iterable = range(self.__n_particles)

        return list(map(mapper, iterable))

    def __find_best(self, swarm):
        return min(swarm, key=lambda p: p.fitness)
