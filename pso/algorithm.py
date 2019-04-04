from pso.particle import Particle


class PSO(object):
    def __init__(self, n_particles, social, cognitive, inertia, communication):
        self.__n_particles = n_particles

        self.__c1 = cognitive
        self.__c2 = social

        self.__inertia = inertia
        self.__communication = communication

    def optimize(self, fn, iterations=10000):
        fitness_evolution = []

        swarm = self.__create_swarm(fn)
        self.__communication.initialize(swarm)

        for it in range(iterations):
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
        swarm_range = range(self.__n_particles)
        return [Particle(fn) for _ in swarm_range]

    @staticmethod
    def __find_best(swarm):
        return min(swarm, key=lambda p: p.fitness)
