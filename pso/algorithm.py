from pso.particle import Particle
import matplotlib.pyplot as plt
import numpy as np


class PSO(object):
    def __init__(self, n_particles, social, cognitive, inertia, communication):
        self.__n_particles = n_particles

        self.__c1 = cognitive
        self.__c2 = social

        self.__inertia = inertia
        self.__communication = communication

    def optimize(self, fn, stop_criterion, visualize=False):
        it = 0
        best_particle = None
        fitness_evolution = []

        swarm = self.__create_swarm(fn)
        self.__communication.initialize(swarm)

        while not stop_criterion(iterations=it, fn=fn):
            it = it + 1
            a, b = self.__inertia(it, self.__c1, self.__c2)

            if visualize:
                self.__visualize(swarm, fn)

            for particle in swarm:
                p = self.__communication(particle, swarm)
                particle.social_reference = p.cognitive_reference
                particle.update(a, b, self.__c1, self.__c2)

            for particle in swarm:
                particle.evaluate()

            best_particle = self.__find_best(swarm)
            fitness_evolution.append(best_particle.fitness)

        return best_particle.position, best_particle.fitness, np.array(fitness_evolution)

    @staticmethod
    def __visualize(swarm, fn):
        def position_mapper(particle):
            return particle.position

        def speed_mapper(particle):
            return particle.speed

        num_samples = fn.max - fn.min
        domain = np.linspace(fn.min, fn.max, num_samples)
        x, y = np.meshgrid(domain.copy(), domain.copy())
        z = fn.evaluate(fn, x, y)

        p = np.array(list(map(position_mapper, swarm)))
        s = np.array(list(map(speed_mapper, swarm)))

        colors = np.zeros(p[:,0].shape)
        colors[0] = 1
        plt.pcolormesh(x, y, z, cmap='RdBu')
        plt.quiver(p[:,0], p[:,1], s[:,0], s[:,1], colors)
        plt.show()
        plt.clf()
        plt.close()

    def __create_swarm(self, fn):
        swarm_range = range(self.__n_particles)
        return [Particle(fn) for _ in swarm_range]

    @staticmethod
    def __find_best(swarm):
        return min(swarm, key=lambda p: p.fitness)
