from optimization.pso.particle import Particle
from optimization import SwarmOptimizationMethod
import matplotlib.pyplot as plt
import numpy as np


class PSO(SwarmOptimizationMethod):
    def __init__(self, n_particles, social, cognitive, inertia, communication):
        super().__init__(Particle, n_particles)

        self.__c1 = cognitive
        self.__c2 = social

        self.__inertia = inertia
        self.__communication = communication

    def __call__(self, fn, stop_criterion, tracker):
        it = 0
        best_particle = None
        self.__communication.initialize(self.swarm)

        while not stop_criterion(iterations=it, fn=fn):
            it = it + 1
            a, b = self.__inertia(it, self.__c1, self.__c2)

            for particle in self.swarm:
                p = self.__communication(particle, self.swarm)
                particle.social_reference = p.cognitive_reference
                particle.update(a, b, self.__c1, self.__c2)

            for particle in self.swarm:
                particle.evaluate()

            best_particle = self.get_best_agent()
            tracker.track_by_iterations(best_particle.fitness)

        return best_particle.position, best_particle.fitness

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
