import math
import numpy as np


def find_best(swarm):
    return min(swarm, key=lambda p: p.fitness)


class InertiaDefinition(object):
    def __call__(self, *args, **kwargs):
        return 1.0, 1.0

    def update(self, *args, **kwargs):
        pass


class Inertia(object):
    @staticmethod
    def constant(weight=0.8):
        class ConstantInertia(InertiaDefinition):
            def __call__(self, *args, **kwargs):
                return weight, 1.0

        return ConstantInertia()

    @staticmethod
    def linear(min_weight=0.4, max_weight=0.9):
        class LinearInertia(InertiaDefinition):
            def __init__(self):
                self.__i = max_weight

            def __call__(self, *args, **kwargs):
                return self.__i, 1.0

            def update(self, stop_criterion, **kwargs):
                v = (max_weight - min_weight) * stop_criterion.count(**kwargs)
                self.__i = max_weight - (v / stop_criterion.limit)

        return LinearInertia()

    @staticmethod
    def clerc():
        class ClercRestrictionInertia(InertiaDefinition):
            def __call__(self, c1, c2):
                t = c1 + c2
                a = 2.0 - t
                b = math.sqrt((t ** 2) - (4.0 * t))

                k = 2.0 / math.fabs(a - b)
                return k, k

        return ClercRestrictionInertia()


class Communication(object):
    @staticmethod
    def fully_connected():
        class GlobalCommunication(object):
            def initialize(self, swarm):
                pass

            @property
            def name(self):
                return 'Global'

            def __call__(self, particle, swarm):
                return find_best(swarm)

        return GlobalCommunication()

    @staticmethod
    def socially_connected():
        class SocialCommunication(object):
            def initialize(self, swarm):
                pass

            @property
            def name(self):
                return 'Social'

            @staticmethod
            def __find_neighborhood(particle, swarm):
                swarm_length = len(swarm)
                particle_index = swarm.index(particle)

                left_index = particle_index - 1
                right_index = particle_index + 1

                if left_index < 0:
                    left_index = swarm_length - 1
                elif right_index >= swarm_length:
                    right_index = 0

                return [swarm[left_index], particle, swarm[right_index]]

            def __call__(self, particle, swarm):
                sub_swarm = self.__find_neighborhood(particle, swarm)
                return find_best(sub_swarm)

        return SocialCommunication()

    @staticmethod
    def nearest_connected(n_neighbors):
        class NearestCommunication(object):
            def initialize(self, swarm):
                pass

            @property
            def name(self):
                return 'Nearest'

            @staticmethod
            def __find_neighborhood(particle, swarm):
                def fn(p):
                    distance = p.distance_to(particle)
                    return distance, p

                with_distances = map(fn, swarm)
                result = sorted(with_distances, key=lambda x: x[0])
                return list(map(lambda x: x[1], result))[:n_neighbors]

            def __call__(self, particle, swarm):
                sub_swarm = self.__find_neighborhood(particle, swarm)
                return find_best(sub_swarm)

        return NearestCommunication()

    @staticmethod
    def focal_connected():
        class FocalCommunication(object):
            def __init__(self):
                self.__focal_particle = None

            def initialize(self, swarm):
                index = np.random.random() * len(swarm)
                self.__focal_particle = swarm[int(index)]

            @property
            def name(self):
                return 'Focal'

            def __call__(self, particle, swarm):
                if particle == self.__focal_particle:
                    return find_best(swarm)
                else:
                    return self.__focal_particle

        return FocalCommunication()
