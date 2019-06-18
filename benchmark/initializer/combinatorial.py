import numpy as np
from benchmark.initializer import SearchSpaceInitializer
from benchmark.initializer import SearchSpaceInitializerFactory


class CombinatorialInitializer(SearchSpaceInitializerFactory):
    @staticmethod
    def uniform_random(choice_range):
        class UniformRandomCombinatorialInitializer(SearchSpaceInitializer):
            def __call__(self):
                return np.random.choice(choice_range)

        return UniformRandomCombinatorialInitializer()

    @staticmethod
    def given_point(point):
        class CombinatorialInitializerByGivenPoint(SearchSpaceInitializer):
            def __call__(self):
                return point

        return CombinatorialInitializerByGivenPoint()
