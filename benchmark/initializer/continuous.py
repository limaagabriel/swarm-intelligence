import numpy as np
from benchmark.initializer import SearchSpaceInitializer
from benchmark.initializer import SearchSpaceInitializerFactory


class ContinuousInitializer(SearchSpaceInitializerFactory):
    @staticmethod
    def uniform_random(bounds, dimensions, region_scaling=True):
        class UniformRandomContinuousInitializer(SearchSpaceInitializer):
            def __call__(self):
                a = bounds[0]
                b = bounds[1]

                if region_scaling:
                    a = bounds[1] / 2.0
                return np.random.uniform(a, b, dimensions)

        return UniformRandomContinuousInitializer()

    @staticmethod
    def given_point(point):
        class ContinuousInitializerByGivenPoint(SearchSpaceInitializer):
            def __call__(self):
                return point

        return ContinuousInitializerByGivenPoint()
