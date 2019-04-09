import numpy as np


class ObjectiveFunction(object):
    def __init__(self, name, dimensions, bounds):
        self.name = name
        self.min = bounds[0]
        self.max = bounds[1]
        self.dimensions = dimensions

    def random_region_scaling(self):
        x = self.max / 2.0
        return (np.random.random(self.dimensions) * x) + x

    def __call__(self, x):
        return 0

    @np.vectorize
    def evaluate(self, x, y):
        return self(np.array([x, y]))


class SphereFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(SphereFunction, self).__init__('Sphere', dim, (-100.0, 100.0))

    def __call__(self, x):
        return (x ** 2).sum()


class RosenbrockFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(RosenbrockFunction, self).__init__('Rosenbrock', dim, (-30.0, 30.0))

    def __call__(self, x):
        a = x[1:] - (x[:-1] ** 2)
        b = x[:1] - 1
        y = 100 * (a ** 2) + (b ** 2)

        return y.sum()


class RastriginFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(RastriginFunction, self).__init__('Rastrigin', dim, (-5.12, 5.12))

    def __call__(self, x):
        y = (x ** 2) - 10 * np.cos(2.0 * np.pi * x) + 10
        return y.sum()
