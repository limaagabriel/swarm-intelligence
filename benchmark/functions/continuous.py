import numpy as np
from abc import ABC
from abc import abstractmethod


class ObjectiveFunction(ABC):
    bounds = (-1, 1)

    def __init__(self, name, dimensions):
        self.name = name
        self.min = self.bounds[0]
        self.max = self.bounds[1]
        self.dimensions = dimensions

        self.__listeners = []
        self.__evaluations = 0

    def on_call(self, fn):
        self.__listeners.append(fn)

    @property
    def evaluations(self):
        return self.__evaluations

    def __call__(self, x):
        self.__evaluations += 1
        r = self.run(x)

        for listener in self.__listeners:
            listener(self, r)
        return r

    @abstractmethod
    def run(self, x):
        pass

    @np.vectorize
    def evaluate(self, x, y):
        return self(np.array([x, y]))


class SphereFunction(ObjectiveFunction):
    bounds = (-100.0, 100.0)

    def __init__(self, dim):
        super(SphereFunction, self).__init__('Sphere', dim)

    def run(self, x):
        return (x ** 2).sum()


class RosenbrockFunction(ObjectiveFunction):
    bounds = (-30.0, 30.0)

    def __init__(self, dim):
        super(RosenbrockFunction, self).__init__('Rosenbrock', dim)

    def run(self, x):
        a = x[1:] - (x[:-1] ** 2)
        b = x[:1] - 1
        y = 100 * (a ** 2) + (b ** 2)

        return y.sum()


class RastriginFunction(ObjectiveFunction):
    bounds = (-5.12, 5.12)

    def __init__(self, dim):
        super(RastriginFunction, self).__init__('Rastrigin', dim)

    def run(self, x):
        y = (x ** 2) - 10 * np.cos(2.0 * np.pi * x) + 10
        return y.sum()
