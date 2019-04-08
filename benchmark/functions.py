from deap.benchmarks import sphere, rosenbrock, rastrigin
from deap.benchmarks.tools import translate, rotate


class ObjectiveFunction(object):
    def __init__(self, name, dimensions, min, max):
        self.name = name
        self.min = min
        self.max = max
        self.dimensions = dimensions

    def evaluate(self, x):
        pass


class SphereFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(SphereFunction, self).__init__('Sphere', dim, -100.0, 100.0)

    @staticmethod
    def region_scaling():
        mean = 50
        std_dev = 5
        return mean, std_dev

    def __call__(self, x):
        return sphere(x)[0]


class RosenbrockFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(RosenbrockFunction, self).__init__('Rosenbrock', dim, -30.0, 30.0)

    @staticmethod
    def region_scaling():
        mean = 20
        std_dev = 2
        return mean, std_dev

    def __call__(self, x):
        return rosenbrock(x)[0]


class RastriginFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(RastriginFunction, self).__init__('Rastrigin', dim, -5.12, 5.12)

    @staticmethod
    def region_scaling():
        mean = -3
        std_dev = 1
        return mean, std_dev

    def __call__(self, x):
        return rastrigin(x)[0]
