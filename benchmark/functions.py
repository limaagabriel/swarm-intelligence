from deap.benchmarks import sphere, rosenbrock, rastrigin
from deap.benchmarks.tools import translate, rotate


class ObjectiveFunction(object):
    def __init__(self, name, dim, minf, maxf, rot=None, trans=None):
        self.name = name
        self.dim = dim
        self.minf = minf
        self.maxf = maxf
        self.rot = rot
        self.trans = trans

    def evaluate(self, x):
        pass


class SphereFunction(ObjectiveFunction):
    def __init__(self, dim, rot=None, trans=None):
        super(SphereFunction, self).__init__('Sphere', dim, -100.0, 100.0, rot, trans)
        self.func = sphere
        if self.rot:
            rotation = rotate(self.rot)
            self.func = rotation(self.func)

        if self.trans:
            translation = translate(self.trans)
            self.func = translation(self.func)

    def __call__(self, x):
        return self.func(x)[0]


class RosenbrockFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(RosenbrockFunction, self).__init__('Rosenbrock', dim, -30.0, 30.0)

    def __call__(self, x):
        return rosenbrock(x)[0]


class RastriginFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(RastriginFunction, self).__init__('Rastrigin', dim, -5.12, 5.12)

    def __call__(self, x):
        return rastrigin(x)[0]
