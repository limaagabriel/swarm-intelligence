from optimization.abc import ABC
from benchmark.stop import StopCriterion
from benchmark.initializer.continuous import ContinuousInitializer
from benchmark.functions.continuous import SphereFunction
from benchmark.functions.continuous import RastriginFunction
from benchmark.functions.continuous import RosenbrockFunction

dimensions = 30
colony_size = 30
trials = 100

sphere_initializer = ContinuousInitializer.uniform_random(SphereFunction.bounds, dimensions)
rastrigin_initializer = ContinuousInitializer.uniform_random(RastriginFunction.bounds, dimensions)
rosenbrock_initializer = ContinuousInitializer.uniform_random(RosenbrockFunction.bounds, dimensions)

abc = ABC(colony_size, trials)
print(abc.optimize(SphereFunction(dimensions), sphere_initializer, StopCriterion.fn_evaluation(500000)))
print(abc.optimize(RastriginFunction(dimensions), rastrigin_initializer, StopCriterion.fn_evaluation(500000)))
print(abc.optimize(RosenbrockFunction(dimensions), rosenbrock_initializer, StopCriterion.fn_evaluation(500000)))


