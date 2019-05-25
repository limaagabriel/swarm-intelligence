from optimization.abc import ABC
from benchmark.stop import StopCriterion
from benchmark.functions.continuous import SphereFunction
from benchmark.functions.continuous import RastriginFunction
from benchmark.functions.continuous import RosenbrockFunction

dimensions = 30
colony_size = 30
trials = 100

abc = ABC(colony_size=colony_size, trials=trials)
print(abc.optimize(SphereFunction(dimensions), StopCriterion.fn_evaluation(500000)))
print(abc.optimize(RastriginFunction(dimensions), StopCriterion.fn_evaluation(500000)))
print(abc.optimize(RosenbrockFunction(dimensions), StopCriterion.fn_evaluation(500000)))


