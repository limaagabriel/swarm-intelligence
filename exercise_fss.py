from optimization.fss import FSS
from optimization.fss.strategies import Step

from benchmark.stop import StopCriterion
from benchmark.functions.continuous import SphereFunction
from benchmark.initializer.continuous import  ContinuousInitializer

dimensions = 30
num_fishes = 30
initializer = ContinuousInitializer.uniform_random(SphereFunction.bounds, dimensions)

fss = FSS(num_fishes, Step.linear())
print(fss.optimize(SphereFunction(dimensions), initializer, StopCriterion.fn_evaluation(500000)))

