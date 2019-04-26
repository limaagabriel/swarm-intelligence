from fss import FSS
from fss.strategies import Step
from benchmark.stop import StopCriterion
from benchmark.functions import SphereFunction

dimensions = 30
num_fishes = 30

fss = FSS(num_fishes, Step.linear())
print(fss.optimize(SphereFunction(dimensions), StopCriterion.fn_evaluation(500000)))

