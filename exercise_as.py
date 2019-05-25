import matplotlib.pyplot as plt

from optimization.aco import AntSystem
from benchmark.stop import StopCriterion
from benchmark.functions.combinatorial import TSP

tsp = TSP('gr17')
aco = AntSystem(tsp.num_states, 1, 5, 0.5, 100)
stop_criterion = StopCriterion.iteration_limit(2500)

travel, cost, tracker = aco.optimize(tsp, stop_criterion=stop_criterion)
print(travel, cost)

y = tracker.iterations
x = range(y.shape[0])

plt.plot(x, y)
plt.xlabel('Iterations')
plt.ylabel('Smallest travel distance')
plt.title('TSP')
plt.show()
plt.clf()
plt.close()
