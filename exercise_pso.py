from pso import PSO
from itertools import product
from benchmark import functions
from pso.strategies import Inertia
from pso.strategies import Communication

social = 2.05
cognitive = 2.05
dimensions = 30
n_particles = 30
statistical_sample_size = 1

objective_functions = [
    functions.SphereFunction(dimensions)
    # functions.RastriginFunction(dimensions),
    # functions.RosenbrockFunction(dimensions)
]

inertia_strategies = [
    #Inertia.constant(weight=0.8),
    #Inertia.linear(min_weight=0.4, max_weight=0.9, iterations=10000),
    Inertia.clerc()
]

communication_strategies = [
    Communication.fully_connected(),
    Communication.nearest_connected(n_neighbors=2),
    Communication.socially_connected(),
    Communication.focal_connected()
]

cases = list(product(objective_functions, inertia_strategies))

for fn, inertia_strategy in cases:
    for communication_strategy in communication_strategies:
        for i in range(1, statistical_sample_size + 1):
            pso = PSO(n_particles=n_particles,
                      social=social,
                      cognitive=cognitive,
                      inertia=inertia_strategy,
                      communication=communication_strategy)

            best, best_fitness, fitness_evolution = pso.optimize(fn)
            print('{} - {}, {}, {}: {}'.format(
                i, communication_strategy.__class__.__name__, inertia_strategy.__class__.__name__, fn.name, best_fitness))
