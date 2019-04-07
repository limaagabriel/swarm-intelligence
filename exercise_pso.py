from pso import PSO
from itertools import product
from benchmark import functions
from pso.strategies import Inertia
from pso.strategies import Communication

import numpy as np
import matplotlib.pyplot as plt

dimensions = 2
n_particles = 30
iterations = 10000
statistical_sample_size = 1
social_coefficient = 2.05
cognitive_coefficient = 2.05

objective_functions = [
    functions.SphereFunction(dimensions),
    functions.RastriginFunction(dimensions),
    functions.RosenbrockFunction(dimensions)
]

inertia_strategies = [
    Inertia.constant(weight=0.8),
    Inertia.linear(min_weight=0.4, max_weight=0.9, iterations=10000),
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
        evolution_acc = np.zeros(iterations)
        fitness_sample = np.zeros(statistical_sample_size)

        for i in range(statistical_sample_size):
            pso = PSO(n_particles=n_particles,
                      social=social_coefficient,
                      cognitive=cognitive_coefficient,
                      inertia=inertia_strategy,
                      communication=communication_strategy)

            best, best_fitness, fitness_evolution = pso.optimize(fn, iterations=iterations)

            result_summary = (
                i + 1,
                communication_strategy.__class__.__name__,
                inertia_strategy.__class__.__name__,
                fn.name,
                best_fitness
            )

            print('{} - {}, {}, {}: {}'.format(*result_summary))
            fitness_sample[i] = best_fitness
            evolution_acc += fitness_evolution / float(statistical_sample_size)


