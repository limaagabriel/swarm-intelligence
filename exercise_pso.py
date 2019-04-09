from pso import PSO
from itertools import product
from benchmark import functions
from pso.strategies import Inertia
from pso.strategies import Communication

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dimensions = 30
n_particles = 30
iterations = 10000
statistical_sample_size = 30
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
    # Communication.socially_connected(),
    Communication.focal_connected()
]

cases = list(product(objective_functions, inertia_strategies))
communication_names = list(map(lambda x: x.name, communication_strategies))

if not os.path.exists('results'):
    os.mkdir('results')

for fn, inertia_strategy in cases:
    evolution_acc = np.zeros((len(communication_strategies), iterations))
    fitness_sample = np.zeros((len(communication_strategies), statistical_sample_size))

    for index, communication_strategy in enumerate(communication_strategies):
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
            fitness_sample[index, i] = best_fitness
            evolution_acc[index] += fitness_evolution / float(statistical_sample_size)

    plt.plot(evolution_acc.transpose())
    plt.xlabel('Number of iterations')
    plt.ylabel('Fitness')
    plt.legend(communication_names)

    plt.savefig(os.path.join('results', '{}{}Evolution.png'.format(fn.name, inertia_strategy.__class__.__name__)))
    plt.clf()
    plt.close()

    plt.boxplot(fitness_sample.transpose())
    plt.xticks(np.arange(1, len(communication_names) + 1), communication_names)
    plt.xlabel('Communication strategy')
    plt.ylabel('Best fitness')

    plt.savefig(os.path.join('results', '{}{}Boxplot.png'.format(fn.name, inertia_strategy.__class__.__name__)))
    plt.clf()
    plt.close()
