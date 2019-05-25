import os
import tqdm
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from util import read_padded_csv

from optimization.pso import PSO
from optimization.fss import FSS
from optimization.abc import ABC

from benchmark.stop import StopCriterion
from benchmark.functions.continuous import SphereFunction
from benchmark.functions.continuous import RastriginFunction
from benchmark.functions.continuous import RosenbrockFunction

from optimization.fss.strategies import Step
from optimization.pso.strategies import Inertia
from optimization.pso.strategies import Communication

dimensions = 30
sample_size = 30
fn_evaluations = 500000
should_evaluate = True
matplotlib.use('Agg')

objective_functions = [
    SphereFunction,
    RastriginFunction,
    RosenbrockFunction
]

methods = ['pso', 'fss', 'abc']

target_directory = os.path.join('results', 'compare')
plot_directory = os.path.join(target_directory, 'plot')
registry_directory = os.path.join(target_directory, 'reg')
sample_base_path = os.path.join(registry_directory, '{}_{}_b.csv')
evolution_base_path = os.path.join(registry_directory, '{}_{}_a.csv')

if not os.path.exists('results'):
    os.mkdir('results')

if not os.path.exists(target_directory):
    os.mkdir(target_directory)
    os.mkdir(registry_directory)
    os.mkdir(plot_directory)


def evaluate(objective_function, paths):
    def run_method(method, fn, parameters):
        fn_instance = fn(dimensions)
        method_instance = method(**parameters)
        stop_criterion = StopCriterion.fn_evaluation(fn_evaluations)
        return method_instance.optimize(fn_instance, stop_criterion=stop_criterion)

    for _ in tqdm.tqdm(range(sample_size), total=sample_size):
        results = {
            'pso': run_method(PSO, objective_function, {
                'n_particles': 30,
                'social': 2.05,
                'cognitive': 2.05,
                'communication': Communication.socially_connected(),
                'inertia': Inertia.linear(min_weight=0.4, max_weight=0.9)
            }),

            'fss': run_method(FSS, objective_function, {
                'n_fishes': 30,
                'step': Step.linear(individual=(0.1, 0.001),
                                    volitive=(0.01, 0.001))
            }),

            'abc': run_method(ABC, objective_function, {
                'colony_size': 30,
                'trials': 100
            })
        }

        for m in methods:
            with open(paths[m]['sp'], 'a+') as f:
                f.write(str(results[m][1]))
                f.write('\n')

            with open(paths[m]['ev'], 'a+') as f:
                tracker = results[m][2]
                evolution = tracker.fn_evaluations.tolist()
                f.write(','.join(map(lambda t: '{:.2f}'.format(t), evolution)))
                f.write('\n')


def plot(objective_function, paths):
    figure_1 = plt.figure()
    figure_2 = plt.figure()
    samples = np.zeros((sample_size, len(methods)))

    sample_figure = figure_1.add_subplot(111)
    evolution_figure = figure_2.add_subplot(111)
    names = list(map(lambda m: m.upper(), methods))

    for i, method in tqdm.tqdm(enumerate(methods), total=len(methods)):
        sample = np.genfromtxt(paths[method]['sp'])
        evolution = read_padded_csv(paths[method]['ev'], 0.0)
        evolution[np.isnan(evolution)] = 0.0

        samples[:, i] = sample.copy()
        evolution_figure.plot(evolution.mean(axis=0))

    sample_figure.boxplot(samples)
    sample_figure.set_xticklabels(names)
    sample_figure.set_xlabel('Algorithm')
    sample_figure.set_ylabel('Best fitness')
    sample_figure.set_title('Best fitness: {}'.format(objective_function))

    evolution_figure.set_xlabel('Function evaluation')
    evolution_figure.set_ylabel('Fitness')
    evolution_figure.legend(names)
    evolution_figure.set_title('Evolution: {}'.format(objective_function))

    sample_name = 'sp_{}.png'.format(objective_function)
    evolution_name = 'ev_{}.png'.format(objective_function)

    sample_figure.figure.savefig(os.path.join(plot_directory, sample_name))
    evolution_figure.figure.savefig(os.path.join(plot_directory, evolution_name))
    plt.clf()
    plt.close()


def main():
    for objective_function in objective_functions:
        operation = 'Plotting'
        if should_evaluate:
            operation = 'Evaluating on'

        print('{} {}'.format(operation, objective_function.__name__))

        paths = {
            m:  {
                'sp': sample_base_path.format(m, objective_function.__name__),
                'ev': evolution_base_path.format(m, objective_function.__name__)
            } for m in methods
        }

        if should_evaluate:
            evaluate(objective_function, paths)
            continue

        plot(objective_function.__name__, paths)


if __name__ == '__main__':
    main()

