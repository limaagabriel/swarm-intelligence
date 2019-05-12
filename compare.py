import os
import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from optimization.pso import PSO
from optimization.fss import FSS
from optimization.abc import ABC

from benchmark import functions
from benchmark.stop import StopCriterion

from optimization.fss.strategies import Step
from optimization.pso.strategies import Inertia
from optimization.pso.strategies import Communication

dimensions = 30
sample_size = 30
fn_evaluations = 500000
matplotlib.use('Agg')

objective_functions = [
    functions.SphereFunction,
    functions.RastriginFunction,
    functions.RosenbrockFunction
]

methods = ['pso', 'fss', 'abc']

target_directory = os.path.join('results', 'compare')
plot_directory = os.path.join(target_directory, 'plot')
registry_directory = os.path.join(target_directory, 'reg')
sample_base_path = os.path.join(registry_directory, '{}_{}_b.csv')
evolution_base_path = os.path.join(registry_directory, '{}_{}_a.csv')

if not os.path.exists(target_directory):
    os.mkdir(target_directory)
    os.mkdir(registry_directory)
    os.mkdir(plot_directory)


def run_method(method, fn, parameters):
    fn_instance = fn(dimensions)
    method_instance = method(**parameters)
    stop_criterion = StopCriterion.fn_evaluation(fn_evaluations)
    return method_instance.optimize(fn_instance, stop_criterion=stop_criterion)


if __name__ == '__main__':
    for objective_function in objective_functions:
        print('Evaluating on {}'.format(objective_function.__name__))

        paths = {
            m:  {
                'sp': sample_base_path.format(m, objective_function.__name__),
                'ev': evolution_base_path.format(m, objective_function.__name__)
            } for m in methods
        }

        for sample_id in tqdm.tqdm(range(sample_size), total=sample_size):
            results = {
                'pso': run_method(PSO, objective_function, {
                    'n_particles': 30,
                    'social': 2.05,
                    'cognitive': 2.05,
                    'communication': Communication.socially_connected(),
                    'inertia': Inertia.linear(min_weight=0.4,max_weight=0.9)
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
