import sys
import uuid
import numpy as np
from abc import ABC
from abc import abstractmethod


class FitnessTracker(object):
    def __init__(self):
        self.__it = []
        self.__ev = []

    @property
    def iterations(self):
        return np.array(self.__it.copy())

    @property
    def fn_evaluations(self):
        return np.array(self.__ev.copy())

    def track_by_iterations(self, fitness):
        self.__it.append(fitness)

    def track_by_fn_evaluations(self, fitness):
        self.__ev.append(fitness)


class Agent(ABC):
    def __init__(self, fn, initializer):
        self.__id = str(uuid.uuid4())
        self.__initializer = initializer

        self.fn = fn
        self.position = initializer()
        self.fitness = fn(self.position)

    def replace(self):
        self.position = self.__initializer()
        self.fitness = self.fn(self.position)

    @property
    def id(self):
        return self.__id

    def distance_to(self, other):
        def squared_euclidean(a, b):
            return np.sum(np.square(a - b))

        return squared_euclidean(self.position, other.position)

    def is_out_of_bounds(self):
        lower_bounds = self.position <= self.fn.min
        upper_bounds = self.position >= self.fn.max

        return lower_bounds.any() or upper_bounds.any()

    def __eq__(self, other):
        return self.id == other.id


class SwarmOptimizationMethod(ABC):
    def __init__(self, agent_class, swarm_size):
        self.__agent_class = agent_class
        self.__size = swarm_size
        self.swarm = []

        self.initializer = None
        self.best_position = None
        self.best_fitness = sys.maxsize

    def create_swarm(self, fn, initializer):
        self.swarm = [self.__agent_class(fn, initializer) for _ in range(self.__size)]

    def optimize(self, fn, initializer, stop_criterion):
        self.initializer = initializer
        self.create_swarm(fn, initializer)
        self.update_best_solution()
        tracker = FitnessTracker()

        def feed_tracker(*_):
            tracker.track_by_fn_evaluations(self.best_fitness)

        fn.on_call(feed_tracker)
        position, fitness = self(fn, stop_criterion, tracker)
        return position, fitness, tracker

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def __get_best_agent(self):
        return min(self.swarm, key=lambda a: a.fitness)

    def update_best_solution(self):
        agent = self.__get_best_agent()

        if agent.fitness < self.best_fitness:
            self.best_fitness = agent.fitness
            self.best_position = agent.position.copy()
