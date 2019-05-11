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
    def __init__(self, fn):
        self.__id = str(uuid.uuid4())

        self.fn = fn
        self.position = fn.search_space_initializer()
        self.fitness = fn(self.position)

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

    def create_swarm(self, fn):
        self.swarm = [self.__agent_class(fn) for _ in range(self.__size)]

    def optimize(self, fn, stop_criterion):
        self.create_swarm(fn)
        tracker = FitnessTracker()

        def feed_tracker(*_):
            agent = self.get_best_agent()
            tracker.track_by_fn_evaluations(agent.fitness)

        fn.on_call(feed_tracker)
        position, fitness = self(fn, stop_criterion, tracker)
        return position, fitness, tracker

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def get_best_agent(self):
        return min(self.swarm, key=lambda a: a.fitness)

