import sys
import numpy as np
from abc import abstractmethod
from optimization import Agent


class BaseAnt(Agent):
    def __init__(self, fn, initializer):
        super().__init__(fn, initializer)

        self.current_state = self.position
        self.failed = False
        self.position = []

    def make_tour(self, *args, **kwargs):
        self.failed = False
        self.position = [self.current_state]

        while len(self.position) < self.fn.num_states:
            neighborhood = self.find_neighborhood()
            if len(neighborhood) == 0:
                self.failed = True
                self.fitness = sys.maxsize
                break

            next_state = self.state_transition_rule(neighborhood, *args, **kwargs)
            self.current_state = next_state
            self.position.append(next_state)

        if not self.failed:
            self.fitness = self.get_distance()

    @abstractmethod
    def state_transition_rule(self, *args, **kwargs):
        pass

    @abstractmethod
    def local_update_rule(self, *args, **kwargs):
        pass

    @abstractmethod
    def global_update_rule(self, *args, **kwargs):
        pass

    def replace(self):
        super().replace()
        self.current_state = self.position
        self.position = []

    def get_distance(self):
        distance = 0
        current_state = self.position[0]
        for state in self.position[1:]:
            distance = distance + self.fn.graph[current_state, state]
            current_state = state

        return distance

    def find_neighborhood(self):
        neighborhood = []
        for i in range(self.fn.num_states):
            a = i in self.position
            b = i == self.current_state
            c = self.fn.graph[self.current_state, i] <= 0

            if a or b or c:
                continue
            neighborhood.append(i)
        return np.array(neighborhood)
