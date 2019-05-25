import sys
import numpy as np
from optimization import Agent


class Ant(Agent):
    def __init__(self, fn):
        super().__init__(fn)

        self.__current_state = self.position
        self.__failed = False
        self.position = []

    def make_travel(self, alpha, beta, trails):
        self.__failed = False
        self.position = [self.__current_state]

        while len(self.position) < self.fn.num_states:
            neighborhood = self.__find_neighborhood()
            if len(neighborhood) == 0:
                self.__failed = True
                self.fitness = sys.maxsize
                break

            possible_trails = trails[self.__current_state, neighborhood]
            attractiveness = self.fn.graph[self.__current_state, neighborhood] ** -1

            targets = (possible_trails ** alpha) * (attractiveness ** beta)
            probabilities = targets / targets.sum()
            next_state = np.random.choice(neighborhood, p=probabilities)

            self.__current_state = next_state
            self.position.append(next_state)

        if not self.__failed:
            self.fitness = self.__get_distance()

    def get_pheromone(self, q):
        n = self.fn.num_states
        pheromone = np.zeros((n, n))

        if not self.__failed:
            step = q / self.fitness
            stack = self.position.copy()
            current_position = stack.pop()

            while len(stack) > 0:
                previous_position = stack.pop()
                pheromone[previous_position, current_position] = step
                pheromone[current_position, previous_position] = step
                current_position = previous_position

        return pheromone

    def __get_distance(self):
        distance = 0
        current_state = self.position[0]
        for state in self.position[1:]:
            distance = distance + self.fn.graph[current_state, state]
            current_state = state

        return distance

    def __find_neighborhood(self):
        neighborhood = []
        for i in range(self.fn.num_states):
            a = i in self.position
            b = i == self.__current_state
            c = self.fn.graph[self.__current_state, i] <= 0

            if a or b or c:
                continue
            neighborhood.append(i)
        return np.array(neighborhood)


