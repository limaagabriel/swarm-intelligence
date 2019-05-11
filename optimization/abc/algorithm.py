import sys
import numpy as np
from optimization import SwarmOptimizationMethod
from optimization.abc.food_source import FoodSource


class ABC(SwarmOptimizationMethod):
    def __init__(self, colony_size, trials):
        self.__num_food_sources = int(colony_size / 2.0)
        self.__colony_size = colony_size
        self.__trials = trials

        super().__init__(FoodSource, self.__num_food_sources)

    def __call__(self, fn, stop_criterion, tracker):
        it = 0
        best_position = None
        best_fitness = sys.maxsize

        while not stop_criterion(iterations=it, evaluations=fn.evaluations):
            it = it + 1

            self.__employed_bees()
            self.__calculate_probabilities()
            self.__onlooker_bees()
            self.__scout_bee()

            best_food_source = self.get_best_agent()
            if best_food_source.fitness < best_fitness:
                best_fitness = best_food_source.fitness
                best_position = best_food_source.position
            tracker.track_by_iterations(best_fitness)

        return best_position, best_fitness

    def __employed_bees(self):
        for idx, food_source in enumerate(self.swarm):
            reference = self.__find_reference(idx, self.swarm)
            food_source.greedy_search(reference)

    def __onlooker_bees(self):
        for bee in range(self.__num_food_sources):
            for idx, food_source in enumerate(self.swarm):
                if food_source.probability < np.random.uniform():
                    continue

                reference = self.__find_reference(idx, self.swarm)
                food_source.greedy_search(reference)

    def __scout_bee(self):
        worst_food_source = max(self.swarm, key=lambda f: f.trials)

        if worst_food_source.trials > self.__trials:
            worst_food_source.replace()

    def __calculate_probabilities(self):
        f_sum = sum(map(lambda f: f.fitness, self.swarm))

        for food_source in self.swarm:
            probability = 1.0 - (food_source.fitness / f_sum)
            food_source.probability = probability

    def __find_reference(self, idx, food_sources):
        indexes = list(range(self.__num_food_sources))
        indexes.remove(idx)

        reference_index = np.random.choice(np.array(indexes))
        return food_sources[reference_index].position

