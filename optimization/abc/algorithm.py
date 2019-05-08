import sys
import numpy as np
from optimization.abc.food_source import FoodSource


class ABC(object):
    def __init__(self, colony_size, trials):
        self.__num_food_sources = int(colony_size / 2.0)
        self.__colony_size = colony_size
        self.__trials = trials

    def optimize(self, fn, stop_criterion):
        it = 0
        best_position = None
        best_fitness = sys.maxsize
        food_sources = self.__create_colony(fn)

        while not stop_criterion(iterations=it, evaluations=fn.evaluations):
            it = it + 1

            self.__employed_bees(food_sources)
            self.__calculate_probabilities(food_sources)
            self.__onlooker_bees(food_sources)
            self.__scout_bee(food_sources)

            best_food_source = min(food_sources, key=lambda x: x.fitness)

            if best_food_source.fitness < best_fitness:
                best_fitness = best_food_source.fitness
                best_position = best_food_source.position

        return best_position, best_fitness, []

    def __employed_bees(self, food_sources):
        for idx, food_source in enumerate(food_sources):
            reference = self.__find_reference(idx, food_sources)
            food_source.greedy_search(reference)

    def __onlooker_bees(self, food_sources):
        for bee in range(self.__num_food_sources):
            for idx, food_source in enumerate(food_sources):
                if food_source.probability < np.random.uniform():
                    continue

                reference = self.__find_reference(idx, food_sources)
                food_source.greedy_search(reference)

    def __scout_bee(self, food_sources):
        worst_food_source = max(food_sources, key=lambda f: f.trials)

        if worst_food_source.trials > self.__trials:
            worst_food_source.replace()

    @staticmethod
    def __calculate_probabilities(food_sources):
        f_sum = sum(map(lambda f: f.fitness, food_sources))

        for food_source in food_sources:
            probability = food_source.fitness / f_sum
            food_source.probability = probability

    def __find_reference(self, idx, food_sources):
        indexes = list(range(self.__num_food_sources))
        indexes.remove(idx)

        reference_index = np.random.choice(np.array(indexes))
        return food_sources[reference_index].position

    def __create_colony(self, fn):
        return [FoodSource(fn) for _ in range(self.__num_food_sources)]
