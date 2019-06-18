from util import foreach
from itertools import product


class BruteCombinatorialOptimizer(object):
    def optimize(self, fn):
        def distance(solution):
            src = solution[:-1]
            dest = solution[1:]
            return fn.graph[src, dest].sum()

        solutions = self.__backtrack(fn)
        best_solution = sorted(solutions, key=distance)[0]

        return best_solution, distance(best_solution)

    @staticmethod
    def __backtrack(fn):
        solutions = []

        def get_next_moves(solution):
            method = BruteCombinatorialOptimizer.__verify_neighbor
            return filter(lambda x: method(fn, solution, x), range(fn.num_states))

        def move(solution):
            if len(solution) == fn.num_states:
                solutions.append(solution)

            for next_move in list(get_next_moves(solution)):
                s = solution.copy()
                s.append(next_move)
                move(s)

        move([])
        return solutions

    @staticmethod
    def __verify_neighbor(fn, solution, j):
        if len(solution) == 0:
            return True

        i = solution[-1]

        a = i != j
        b = j not in solution
        c = fn.graph[i, j] > 0

        return a and b and c
