import os
import sys
import numpy as np
from lxml import etree
from abc import ABC, abstractmethod


class ObjectiveFunction(ABC):
    __sets_home = os.path.join('benchmark', 'functions', 'sets')

    def __init__(self, name, path):
        self.name = name
        self.graph = self.__read(os.path.join(self.__sets_home, path))
        self.num_states = self.graph.shape[0]

        self.__listeners = []
        self.__evaluations = 0

    @property
    def states(self):
        return np.arange(self.num_states)

    @staticmethod
    def __read(path):
        tree = etree.parse(path)
        graph = tree.getroot().find('graph')
        vertices = list(graph.findall('vertex'))
        mat = np.zeros((len(vertices), len(vertices)))

        for i, vertex in enumerate(vertices):
            for edge in vertex.findall('edge'):
                j = int(float(edge.text))
                cost = float(edge.get('cost'))
                mat[i, j] = cost

        return mat

    def on_call(self, fn):
        pass

    @property
    def evaluations(self):
        return self.__evaluations

    def __call__(self, x):
        return sys.maxsize

    @abstractmethod
    def run(self, x):
        pass

    @np.vectorize
    def evaluate(self, x, y):
        return self(np.array([x, y]))


class TSP(ObjectiveFunction):
    def __init__(self, dataset):
        path = os.path.join('tsp', '{}.xml'.format(dataset))
        super(TSP, self).__init__('TSP', path)

    def run(self, x):
        pass

