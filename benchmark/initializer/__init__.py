from abc import ABC
from abc import abstractmethod


class SearchSpaceInitializer(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class SearchSpaceInitializerFactory(ABC):
    @staticmethod
    @abstractmethod
    def uniform_random(*args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def given_point(*args, **kwargs):
        pass
