# coding: utf-8
from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, train_x, train_y, valid_x, valid_y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def dump(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
