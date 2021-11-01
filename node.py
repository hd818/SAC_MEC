import uuid
from abc import abstractmethod, ABCMeta
import numpy as np


class Node(metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        self.uuid = uuid.uuid4()
        self.node_type = 0

    def print_me(self):
        pass

    def ran(self, Range):
        return np.random.uniform(Range[0], Range[1], 1)

    def ran01(self):
        return np.random.uniform(0, 1, 1)

    def get_uuid(self):
        return self.uuid.hex