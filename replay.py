
from collections import deque

class Relay:
    def __init__(self, buffer_size):
        self.memory = self.create_memory(buffer_size)
        self.full_memory = False
        self.buffer_size = buffer_size
    def create_memory(self, buffer_size):
        return deque([], maxlen=buffer_size)

    def store_exp(self, exp):
        if not self.full_memory and len(self.memory) == self.buffer_size:
            self.full_memory = True
        self.memory.append(exp)

