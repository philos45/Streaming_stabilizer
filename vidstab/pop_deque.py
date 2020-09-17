from collections import deque

class PopDeque(deque):
    def deque_full(self):
        return len(self) == self.maxlen

    def pop_append(self, x):
        popped_element = None
        if self.deque_full():
            popped_element = self.popleft()
        self.append(x)
        return popped_element

    def increment_append(self, increment=1, pop_append=True):
        if len(self) == 0:
            popped_element = self.pop_append(0)
        else:
            popped_element = self.pop_append(self[-1] + increment)
        if not pop_append:
            return None
        return popped_element
