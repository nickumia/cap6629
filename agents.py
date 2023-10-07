
import numpy as np
import math
from cap6635.agents.blindsearch.vacuum import Vacuum

from cap6635.utilities.constants import (
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
)


class MazeRunner(Vacuum):
    '''
    Base MazeRunner Class
    - See parent class for more details
    - Every step taken is (-1)
    '''

    def __init__(self, environ, policy, state=0,
                 start=(1, 1), elements_in_row=10):
        super(MazeRunner, self).__init__(environ, start)
        self._policy = policy
        self._state = state
        self._elements_in_row = elements_in_row

    @property
    def state(self):
        return self._state

    def move(self):
        action = np.where(self._policy[self._state] == 1)[0][0]
        if action == MOVE_UP:
            self._state -= self._elements_in_row
        elif action == MOVE_DOWN:
            self._state += self._elements_in_row
        elif action == MOVE_LEFT:
            self._state -= 1
        elif action == MOVE_RIGHT:
            self._state += 1
        self._y = (self._state % self._elements_in_row) + 1
        self._x = (math.floor(self._state / self._elements_in_row)) + 1
        self.utility = -1
        self.add_to_path((self._x, self._y))
        self.time = 1
