
from cap6635.environment.map import Map2D
from cap6635.utilities.constants import (
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
)
import math
import matplotlib.pyplot as plt
import numpy as np

get_action = {MOVE_UP: '^', MOVE_DOWN: 'v', MOVE_LEFT: '<', MOVE_RIGHT: '>'}


def pretty_policy(P, elements_in_row, E, B, text):
    policy = Map2D(elements_in_row + 2, elements_in_row + 2)
    for i, s in enumerate(P):
        action = np.where(s == 1)[0][0]
        y = (i % elements_in_row) + 1
        x = (math.floor(i / elements_in_row)) + 1
        if i in E:
            plt.text(y, x, "x", color='green')
        elif i in B:
            plt.text(y, x, "o", color='green')
        else:
            plt.text(y, x, get_action[action], color='green')
        plt.imshow(policy.map, 'pink')
    plt.savefig('policy_%s.png' % (text))
    plt.clf()
