
import math
import numpy as np

from cap6635.utilities.constants import (
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
)

# Probabiistic Transition:
alpha = 0.0225


'''''''''''''''''''''''''''''''''''''''''
Section A (Part 3-1)
'''''''''''''''''''''''''''''''''''''''''
# Transition probability is |S| x |S'| x |A| array
# T[i][j][k]= prob. moving from state i to j when doing action k
# moving out of boundary or to block stays in current state

# 1-1) Deterministic Transition
#       agent moves to its intented next state with prob=1.0
#       complete T matrix for deterministic transition

# 1-2) Probabiistic Transition
#       complete T matrix for probabilistic transition


def generate(no_states, no_actions, state_reward, elements_in_row, B=[], R=[],
             deterministic=True):
    T = np.zeros((no_states, no_states, no_actions))

    for i in range(no_states):
        if i % elements_in_row != 0:
            # print("LEFT", i, i-1)
            if deterministic:
                T[i][i-1][MOVE_LEFT] = 1
            else:
                T[i][i-1][MOVE_LEFT] = (1 - no_actions * alpha)
                for slippery in [i, i-2, i-1-elements_in_row,
                                 i-1+elements_in_row]:
                    if slippery not in B:
                        if is_valid(i, slippery,
                                    elements_in_row, no_states, relaxed=True):
                            T[i][slippery][MOVE_LEFT] = alpha
            if i - 1 not in B:
                R[i][MOVE_LEFT] = state_reward
        if (i % elements_in_row != (elements_in_row-1) and i != (no_states-1)):
            # print('RIGHT', i, i+1)
            if deterministic:
                T[i][i+1][MOVE_RIGHT] = 1
            else:
                T[i][i+1][MOVE_RIGHT] = (1 - no_actions * alpha)
                for slippery in [i, i+2, i+1-elements_in_row,
                                 i+1+elements_in_row]:
                    if slippery not in B:
                        if is_valid(i, slippery,
                                    elements_in_row, no_states, relaxed=True):
                            T[i][slippery][MOVE_RIGHT] = alpha
            if i + 1 not in B:
                R[i][MOVE_RIGHT] = state_reward
        if i < (no_states - elements_in_row):
            # print("DOWN", i, i+10)
            if deterministic:
                T[i][i + elements_in_row][MOVE_DOWN] = 1
            else:
                T[i][i + elements_in_row][MOVE_DOWN] = (1 - no_actions * alpha)
                for slippery in [i, i+elements_in_row+1,
                                 i+elements_in_row+elements_in_row,
                                 i+elements_in_row-1]:
                    if slippery not in B:
                        if is_valid(i, slippery,
                                    elements_in_row, no_states, relaxed=True):
                            T[i][slippery][MOVE_DOWN] = alpha
            if i + elements_in_row not in B:
                R[i][MOVE_DOWN] = state_reward
        if i > elements_in_row-1:
            if deterministic:
                T[i][i - elements_in_row][MOVE_UP] = 1
            else:
                T[i][i - elements_in_row][MOVE_UP] = (1 - no_actions * alpha)
                for slippery in [i, i-elements_in_row+1,
                                 i-elements_in_row-elements_in_row,
                                 i-elements_in_row-1]:
                    if slippery not in B:
                        if is_valid(i, slippery,
                                    elements_in_row, no_states, relaxed=True):
                            T[i][slippery][MOVE_UP] = alpha
            if i - elements_in_row not in B:
                R[i][MOVE_UP] = state_reward

    for b in B:
        if b - elements_in_row >= 0:
            # Space above block
            # print(b-10, 'above', b)
            T[b - elements_in_row][b][MOVE_DOWN] = 0
            T[b][b - elements_in_row][MOVE_UP] = 0
        if b + elements_in_row < no_states:
            # Space below block
            # print(b+10, 'below', b)
            T[b + elements_in_row][b][MOVE_UP] = 0
            T[b][b + elements_in_row][MOVE_DOWN] = 0
        if b + 1 < no_states:
            # Space to the right of block
            T[b + 1][b][MOVE_LEFT] = 0
            T[b][b + 1][MOVE_RIGHT] = 0
        if b - 1 > 0:
            # Space to the left of block
            T[b - 1][b][MOVE_RIGHT] = 0
            T[b][b - 1][MOVE_LEFT] = 0

    return T


def is_valid(old_state, new_state, elements_in_row, no_states, relaxed=False):
    if new_state > 0 and new_state < no_states:
        # Same Row, Left/Right Possible
        if math.floor(new_state / elements_in_row) == \
           math.floor(old_state / elements_in_row):
            return True
        # Same Column, Up/Down Possible
        if (new_state % elements_in_row) == (old_state % elements_in_row):
            return True
        if relaxed:
            return True
    return False

# print(is_valid(5,7,10, 100))
# print(is_valid(9,11,10, 100))
# print(is_valid(50,49,10, 100))
# print(is_valid(76,66,10, 100))
