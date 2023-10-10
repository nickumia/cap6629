
import math
import matplotlib.pyplot as plt
import numpy as np
import os

from agents import MazeRunner
import transition_probability
from cap6635.environment.map import Map2D
from cap6635.utilities.plot import MazeAnimator
from cap6635.utilities.constants import (
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
)

# import matplotlib
# matplotlib.use('TkAgg')


# States: size of your Grid (|column| * |row|)
elements_in_row = 10
no_states = elements_in_row ** 2
# Actions: up(0)|down(1)|left(2)|right(3)
no_actions = 4
get_action = {MOVE_UP: '^', MOVE_DOWN: 'v', MOVE_LEFT: '<', MOVE_RIGHT: '>'}
# Discount factor: scalar in [0,1)
gamma = 0.9
# Goal Reward
goal_reward = 0
state_reward = -1
# GUI on?
gui = True

'''''''''''''''''''''''''''''''''''''''''
Section 0 (Part 2)

Map Legend | (s): start | (b): block | (e): end

    0   1   2   3   4   5   6   7   8   9
0   s           b       s           b
1   b       b   b   b           b
2
3       b               b
4   b                                   b
5                       b
6           b                           e
7
8                   b               b
9           e
'''''''''''''''''''''''''''''''''''''''''
# States that have obstacles
B = [3, 8, 10, 12, 13, 14, 17, 31, 35, 40, 49,
     55, 62, 84, 88]
# Final States
E = [69, 92]

# 1-3) Reward Function: |S| x |A| array
#       R[i][j]= reward from state i and action j
#       each move generates -1 reward
R = np.ones((no_states, no_actions)) * state_reward * 10
for e in E:
    R[e][:] = state_reward


# Deterministic Transition
T = transition_probability.generate(no_states, no_actions, state_reward,
                                    elements_in_row, B=B, R=R,
                                    deterministic=False)


'''''''''''''''''''''''''''''''''''''''''
Section B (Part 3-2)
'''''''''''''''''''''''''''''''''''''''''
# Policy: |S| x |A| array
# P[i][j]= prob of choosing action j in state i

# 2-1) Random(uniform) Policy
#       initialize policy P with uniform policy
P = np.ones((no_states, no_actions)) * (1 / no_actions)


# 2-2) Implement Prediction (Policy Evaluation)
#       compute V values from a policy
#       implement prediction(policy evaluation) algorithm in slide page 7.
def policy_eval(policy, max_iter):
    '''
    Input:
    policy: input Policy array
    max_iter: maximum iteration (use large number for policy iteration)

    Ouput:
    V -- Value function: array of |S| entries
    '''
    # V value begins with 0
    V_0 = np.zeros(no_states)
    convergance = np.zeros(no_states)
    no_iter = 0

    diff = np.ones(no_states)
    while any(diff != convergance) and no_iter < max_iter:
        V_1 = np.zeros(no_states)
        for a in range(no_actions):
            for s in range(no_states):
                v_all = 0
                for sp in range(no_states):
                    # This should only return valid moves
                    # if T[s][sp][a] == 1:
                    #     print(s, sp, get_action[a], T[s][sp][a])
                    if s not in E:
                        v_all += T[s][sp][a] * V_0[sp]
                q = R[s][a] + gamma * v_all
                V_1[s] += policy[s][a] * q
        diff = V_1 - V_0
        V_0 = np.copy(V_1)
        no_iter += 1

    return (V_0, no_iter)


# 2-3) Implement Policy Improvement with V value using Greedy Method
#       The formula for choosing the best action using V value is given
#       in question.
def extract_policy(V):
    '''
    Procedure to extract a policy from a value function
    pi <-- argmax_a R^a + gamma T^a V

    Inputs:
    V -- Value function: array of |S| entries

    Output:
    policy -- Policy array P
    '''
    # initialize random(uniform) policy
    # P = np.ones((no_states, no_actions)) * (1 / no_actions)
    P = np.zeros((no_states, no_actions))

    for s in range(no_states):
        best_action = 0
        # TODO: Choose a better absolute worst q
        max_q = -100000000
        for i, a in enumerate(range(no_actions)):
            v = 0
            for sp in range(no_states):
                v += T[s][sp][a] * V[sp]
            q = R[s][a] + gamma * v
            if q > max_q:
                best_action = i
                max_q = q
        P[s][best_action] = 1

    return P


# 2-4) Implement Policy Iteration Method
#       implement policy iteration in slide page 13
def policy_iter(in_policy, max_iter):

    '''    Policy iteration procedure: alternate between
    1) policy evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and
    2) policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

    Inputs:
    in_policy -- Initial policy
    max_iter -- maximum # of iterations: scalar (use a large number)

    Outputs:
    policy -- Policy P
    V -- Value function: array of |S| entries
    no_iter -- the actual # of iterations peformed by policy iteration: scalar
    '''
    P = in_policy
    V_0 = np.zeros(no_states)
    convergance = np.zeros(no_states)
    diff = np.ones(no_states)
    no_iter = 0

    while any(diff != convergance) and no_iter < max_iter:
        V_1, no_iter_2 = policy_eval(P, max_iter)
        P = extract_policy(V_1)
        diff = abs(V_1 - V_0)
        V_0 = np.copy(V_1)
        no_iter += no_iter_2

    # returns policy, state value, and # of iteration
    return (P, V_0, no_iter)


# 2-5) Implement Value Iteration Method
#       implement value iteration in slide page 23
def value_iter(in_V, max_iter):
    '''
    Value iteration procedure
    V <-- max_a R^a + gamma T^a V

    Inputs:
    in_V -- Initial value function: array of |S| entries
    max_iter -- limit on the # of iterations: scalar (use large number)

    Outputs:
    V -- Value function: array of |S| entries
    no_iter -- the actual # of iterations peformed by policy iteration: scalar
    '''
    V_0 = in_V
    convergance = np.zeros(no_states)
    no_iter = 0

    diff = np.ones(no_states)
    V_1 = np.zeros(no_states)
    while any(diff != convergance) and no_iter < max_iter:
        for s in range(no_states):
            all_q = []
            for a in range(no_actions):
                v_all = 0
                for sp in range(no_states):
                    v_all += T[s][sp][a] * V_0[sp]
                q = R[s][a] + gamma * v_all
                all_q.append(q)
            V_1[s] = max(all_q)
        diff = V_1 - V_0
        V_0 = np.copy(V_1)
        no_iter += 1

    return [V_1, no_iter]


'''''''''''''''''''''''''''''''''''''''''
Section C (Part 4)
'''''''''''''''''''''''''''''''''''''''''

# 4.1.1a Random(uniform) Policy defined above
# 4.1.1b Show the results of policy_eval
# V, no_iter = policy_eval(P, 1000)
# print(V)
# print("Number of Iterations: %d" % (no_iter))

# 4.1.2 Run Policy Iteration and show the results
# (P, V, no_iter) = policy_iter(P, 500)
# print(V)
# print("Number of Iterations: %d" % (no_iter))

# 4.1.3a Run Value Iteration and show the results
V = np.ones(no_states) * -1000
(V, no_iter) = value_iter(V, 500)
print(V)
print("Number of Iterations: %d" % (no_iter))

# 4.1.3b Extract policy from V values
P = extract_policy(V)
print(P)


'''''''''''''''''''''''''''''''''''''''''
Section D (Extra)
'''''''''''''''''''''''''''''''''''''''''
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
plt.savefig('policy.png')
plt.clf()


if gui:
    maze = Map2D(elements_in_row + 2, elements_in_row + 2)
    for e in E:
        maze.map[
            (math.floor(e / elements_in_row)) + 1][
            (e % elements_in_row) + 1
            ] = 20
    for b in B:
        maze.map[
            (math.floor(b / elements_in_row)) + 1][
            (b % elements_in_row) + 1
            ] = 5

    a = MazeRunner(maze, P, state=40, start=(4, 1),
                   elements_in_row=elements_in_row)
    animator = MazeAnimator(os.getcwd(), '/maze.gif')
    animator.temp = '/temp/'
    animator.save_state(i, maze, a)
    i = 0
    while a.state not in E:
        # Starting position
        a.move()
        maze.map[a._x_path[0], a._y_path[0]] = 30
        maze.map[a._x_path[-1], a._y_path[-1]] = 10
        maze.map[a._x_path[-2], a._y_path[-2]] = 0
        animator.save_state(i, maze, a)
        i += 1

    animator.make_gif()
    del animator.temp
