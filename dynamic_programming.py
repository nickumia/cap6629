
import numpy as np
from cap6635.environment.map import Map2D
import math
import matplotlib.pyplot as plt
import time

import matplotlib
matplotlib.use('TkAgg')

# States: size of your Grid (|column| * |row|)
no_states = 100
elements_in_row = 10
# Actions: up(0)|down(1)|left(2)|right(3)
no_actions = 4
# Probabiistic Transition:
alpha = 0.05
# Discount factor: scalar in [0,1)
gamma = 0.9

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

'''''''''''''''''''''''''''''''''''''''''
Section A (Part 3-1)
'''''''''''''''''''''''''''''''''''''''''
# Transition probability is |S| x |S'| x |A| array
# T[i][j][k]= prob. moving from state i to j when doing action k
# moving out of boundary or to block stays in current state

# 1-1) Deterministic Transition
#       agent moves to its intented next state with prob=1.0
#       complete T matrix for deterministic transition
T = np.zeros((no_states, no_states, no_actions))
for i in range(no_states):
    if i % elements_in_row != 0:
        # Leftward movements
        # print("LEFT", i, i-1)
        T[i][i-1][2] = 1
    if (i % elements_in_row != (elements_in_row-1) and i != (no_states-1)):
        # Rightward movements
        # print('RIGHT', i, i+1)
        T[i][i+1][3] = 1
    if i < (no_states - elements_in_row):
        # Downward movement
        # print("DOWN", i, i+10)
        T[i][i+10][1] = 1
    if i > elements_in_row:
        # Upwrd movement
        T[i][i-10][0] = 1

for b in B:
    if b - elements_in_row >= 0 and (b - elements_in_row) not in B:
        # Space above block
        # print(b-10, 'above', b)
        T[b - elements_in_row][b][1] = 0
    if b + elements_in_row < no_states and (b + elements_in_row) not in B:
        # Space below block
        # print(b+10, 'below', b)
        T[b + elements_in_row][b][0] = 0
    if b + 1 < no_states and (b + 1) not in B:
        # Space to the right of block
        T[b + 1][b][2] = 0
    if b - 1 > 0 and (b - 1) not in B:
        # Space to the left of block
        T[b - 1][b][3] = 0

# 1-2) Probabiistic Transition
#       complete T matrix for probabilistic transition
# T = T * (1 - no_actions * alpha)

# 1-3) Reward Function: |S| x |A| array
#       R[i][j]= reward from state i and action j
#       each move generates -1 reward
R = np.ones((no_states, no_actions)) * -1
E = [69, 92]
# E = [30]
for e in E:
    if e - elements_in_row > 0:
        # Space above block
        R[e - elements_in_row][1] = 10
    if e + elements_in_row < no_states:
        # Space below block
        R[e + elements_in_row][0] = 10
    if e + 1 < no_states:
        # Space to the right of block
        R[e + 1][3] = 10
    if e - 1 > 0:
        # Space to the left of block
        R[e - 1][2] = 10


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

    diff = False
    while not diff and no_iter < max_iter:
        V_1 = np.zeros(no_states)
        for a in range(no_actions):
            for s in range(no_states):
                v_all = 0
                for sp in range(no_states):
                    v_all += T[s][sp][a] * V_0[sp]
                q = R[s][a] + gamma * v_all
                V_1[s] += P[s][a] * q
        diff = all(V_1 - V_0 < convergance)
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
    P = np.zeros((no_states, no_actions))

    for s in range(no_states):
        best_action = 0
        max_q = 0
        for i, a in enumerate(range(no_actions)):
            v = 0
            for sp in range(no_states):
                v += T[s][sp][a] * V[sp]
            q = abs(R[s][a] + gamma * v)
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

    # Initialization P and V using np.zeros
    # P = np.zeros((no_states, no_actions))
    P = in_policy
    V_0 = np.zeros(no_states)
    no_iter = 0
    convergance = np.zeros(no_states)
    diff = False

    while not diff and no_iter < max_iter:
        V_1, no_iter_2 = policy_eval(P, max_iter)
        P = extract_policy(V)
        diff = all(V_1 - V_0 < convergance)
        V_0 = V_1
        no_iter += no_iter_2

    # returns policy, state value, and # of iteration
    return [P, V, no_iter]


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

    # Initialization V using np.zeros
    # V = np.zeros(no_states)
    V_0 = in_V
    no_iter = 0
    changed = 1
    while changed != 0 and no_iter < max_iter:
        V_1 = np.zeros(no_states)
        changed = 0
        for s in range(no_states):
            max_q = V_0[s]
            for a in range(no_actions):
                v_all = 0
                for sp in range(no_states):
                    v_all += T[s][sp][a] * V_0[sp]
                q = R[s][a] + gamma * v_all
                V_1[s] += P[s][a] * q
            if q > max_q:
                max_q = q
                changed += 1
            V_0[s] = q
        no_iter += 1

    return [V, no_iter]


'''''''''''''''''''''''''''''''''''''''''
Section C (Part 4)
'''''''''''''''''''''''''''''''''''''''''

# 4.1.1a Random(uniform) Policy defined above
# 4.1.1b Show the results of policy_eva
V, no_iter = policy_eval(P, 100)
# print(V)
print("Number of Iterations: %d" % (no_iter))

# 4.1.2 Run Policy Iteration and show the results
(P, V, no_iter) = policy_iter(P, 100)
# print(P)
# print(V)
print("Number of Iterations: %d" % (no_iter))

# 4.1.3a Run Value Iteration and show the results
(V, no_iter) = value_iter(V, 1000)
print(V)
print("Number of Iterations: %d" % (no_iter))

# 4.1.3b Extract policy from V values
P = extract_policy(V)
print('b', no_iter)


'''''''''''''''''''''''''''''''''''''''''
Section D (Extra)
'''''''''''''''''''''''''''''''''''''''''
# maze = Map2D(11, 11)
# 
# agent_loc = 0
# x = 0
# y = 0
# for i in range(10):
#     # Actions: up(0)|down(1)|left(2)|right(3)
#     maze.map[x, y] = 10
#     action = np.where(P[agent_loc] == 1)[0][0]
#     if action == 0:
#         agent_loc -= 10
#     elif action == 1:
#         agent_loc += 10
#     elif action == 2:
#         agent_loc -= 1
#     elif action == 3:
#         agent_loc += 1
#     y = (agent_loc % 10) + 1
#     x = (math.floor(agent_loc / 10)) + 1
#     time.sleep(0.5)
#     # label = "Time Elapsed:%d; Utility: %.1f" % (agent.time, agent.utility)
#     # plt.text(0, 0, label)
#     plt.imshow(maze.map, 'pink')
#     plt.show()
#     # plt.plot(agent.y_path, agent.x_path, 'r:', linewidth=1)
#     # plt.plot(agent.y_path[-1], agent.x_path[-1], '*r', 'Robot field', 5)
