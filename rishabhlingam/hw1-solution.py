import operator
import numpy as np
from prettytable import PrettyTable
# import any necessary package

# depending on the size of your Grid
# define the size of states (|column| * |row|)

rows, cols = 6, 6
no_states = rows * cols
no_actions = 4
LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3
START = [7, 35]
STOP = [15, 18]
# B is an array that stores block state numbers 
Blocks = [8, 14, 20, 31, 32, 33]

# Transition probability is |S| x |S'| x |A| array
# T[i][j][k]= prob. moving from state i to j when doing action k
# moving out of boundary or to block stays in current state

# utility functions
def get_row_col_index_0(position, t_rows, t_cols):
    row_idx = position // t_cols
    col_idx = (position % t_cols)
    return row_idx, col_idx

def get_row_col_index(position):
    return get_row_col_index_0(position, rows, cols)

def get_position_0(row_idx, col_idx, t_rows, t_cols):
    if row_idx < 0 :
        row_idx = 0
    elif row_idx == t_rows:
        row_idx = t_rows - 1
    if col_idx < 0:
        col_idx = 0
    elif col_idx == t_cols:
        col_idx = t_cols - 1
    return (row_idx * t_cols) + (col_idx)

def get_position(row_idx, col_idx):
    return get_position_0(row_idx, col_idx, rows, cols)

def can_go_left(position):
    row_idx, col_idx = get_row_col_index(position)
    lef_pos = get_position(row_idx, col_idx - 1)
    return (col_idx != 0) and (lef_pos not in Blocks)

def can_go_right(position):
    row_idx, col_idx = get_row_col_index(position)
    right_pos = get_position(row_idx, col_idx + 1)
    return (col_idx != cols - 1) and (right_pos not in Blocks)

def can_go_up(position):
    row_idx, col_idx = get_row_col_index(position)
    up_pos = get_position(row_idx - 1, col_idx)
    return (row_idx != 0) and (up_pos not in Blocks)

def can_go_down(position):
    row_idx, col_idx = get_row_col_index(position)
    down_pos = get_position(row_idx + 1, col_idx)
    return (row_idx != rows - 1) and (down_pos not in Blocks)

def get_possible_action_state(position):
    row_idx, col_idx = get_row_col_index(position)
    actions, states = [], []
    
    if can_go_left(position):
        actions.append(LEFT)
        states.append(get_position(row_idx, col_idx-1))
    if can_go_up(position):
        actions.append(UP)
        states.append(get_position(row_idx-1, col_idx))
    if can_go_right(position):
        actions.append(RIGHT)
        states.append(get_position(row_idx, col_idx + 1))
    if can_go_down(position):
        actions.append(DOWN)
        states.append(get_position(row_idx+1, col_idx))
        
    return actions, states

def get_string_policy(policy):
    n = len(policy)
    string_policy = np.chararray(n, itemsize=10, unicode=True)
    for s in range(n):
        max_idx = np.argmax(policy[s])
        if max_idx == 0:
            string_policy[s] = 'LEFT'
        elif max_idx == 1:
            string_policy[s] = 'UP'
        elif max_idx == 2:
            string_policy[s] = 'RIGHT'
        elif max_idx == 3:
            string_policy[s] = 'DOWN'
    string_policy[START] = 'START'
    string_policy[STOP] = 'STOP'
    string_policy[Blocks] = 'BLOCK'
    return string_policy

def get_as_table(policy):
    pTable = PrettyTable()
    pTable.clear()
    pol_temp = policy.reshape((rows, cols))
    n = len(pol_temp)
    for i in range(n):
        pTable.add_row(pol_temp[i], divider=True)
    pTable.header = False
    return pTable

'''
(A)
'''

# 1-1) deterministic transition
#      agent moves to its intented next state with prob=1.0

#      complete T matrix for deterministic transition
T = np.zeros((no_states, no_states, no_actions))

for position in range(0, no_states):
        row_idx, col_idx = get_row_col_index(position)
        
        left_pos = get_position(row_idx, col_idx - 1)
        right_pos = get_position(row_idx, col_idx + 1)
        up_pos = get_position(row_idx - 1, col_idx)
        down_pos = get_position(row_idx + 1, col_idx)
        
        if can_go_left(position):
            T[position, left_pos, LEFT] = 1
        else:
            T[position, position, LEFT] = 1
        
        if can_go_up(position):
            T[position, up_pos, UP] = 1
        else:
            T[position, position, UP] = 1
        
        if can_go_right(position):
            T[position, right_pos, RIGHT] = 1
        else:
            T[position, position, RIGHT] = 1
        
        if can_go_down(position):
            T[position, down_pos, DOWN] = 1
        else:
            T[position, position, DOWN] = 1

# 1-2) probabiistic transition
#      complete T matrix for probabilistic transition

T_prob = np.zeros((no_states, no_states, no_actions))

for position in range(0, no_states):
        row_idx, col_idx = get_row_col_index(position)
        
        left_pos = get_position(row_idx, col_idx - 1)
        right_pos = get_position(row_idx, col_idx + 1)
        up_pos = get_position(row_idx - 1, col_idx)
        down_pos = get_position(row_idx + 1, col_idx)
        
        if can_go_left(position):
            T_prob[position, left_pos, LEFT] = np.round(np.random.rand(), 2)
        else:
            T_prob[position, position, LEFT] = 1
        
        if can_go_up(position):
            T_prob[position, up_pos, UP] = np.round(np.random.rand(), 2)
        else:
            T_prob[position, position, UP] = 1
        
        if can_go_right(position):
            T_prob[position, right_pos, RIGHT] = np.round(np.random.rand(), 2)
        else:
            T_prob[position, position, RIGHT] = 1
        
        if can_go_down(position):
            T_prob[position, down_pos, DOWN] = np.round(np.random.rand(), 2)
        else:
            T_prob[position, position, DOWN] = 1

# 1-3) Reward function: |S| x |A| array
#      R[i][j]= reward from state i and action j
#      each move generates -1 reward
R = np.full((no_states, no_actions), -1)

# Discount factor: scalar in [0,1)
gamma = 0.9        

'''
(B)
'''
#Policy: |S| x |A| array
#P[i][j]= prob of choosing action j in state i
# 2-1) initialize policy P with uniform policy
P = np.full((no_states, no_actions), 0.25)


# 2-2) implement prediction (policy evaluation)
#      compute V values from a policy
#      implement prediction(policy evaluation) algorithm in slide page 7.
def policy_eval(policy, max_iter, V = np.full(no_states, 0)):
    '''
    Input:
    policy: input Policy array
    max_iter: maximum iteration (use large number for policy iteration)

    Ouput:
    V -- Value function: array of |S| entries
    '''
    # V value begins with 0
    V = np.zeros(no_states)

    for i in range(max_iter):
        V_temp = np.zeros(no_states)
        for s in range(no_states):
            v_s = 0
            for action in range(no_actions):
                q_sa = 0
                for s_prime in range(no_states):
                    q_sa += T[s, s_prime, action] * V[s_prime]
                q_sa *= gamma
                q_sa += R[s, action]
                q_sa *= policy[s, action]
                v_s += q_sa
            V_temp[s] = np.round(v_s, 2)
            V_temp[STOP] = 0
        V = np.copy(V_temp)
    return V

# 2-3) implement policy improvement with V value using greedy method
#      The formula for choosing the best action using V value is given in question.

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
    new_policy = np.zeros((no_states, no_actions))
    
    for s in range(no_states):
        actions, states = get_possible_action_state(s)
        max_idx = np.argmax(V[states])
        max_state = states[max_idx]
        max_act = actions[max_idx]
        new_policy[s, max_act] = 1

    return new_policy

# 2-4) implement policy iteration method
#      implement policy iteration in slide page 13
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
    P = in_policy
    V = np.zeros(no_states)
    no_iter = 0

    # complete this part
    # you can use 'policy_eval' and 'extract_policy' function
    for i in range(max_iter):
        no_iter += 1
        V = policy_eval(P, 10)
        P_temp = extract_policy(V)
        if np.sum(P == P_temp) == (no_states * no_actions):
            break
        P = np.copy(P_temp)


    # returns policy, state value, and # of iteration
    return [P, V, no_iter]

#
# 2-5) implement value iteration method
#      implement value iteration in slide page 23
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
    V = in_V
    no_iter = 0
    
    # generating random policy
    rand_policy = np.full((no_states, no_actions), 0.25)
    for i in range(max_iter):
        no_iter += 1
        V_temp = policy_eval(rand_policy, 1, V)
        if np.sum(V == V_temp) == no_states:
            break
        V = np.copy(V_temp)
        
    return [V, no_iter]

'''

'''

# show the results of prediction (policy evaluation) for random(uniform) policy 
# extract policy 
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))

policy = mdp.extractPolicy(V)

V = mdp.evaluatePolicy()

[policy,V,iterId] = mdp.policyIteration()

[V,iterId,epsilon] = mdp.evaluatePolicyPartially()

[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration()
