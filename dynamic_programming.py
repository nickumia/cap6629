
import numpy as np
# import any necessary package

# depending on the size of your Grid
# define the size of states (|column| * |row|)
no_states=
no_actions=4

# Transition probability is |S| x |S'| x |A| array
# T[i][j][k]= prob. moving from state i to j when doing action k
#
# moving out of boundary or to block stays in current state
#

'''
(A)
'''
#
# B is an array that stores block state numbers 
# B=
#

# 
# 1-1) deterministic transition
#    agent moves to its intented next state with prob=1.0

#    complete T matrix for degtterministic transition
#T = 

# 1-2) probabiistic transition
#    
#    complete T matrix for probabilistic transition
#T = 

# 1-3) Reward function: |S| x |A| array
# R[i][j]= reward from state i and action j
# each move generates -1 reward
#R = 

# Discount factor: scalar in [0,1)
#gamma = 0.9        

'''
(B)
'''
#
#Policy: |S| x |A| array
#P[i][j]= prob of choosing action j in state i 
#
# 2-1) initialize policy P with uniform policy
P=


# 2-2) implement prediction (policy evaluation)
# compute V values from a policy
# implement prediction(policy evaluation) algorithm in slide page 7.
def policy_eval(policy, max_iter):
    '''
    Input:
    policy: input Policy array
    max_iter: maximum iteration (use large number for policy iteration)

    Ouput:
    V -- Value function: array of |S| entries
    '''

    # V value begins with 0
    V = np.zeros(no_states)

    #
    # complete this part
    #

    return V

#
# 2-3) implement policy improvement with V value using greedy method
# The formula for choosing the best action using V value is given in question.
# 
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
    P = np.zeros(no_states)

    return P

#
# 2-4) implement policy iteration method
# implement policy iteration in slide page 13
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
    P = 
    V = 
    no_iter = 0

    #
    # complete this part
    # you can use 'policy_eval' and 'extract_policy' function
    #


    # returns policy, state value, and # of iteration
    return [P, V, no_iter]

#
# 2-5) implement value iteration method
# implement value iteration in slide page 23
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
    V = np.zeros(no_states)
    no_iter = 0
        
    return [V, no_iter]

'''

'''

# show the results of prediction (policy evaluation) for random(uniform) policy

# 

# extract policy 
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))

policy = mdp.extractPolicy(V)

V = mdp.evaluatePolicy()

[policy,V,iterId] = mdp.policyIteration()

[V,iterId,epsilon] = mdp.evaluatePolicyPartially()

[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration()

