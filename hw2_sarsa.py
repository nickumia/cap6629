
from cap6635.utilities.constants import (
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
)
import numpy as np
import random

from hw1_utils import pretty_policy
import transition_probability as tp

state_reward = -1
elements_in_row = 10


class GridEnv:
    '''
    # Question 1

    # define your transition prob array and reward
    # (you can use the same trans prob and reward from HW 1)
    # For simplisity, you can use deterministic transition only.
    # now trans prob and reward are not visible to agent
    # TODO: Add probabilistic transition
    #
    '''

    def __init__(self, no_states, no_actions, start=0):
        self._start = start
        self.current_state = start
        # States that have obstacles
        self.B = [3, 8, 10, 12, 13, 14, 17, 31, 35, 40, 49, 55, 62, 84, 88]
        # Final States
        self.E = [69, 92]

        # 1-3) Reward Function: |S| x |A| array
        self.R = np.ones((no_states, no_actions)) * state_reward * 10

        # Deterministic Transition
        self.T = tp.generate(no_states, no_actions, state_reward,
                             elements_in_row, B=self.B, R=self.R,
                             deterministic=True)
        for e in self.E:
            self.R[e][:] = 0

    #
    # step function: reads action and returns next-state, reward, done, info
    # Input
    #   action: action value of agent
    #
    # Output
    #   state: next state
    #   reward: reward value
    #   done: whether the agent terminated (e.g., reached final state or
    #   maximum itertation, etc)
    #   info: other information
    def step(self, action):
        possible_states = []
        for i in range(100):
            if self.T[self.current_state][i][action] >= 1:
                possible_states.append(i)
        try:
            new_state = random.choice(possible_states)
            self.current_state = new_state
        except IndexError:
            return self.current_state, 0, False, "Action not possible"
        reward = self.R[self.current_state][action]
        if new_state in self.E:
            done = True
        else:
            done = False
        return new_state, reward, done, ""

    # agent returns to start state
    # begins another episode
    def reset(self):
        self.current_state = self._start
        return self.current_state

    # not implemented yet
    # def render(self):


# create environment
no_states = 100
no_actions = 4
env = GridEnv(no_states, no_actions)
# print(mdp.step(MOVE_RIGHT))
# print(mdp.step(MOVE_DOWN))


'''
# Question 2 - Q learning
'''
# initialize alpha value
alpha = 0.02

# define parameters: epsilon, gamma (discount factor)
epsilon = 0.2
gamma = 0.9

max_iter = 200
no_episodes = 1000

min_exploration_proba = 0.05
exploration_decreasing_decay = 0.9
rewards_per_episode = []

Q = np.zeros((no_states, no_actions))

for e in range(no_episodes):
    current_state = env.reset()
    done = False
    total_episode_reward = 0

    for i in range(max_iter):
        # Epsilon-Greedy Behavior Policy
        choice = random.random()
        if choice < epsilon:
            action = random.choice([MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT])
        else:
            action = np.argmax(Q[current_state])

        # The environment mdp runs the chosen action and returns
        # the next state, a reward and true if the epiosed is ended.
        next_state, reward, done, _ = env.step(action)

        # PART (A)
        # We update our Q-table using the Q-learning iteration
        Q[current_state, action] += \
            alpha * (reward + gamma * max(Q[next_state, :])
                     - Q[current_state, action])
        # END PART (A)

        total_episode_reward += reward
        # If the episode is finished, we leave the for loop
        if done:
            break
        current_state = next_state

    # decrease epsilon value
    exploration_proba = max(min_exploration_proba,
                            np.exp(-exploration_decreasing_decay*e))
    rewards_per_episode.append(total_episode_reward)

#########################
# show Q values
print(Q)


#########################
# show the final policy
# in each state, choose action that maximizes Q value
def extract_policy(Q, no_states, no_actions):
    '''
    Extract Policy from Q-learning matrix
    Since Q is already calculated, just retrieve the max.
    '''
    P = np.zeros((no_states, no_actions))

    for s in range(no_states):
        best_action = np.argmax(Q[s])
        P[s][best_action] = 1

    return P


# for s in range(no_states):
#     for a in range(no_actions):
#         if Q[s][a] == 0:
#             Q[s][a] -= 10000
P = extract_policy(Q, no_states, no_actions)
pretty_policy(P, elements_in_row, env.E, env.B, 'hw2')

#
# #
# # Question 3
# #
# # Sarsa
# #
# # modify PART (A) in Q learning as follows
# # 1) next_action = action chosen in 'next_state' using epsilon greedy.
# # 2) max(Q[next_state,:] is replaced by Q[next_state, next_action]
# # the rest of the code is same.
# #
#
# #
# # show Q values
# #
#
# #
# # show the final policy
# # in each state, choose action that maximizes Q value
# #
