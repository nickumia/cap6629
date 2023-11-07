
from cap6635.utilities.constants import (
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
)
import numpy as np
import random

from hw1_utils import pretty_policy, gather_inputs, hw2_usage, animate
import transition_probability as tp

state_reward = -1
elements_in_row, algo, _, gui, start_pos = gather_inputs(hw2_usage)
no_states = elements_in_row ** 2


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
        # Save starting state to track each episode
        self._start = start
        self._current_state = start

        # States that have obstacles
        self.B = [3, 8, 10, 12, 13, 14, 17, 31, 35, 40, 49, 55, 62, 84, 88]
        # Final States
        self.E = [69, 92]

        # Reward Function (same as hw1)
        self.R = np.ones((no_states, no_actions)) * state_reward * 10

        # Transition Probability (deterministic-only)
        self.T = tp.generate(no_states, no_actions, state_reward,
                             elements_in_row, B=self.B, R=self.R,
                             deterministic=True)
        for e in self.E:
            self.R[e][:] = 0

    def step(self, action):
        '''
        Reads action and returns next-state, reward, done, info
        '''
        done = False
        msg = ""
        # States that can be achieved with given action
        possible_states = []
        for i in range(100):
            if self.T[self._current_state][i][action] >= 1:
                possible_states.append(i)
        try:
            # Reward for new state
            self._current_state = random.choice(possible_states)
            reward = self.R[self._current_state][action]

            # Is the new state a goal?
            if self._current_state in self.E:
                done = True
                msg = "Goal!"
            else:
                msg = "Still searching..."
        except IndexError:
            msg = "Action not possible"

        reward = self.R[self._current_state][action]
        return self._current_state, reward, done, msg

    def reset(self):
        ''' Begin another episode at starting state '''
        self._current_state = self._start
        return self._current_state

    # not implemented yet
    # def render(self):


class QLearning:
    ''' Question 2: Q-Learning '''
    def __init__(self, no_states, no_actions, max_iter=100, no_episodes=1000):
        # initialize alpha value
        self._alpha = 0.2

        # define parameters: epsilon, gamma (discount factor)
        self._epsilon = 0.2
        self._gamma = 0.9

        self._max_iter = max_iter
        self._no_episodes = no_episodes

        self._exploration_proba = self._epsilon
        self._min_exploration_proba = 0.05
        self._exploration_decay = 0.9
        self._rewards_per_episode = []

        self.reset()

    def q_learning(self, current_state, next_state, action, reward):
        self._Q[current_state, action] += \
            self._alpha * (
                reward + self._gamma * max(self._Q[next_state, :])
                - self._Q[current_state, action])

    def sarsa(self, current_state, next_state, action, reward):
        self._Q[current_state, action] += \
            self._alpha * (
                reward + self._gamma * self._Q[next_state, action]
                - self._Q[current_state, action])

    def learn(self, env, sarsa=False):
        for e in range(self._no_episodes):
            current_state = env.reset()
            done = False
            total_episode_reward = 0

            for i in range(self._max_iter):
                # Epsilon-Greedy Behavior Policy
                choice = random.random()
                if choice < self._exploration_proba:
                    action = random.choice([MOVE_UP, MOVE_DOWN,
                                            MOVE_LEFT, MOVE_RIGHT])
                else:
                    action = np.argmax(self._Q[current_state])

                # The environment env runs the chosen action and returns
                # the next state, a reward and true if the epiosed is ended.
                next_state, reward, done, _ = env.step(action)

                # We update our Q-table using the Q-learning iteration
                if sarsa:
                    self.sarsa(current_state, next_state, action, reward)
                else:
                    self.q_learning(current_state, next_state, action, reward)

                total_episode_reward += reward
                # If the episode is finished, we leave the for loop
                if done:
                    break
                current_state = next_state

            # decrease epsilon value
            self._exploration_proba = \
                max(self._min_exploration_proba,
                    np.exp(-self._exploration_decay*self._epsilon))
            self._rewards_per_episode.append(total_episode_reward)
        return self._Q

    def reset(self):
        self._Q = np.zeros((no_states, no_actions))


#########################
# Main Algorithm
no_states = 100
no_actions = 4

env = GridEnv(no_states, no_actions)
ql = QLearning(no_states, no_actions)

Q = ql.learn(env)
print(Q)

P = extract_policy(Q, no_states, no_actions)
pretty_policy(P, elements_in_row, env.E, env.B,
              str(elements_in_row)+algo+str(start_pos))
print('Animating...')


maze_name = '/maze_%s_%s_%s.gif' % (elements_in_row, algo, start_pos)
if gui:
    animate(start_pos, env.E, env.B, elements_in_row, algo, P, maze_name)
