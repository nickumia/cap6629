
import numpy as np
# import any necessary package

class GridEnv:

    #
    # Question 1
    #
    #
    # define your transition prob array and reward 
    # (you can use the same trans prob and reward from HW 1)
    # For simplisity, you can use deterministic transition only    
    #
    # now trans prob and reward are not visible to agent
    # 
    # trans prob and reward can move to __init__ constructor
    #

    #
    # Constructor
    #
    def __init__(self):
        # 
        # initialize start state
        # return start state
        return your start state

    #
    # step function: reads action and returns next-state, reward, done, info
    # Input    
    #   action: action value of agent
    #
    # Output
    #   state: next state
    #   reward: reward value
    #   done: whether the agent terminated (e.g., reached final state or maximum itertation, etc)
    #   info: other information
    def step(self, action):
        #
        # implement this part
        #
        return state, reward, done, info

    # agent returns to start state
    # begins another episode
    def reset(self):
        #
        # implement this part
        #
        return your start state

    # not implemented yet
    #def render(self):

# create environment
mdp=GridEnv()


#
# Question 2
#
# Q learning
#
# initialize alpha value
alpha=

# define parameters: epsilon, gamma (discount factor)
epsilon=
gamma=

#we iterate over episodes
for e in range(no_episodes):
    # initialize the first state of the episode
    current_state = env.reset()
    done = False
    
    #sum the rewards that the agent gets from the environment
    total_episode_reward = 0
    
    for i in range(max_iter): 
        #
        # epsilon greedy behavior policy
        #
        # select an action based on epsilon greedy policy
        if (epsilon):
            action=random action
        else:
            action=greedy action

        # The environment mdp runs the chosen action and returns
        # the next state, a reward and true if the epiosed is ended.
        next_state, reward, done, _ = mdp.step(action)
        
        # PART (A)
        # We update our Q-table using the Q-learning iteration
        Q[current_state, action] = Q[current_state, action] +alpha*(reward + gamma*max(Q[next_state,:]))
        # END PART (A)
        
        total_episode_reward = total_episode_reward + reward
        # If the episode is finished, we leave the for loop
        if done:
            break
        current_state = next_state
    
    # decrease epsilon value
    exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
    rewards_per_episode.append(total_episode_reward)

#
# show Q values
#

#
# show the final policy
# in each state, choose action that maximizes Q value
#

#
# Question 3
#
# Sarsa
#
# modify PART (A) in Q learning as follows
# 1) next_action = action chosen in 'next_state' using epsilon greedy. 
# 2) max(Q[next_state,:] is replaced by Q[next_state, next_action]
# the rest of the code is same.
#

#
# show Q values
#

#
# show the final policy
# in each state, choose action that maximizes Q value
#
