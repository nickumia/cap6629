## Program Setup

Borrowing from dynamic programming, the `GridEnv` class is initialized with many of the same constructs of 
a list of blocks, a list of final states, the reward and transition probability matrices.  
A new attribute was the starting state which defined the starting point for any given episode.
This was mentioned in the first homework; however, the starting position did not effect the learning as it
does in q-learning.  All of these data attributes are attached to the identity of the class to make them
persistent across function calls.

The second major element of the `GridEnv` class is the `step` function which defined the new state and reward
based on the transition probability and reward matrices in the initialization function given a specified action.
In the deterministic case, each action can only lead to one state; however, the code was written to provide the
possibility that the environment may choose from a set of states provided the action could result in multiple
possible states.

The last component of the `GridEnv` class is the `reset` function which resets the state of the environment to
the starting state provided in the initialization function.

Because the new state depends on the last state as well as the action, the `GridEnv` class is given a `current_state`
variable to track the position of the agent.  This was not in the provided design; however, it felt like a necessary
step to determine the `step` output.

The code uses the same parameters of 100 states with the same block and final state positions.  All of which can be
modified, either through input parameters or code changes.

A smaller value for `alpha` forces a slower learning model.  A smaller `epsilon` value ensures the agent takes more
optimal actions than random actions.  If `epsilon` is 0.2:
- 80% of the time, the agent chooses the greedy action and
- 20% of the time, the agent performs a random action.

The maximum number of iterations for a given episode determines how many actions should be taken before an episode is
deeemed hopeless where the agent will never reach the goal.  With a grid space of 100 states, if the agent hasn't
reached the goal within 200 steps, it is very safe to assume that the agent will not choose a good path to the goal.
If this value is too low, the results are not reliable since the agent may be able to get to the goal and the algorithm
ends prematurely.  If the value is too high, the algorithm wastes a lot of time pursuing non-optimal paths.

The number of episodes was set at 1000 as a test case.  The Q-values tended to not change too much after 1000, so
the value seemed safe enough.  The reason the Q-values stopped changing was because the exploration_decay essentially
made the agent greedy and just reinforce the same actions that it had taken previously.  There is room for optimization
of all of these variables.

## Implementation Details

The Q-learning algorithm is a simple for loop that iterates over each episode, calculating the Q-values cumulatively
through all episodes.  The agent starts in the starting state and chooses an action using epsilon-greedy behavior.
This behavior relies on the `random.random()` python built-in class.  After the agent has decided on which action to take,
the environment sends back the new state, reward and whether the agent has reached the goal.  If the agent took an
illegal action, the agent would remain in the same state and be forced to choose another action.  The key in making the
algorithm successful was providing an accurate reward, indicative of the agent wasting energy/time/resources taking an
illegal action.  Currently, there is no logic to prevent the agent from taking the same illegal move over and over again;
however, epsilon-greedy should randomize actions to help the agent choose different moves.

With the information about the new state and reward, the agent performs Q-learning, updating the Q-learning matrix based
on the iterative bellman equation.  If the agent reaches the final state, the for loop is broken and the agent moves on
to another episode.  Once all of the episodes are complete, the Q values are printed out to the user.

A modified version of the `extract_policy` function was written to compute the optimal policy based on the Q value matrix.
Since the Q value matrix directly has the optimal actions, the function iterates through all of the states and performs
the argmax operation.  The same helper function that outputs the optimal policy as a picture in HW1 is implemented in HW2,
providing a visual representation of the policy.  Please see the artifacts in the 
[latest run](https://github.com/nickumia/cap6629/actions/workflows/test.yml) for an example output.

![image](https://github.com/nickumia/cap6629/assets/10157100/1bb95456-8ef0-48f1-9bce-5b78a881745b)
