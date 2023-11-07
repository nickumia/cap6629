## Program Setup

The maze world was defined as an `√n` by `√n` square grid.  Each square in the grid represents a state and each state is numbered from `0` to `n-1`.  

There are four actions: `(0) up`, `(1) down`, `(2) left` and `(3) right`.

The reward is represented by an `n by 4` matrix.  The reward for each action from every state is `-1` with two special conditions: (1) the reward for an out-of-boundary move is -10 and (2) the reward for an end state is 0.  There is a third special condition: blocks.  The reward for a move to a state that contains a block is the same as an out-of-boundary move, i.e. -10.

The program defines two arrays: (1) blocks and (2) final states.  The blocks is a list of states that have blocks in them.  The final states are a list of states that define the goals.  Both of these arrays effect the Reward and Transition Probability Matrices.  The effect on reward is described above.  Transition probability to a block is impossible (i.e. 0).  Transition from a block is also impossible.  Transition from an end state is possible and transition to an end state is also possible.

For the deterministic case, the transition probability is defined as a matrix with dimensions as n by n by 4.  Each state can only transition to any other state using adjacency rules.  Assuning a 10 by 10 grid, 
-	State 0 can only transition to State 1 through the “right” action.
-	State 0 can only transition to State 10 through the “down” action.
-	State 0 can NOT transition to any other state.
-	State 11 can only transition to State 12 through the “right” action.
-	State 11 can only transition to State 10 through the “left” action.
-	State 11 can only transition to State 1 through the “up” action.
-	State 11 can only transition to State 21 through the “down” action.
-	State 11 can NOT transition to any other state.

For the probabilistic case, the transition probability has the same dimensions as the deterministic case; however, there is slippage when moving in any of the directions.  Borrowing the example above, State 11 would have the following probabilities,
-	Taking the “right” action from State 11 will have the following outcomes:
    -	State 11 can transition to State 12 with a probability of 80%.
    - State 11 can transition to State 13 with a probability of 5%.
    -	State 11 can transition to State 2 with a probability of 5%.
    -	State 11 can transition to State 22 with a probability of 5%.
    -	State 11 can transition to State 11 with a probability of 5%.
-	Taking the “left” action from State 11 will have the following outcomes:
    -	State 11 can transition to State 10 with a probability of 80%.
    -	State 11 can transition to State 0 with a probability of 5%.
    -	State 11 can transition to State 20 with a probability of 5%.
    -	State 11 can transition to State 11 with a probability of 5%.
    -	Because 10 is an edge state, it cannot slip into the wall.  The extra 5% isn’t lost.  Instead the probabilities would be 84%, 5.2%, 5.2% and 5.2%.

There is extra setup related to creating animations and printing the generated policy out as picture.  This borrows heavily from my work in Dr. Zhu’s [CAP6635](https://github.com/nickumia/cap6635) class, please see [the code for details](https://github.com/nickumia/cap6635/blob/main/cap6635/utilities/plot.py).


## Implementation Details

There are four main functions: (1) Policy Evaluation (`policy_eval`), (2) Policy Extraction (`extract_policy`), (3) Policy Iteration (`policy_iter`) and (4) Value Iteration (`value_iter`).

### Policy Evaluation (`policy_eval`)

The input to this function is a policy with the dimensions of `n x 4` and the maximum number of iterations that the function should consider.  The function initializes three variables:
-	State value matrix, `n x 1`
-	Convergence matrix, `n x 1`
-	The count of how many iterations have been taken

While the convergence computation does not equal a matrix of all zeros, the function keeps running the bellman equation.  The bellman equation computes a matrix of new state values based on the last iteration of state values.  The difference between the new and old state values is then compared to the convergence matrix and if it is not equal (i.e. there is a change in state values), the loop keeps running.  Each iteration also increments the iteration counter and if it exceeds the maximum number of iterations defined in the input, then the loop ends as well.

The function returns the converged state values or the state values at the maximum allowed number of iterations.

### Policy Extraction (`extract_policy`)

The input to this function is a state value matrix `n x 1`.

An empty policy matrix is instantiated to store the extracted policy with the dimensions of n x 4.  For each state in the input matrix, an action value is computed for all of the possible actions.  Out of all of the actions, the best action is chosen and the policy becomes a deterministic policy with each state only selecting the best action. For example, if the action values for State 24 were 
-	“up”: 4.5
-	“down”: 6.2
-	“left”: 4.3
-	“right”: 5.7
The policy for State 24 would be to chose down, i.e. `P[24]=[0 1 0 0]`.

The function returns the policy that relates to the best actions based on the inputted state values.

### Policy Iteration (`policy_iter`)

The input to this function is an initial policy (in this case, the random uniform policy) and the maximum number of iterations.

Starting with policy evaluation, the function alternates between policy evaluation and policy extraction until the state values converge in the same way that the policy extraction computes convergence.

If the function completes within the maximum allowed iterations, this function returns the optimal policy, the optimal state values and the number of iterations that it took to get there.  Otherwise, it returns the final policy and state values at the maximum number of iterations.

### Value Iteration (`value_iter`)

The input to this function is an initial set of state values (in this case, all zero values) and a maximum number of iterations.

For each iteration, the bellman equation is run for all actions and the value of the new state value is directly the best action value.  There is no intermediary policy generated.  The difference between the old and new state values are computed and while it state values do not converge, the function keeps iterating.

If the function completes within the maximum allowed iterations, this function returns the optimal state values and the number of iterations that it took to get there.  Otherwise, it return the final state values at the maximum number of iterations.

### Other Logic

There is special logic in the code that lets you choose which algorithm to run and based on the selection, it will run the relevant code.  There is also plotting logic in the code that lets you visualize the policy as a picture.  The code also generates an GIF animation of the grid world given a starting point.

There is a super cool diagram that @RaviMachavarapu, @rishabhlingam and I came up with.  Please see it for more details too!
