import numpy as np
import matplotlib.pyplot as plt

from run_cartpole import CartPoleLinearPolicy, run_policy
from finite_diff import scalar_finite_diff


def train_cartpole_policy(num_training_iterations, num_rollouts_per_eval, learning_rate):

    def eval_params(params):
        '''
        TODO(Q3.1): 

        Implement the eval_params function, which takes in a policy params and:
        - forms a policy
        - runs it for num_rollouts_per3_eval number of times
        - returns the mean reward across these rollouts
        '''

        rewards = 0
        if params.shape != (4,):
            params = np.array([params[0,0],params[1,1],params[2,2],params[3,3]])

        eval_policy = CartPoleLinearPolicy(params)
        for i in range(num_rollouts_per_eval):
            rewards += run_policy(eval_policy, render=False)
        rewards /= num_rollouts_per_eval

        return rewards

    params = np.zeros(4)

    rews = []
    rews.append(eval_params(params))
    h = np.ones(4) * 1e-2
    for _ in range(num_training_iterations):
        '''
        TODO(Q3.2): 
        
        - Implement the gradient ascent step to update params using scalar_finiite_diff
        - At the end of each iteration, use eval_params to compute the reward of a policy with the current params and record it in rews
        '''
        # g(theta) = E[reward(policy(theta))]
        params += learning_rate * scalar_finite_diff(eval_params, params, h)
        rews.append(eval_params(params))

    policy = CartPoleLinearPolicy(params)

    return policy, rews


if __name__ == "__main__":
    # Don't change these!
    max_reward = 500
    num_training_iterations = 20
    num_rollouts_per_eval = 20

    '''
    TODO(Q3.3): 
    
    - Plot the rews during training vs. training iterations
    - Tune the learning_rate so the training converges to max reward (500) in 7 iterations or less.
    - Record the values of the final policy parameters
    '''
    learning_rate = 2e-5

    train_policy, train_rew = train_cartpole_policy(num_training_iterations, num_rollouts_per_eval, learning_rate)

    print(train_policy.params)
    plt.plot(train_rew)
    plt.xlabel('iteration')
    plt.ylabel('rewards')
    plt.title('learning rate = '+ str(learning_rate))
    plt.show()
    