import numpy as np
import matplotlib.pyplot as plt

import gym


class CartPoleLinearPolicy:

    def __init__(self, params):
        assert params.shape == (4,)
        self._params = params

    @property
    def params(self):
        return self._params.copy()

    def __call__(self, obs):
        '''
        TODO(Q1.1):

        Implement the binary linear policy.
        The policy should return 0 if the dot product of its parameters with the observation is negative, and 1 otherwise.
        '''
        reward = np.dot(np.transpose(self.params), obs)

        if reward >=0:
            res=1
        else:
            res=0

        return res


def run_policy(policy, render=False):
    env = gym.make('CartPole-v1')
    obs = env.reset()

    rews = []
    while True:
        action = policy(obs)

        if render:
            env.render()

        obs, rew, done, _ = env.step(action)
        rews.append(rew)

        if done:
            break
    
    env.close()

    return np.sum(rews)


if __name__ == "__main__":
    '''
    TODO(Q1.2): Try filling in different numbers for params and run this script. Observe CartPole behavior
    '''

    # location, linear velocity, angle, angular velocity
    # params = np.array([0, 0, 1, 1])
    # policy = CartPoleLinearPolicy(params)
    # rew = run_policy(policy, render=False)
    #
    # print('Params {} got reward {}'.format(params, rew))

    '''
    TODO(Q1.3): Sample 1000 policies and run all:
    - Plot histogram of total rewards
    - Count what percentage of random policies achieved full reward (500)

    '''
    policies = [CartPoleLinearPolicy(np.random.uniform(-1,1,4)) for _ in range(1000)]

    # policies = []
    # for i in range(1000):
    #     policies.append(CartPoleLinearPolicy(np.random.uniform(-1,1,4)))

    all_rews = np.array([run_policy(policy, render=False) for policy in policies])
    
    plt.figure(figsize=(8, 6))
    plt.hist(all_rews)
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.title('CartPole Random Policy Performance Histogram')
    plt.savefig('p1-q1.png')
    plt.show()

    num_full_rewards = np.sum(all_rews == 500)
    print('Percetange of policies that achieved full reward: {}'.format(num_full_rewards / len(all_rews)))
