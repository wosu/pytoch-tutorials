'''
Deep Reinforcement technique is used in this article to optimize Ad placement on a website
to maximize the probability of user clicks and increase digital marketing revenue.
A detailed case study with code is presented to help users implement the solution on any real world example.

refer:https://towardsdatascience.com/deep-reinforcement-learning-hands-on-for-optimized-ad-placement-b402ffa47245
https://github.com/NandaKishoreJoshi/Reinforcement_Lerning/blob/main/RL_course/1_Ad_placement.ipynb
'''
import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt
import numpy as np
import torch


def softmax(av, tau=1.12):
    '''
    define a softmax function which is used for explore and exploit feature in RL
    :param av:
    :param tau:
    :return:
    '''
    softm = ( np.exp(av / tau) / np.sum( np.exp(av / tau) ) )
    return softm


class ContextBandit:
    '''
    Environmet proviedes all the required information to the RL agent. Defining a Environment is very
      important and usually its done by the SME's
    Creating a environment class for our Ad Placement problem.
    '''
    def __init__(self, arms=10):
        self.arms = arms
        self.init_distribution(arms)
        self.update_state()

    def init_distribution(self, arms):
        # Num states = Num Arms to keep things simple
        self.bandit_matrix = np.random.rand(arms, arms)
        # each row represents a state, each column an arm

    def reward(self, prob):
        reward = 0
        for i in range(self.arms):
            if random.random() < prob:
                reward += 1
        return reward

    def get_state(self):
        return self.state

    def update_state(self):
        self.state = np.random.randint(0, self.arms)

    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])

    def choose_arm(self, arm):
        reward = self.get_reward(arm)
        self.update_state()
        return reward


def one_hot(N, pos, val=1):
    one_hot_vec = np.zeros(N)
    one_hot_vec[pos] = val
    return one_hot_vec


def running_mean(x,N=50):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y


def train(env, epochs=5000, learning_rate=1e-2):
    cur_state = torch.Tensor(one_hot(arms,env.get_state())) #A
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rewards = []
    for i in range(epochs):
        y_pred = model(cur_state) #B
        av_softmax = softmax(y_pred.data.numpy(), tau=2.0) #C
        av_softmax /= av_softmax.sum() #D
        choice = np.random.choice(arms, p=av_softmax) #E
        cur_reward = env.choose_arm(choice) #F
        one_hot_reward = y_pred.data.numpy().copy() #G
        one_hot_reward[choice] = cur_reward #H
        reward = torch.Tensor(one_hot_reward)
        rewards.append(cur_reward)
        loss = loss_fn(y_pred, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_state = torch.Tensor(one_hot(arms,env.get_state())) #I
    return np.array(rewards),cur_state

arms = 10
N, D_in, H, D_out = 1, arms, 100, arms

env = ContextBandit(arms=10)
state = env.get_state()
reward = env.choose_arm(1)
print(state,reward)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.ReLU(),
)

loss_fn = torch.nn.MSELoss()
env = ContextBandit(arms)
rewards,state = train(env)
plt.plot(running_mean(rewards,N=500))
plt.show()
print(len(rewards))

env = ContextBandit(arms=10)
state = torch.Tensor(one_hot(arms,env.get_state()))
print(state)
y_pred = model(state)
print(y_pred)
av_softmax = softmax(y_pred.data.numpy(), tau=2.0)
av_softmax /= av_softmax.sum()
print(av_softmax)
choice = np.random.choice(arms, p=av_softmax)
print(choice)
cur_reward = env.choose_arm(choice)
print(cur_reward)