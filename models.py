import numpy as np
import matplotlib.pyplot as plt
from probabilities import get_probs, calc_p12

# create an environment for the agent to interact with
class env(object):
    def __init__(self, arms, chng = False, chng_arm = 0):
        self.mean = [i+1 for i in range(arms, 0, -1)]
        self.std = [1 for i in range(arms)]
        if chng:
            self.mean[chng_arm] = max(self.mean)+1
        self.k_arms = arms
    
    def pull(self, arm):
        if arm not in range(self.k_arms):
            raise ValueError(f"arm {arm} must be in range of k_arms")
        return np.random.normal(self.mean[arm], self.std[arm])


# config
class config():
    arms = 32
    mean_dist = [i+1 for i in range(arms, 0, -1)]
    std = [1 for i in range(arms)]
    time = 2**11
    chng_arm = 2
    chng_time = 10
    new_mean = max(mean_dist) + 1
    new_mean_dist = [i+1 for i in range(arms, 0, -1)]
    new_mean_dist[chng_arm] = new_mean
    a, rs =arms, -1
    while a>1:
        a1, a2 = a, a//2
        if chng_arm>a2 and chng_arm<=a1: 
            break
        else : 
            rs+=1
            a = a2
    times=[0,0]
    t = 0
    best_arms = arms
    for i in range(int(np.log2(arms))):
        times[0]=t
        r_time = int(time/(np.log2(arms)*best_arms))
        for j in range(int(r_time)):
            for k in range(int(best_arms)):
                t+=1
        best_arms = int(best_arms/2)
        times[1]=t
        if times[0]<=chng_time and times[1]>=chng_time: 
            rc=i
            break


# function to plot history
def plot_history(history):
  cum_rewards = history["cum_rewards"]
  chosen_arms = history["arms"]

  fig = plt.figure(figsize=[30,8])

  ax2 = fig.add_subplot(121)
  ax2.plot(cum_rewards, label="avg rewards")
  ax2.set_title("Cummulative Rewards")

  ax3 = fig.add_subplot(122)
  ax3.bar([i for i in range(len(chosen_arms))], chosen_arms, label="chosen arms")
  ax3.set_title("Chosen Actions")
  plt.show()


# Random Agent
class RandomAgent(object):

  def __init__(self, env, max_iterations=2000):
    self.env = env
    self.iterations = max_iterations

  def act(self):
    arm_counts = np.zeros(self.env.k_arms)
    rewards = []
    cum_rewards = []

    for i in range(1, self.iterations + 1):
      arm = np.random.choice(self.env.k_arms)
      reward = self.env.pull(arm)

      arm_counts[arm] += 1
      rewards.append(reward)
      cum_rewards.append(sum(rewards)/ len(rewards))

    return {"arms": arm_counts, "rewards": rewards, "cum_rewards": cum_rewards}


# Epsilon Greedy Agent
class EpsilonGreedyAgent(object):

  def __init__(self, env, max_iterations=200, epsilon=0.01, decay=0.001, decay_interval=50):
    self.env = env 
    self.iterations = max_iterations
    self.epsilon = epsilon 
    self.decay = decay 
    self.decay_interval = decay_interval

  def act(self):
    q_values = np.zeros(self.env.k_arms)
    arm_rewards = np.zeros(self.env.k_arms)
    arm_counts = np.zeros(self.env.k_arms)

    rewards = []
    cum_rewards = []

    for i in range(1, self.iterations + 1):
      arm = np.random.choice(self.env.k_arms) if np.random.random() < self.epsilon else np.argmax(q_values)
      reward = self.env.pull(arm)

      arm_rewards[arm] += reward
      arm_counts[arm] += 1
      q_values[arm] = arm_rewards[arm]/arm_counts[arm]

      rewards.append(reward)
      cum_rewards.append(sum(rewards)/ len(rewards))

      if i % self.decay_interval == 0:
        self.epsilon = self.epsilon * self.decay 

    return {"arms": arm_counts, "rewards": rewards, "cum_rewards": cum_rewards}


# Sequential Halving with option to change mean of one arm
class SeqHalf_with_change(object):
    def __init__(self, config, change = True):
        self.config = config
        self.arms = config.arms
        self.time = config.time
        # self.chng_arm = config.chng_arm
        # self.chng_time = config.chng_time
        self.mean_dist = config.mean_dist
        self.new_mean_dist = config.new_mean_dist
        self.std = config.std
        self.change = change

    def act(self, chng_arm, chng_time):
        env1 = env(self.arms, chng = False)
        env2 = env(self.arms, chng = True, chng_arm = chng_arm)
        arms = env1.k_arms
        change = self.change
        rounds = int(np.log2(arms))
        best_arms = np.array([i for i in range(arms)])
        arm_counts = np.zeros(arms)
        chng_time = chng_time
        t=0

        probs = {'p11':[], 'p12':0, 'p13':[], 'p21':[], 'p22':[], 'p23':0, 'p24':[]}
        rewards = []
        cum_rewards = []
        times = [0,0]
        for i in range(rounds):
            times[0]=t
            arm_rewards = np.zeros(arms)
            r_time = self.time//(rounds*len(best_arms))
            for j in range(r_time):
                for k in best_arms:
                    if t < chng_time: reward = env1.pull(k)
                    else:
                        if change: reward = env2.pull(k)
                        else: reward = env1.pull(k)
                    t+=1
                    arm_rewards[k] += reward
                    arm_counts[k] += 1
                    rewards.append(reward)
                    cum_rewards.append(sum(rewards)/ len(rewards))
            arm_rewards = arm_rewards/r_time
            times[1]=t
            # select best half arms
            best_arms = np.argsort(arm_rewards)[::-1][:len(best_arms)//2]
            # get_probs(config, best_arms, times, chng_time, t, probs, i)
        # print('Best arm is', best_arms[0])
        return {"arms": arm_counts, "rewards": rewards, "cum_rewards": cum_rewards, "probs": probs, 'best_arm': best_arms[0]}

    def loop(self):
        arms = self.config.arms
        time = self.config.time
        arm_list = [i for i in range(arms)]
        tim = []
        for i in range(arms):
            if i%4==0:print(i)
            j=time
            run = True
            while run and j>=0:
                hist = self.act(i, j)
                # print(j)
                if hist['best_arm'] >0:
                    tim.append(j)
                    run = False
                j-=1
            if j<0: tim.append(time-1)
        print(tim)
        return tim, arm_list                 

# Pure exploration where we change the mean of one arm
class PureExploration(object):
    def __init__(self, config, change = True):
        self.config = config
        self.time = config.time
        self.chng_arm = config.chng_arm
        self.chng_time = config.chng_time
        self.mean_dist = config.mean_dist
        self.new_mean_dist = config.new_mean_dist
        self.std = config.std
        self.change = change

    def act(self):
        env1 = env(self.mean_dist, self.std)
        env2 = env(self.new_mean_dist, self.std)
        arms = env1.k_arms
        change = self.change
        arm_counts = np.zeros(arms)
        chng_time = self.chng_time
        arm_rewards = np.zeros(arms)
        t=0

        rewards = []
        cum_rewards = []
        r_time = self.time//(arms)
        for j in range(r_time):
            for k in range(arms):
                if t < chng_time: reward = env1.pull(k)
                else:
                    if change: reward = env2.pull(k)
                    else: reward = env1.pull(k)
                t+=1
                arm_counts[k] += 1
                rewards.append(reward)
                cum_rewards.append(sum(rewards)/ len(rewards))
                arm_rewards[k] += reward
        prob = calc_p12(config, arm_counts, 0, self.time, chng_time)
        print('Best arm is', np.argmax(arm_rewards))
        return {"arms": arm_counts, "rewards": rewards, "cum_rewards": cum_rewards, "probs": prob}