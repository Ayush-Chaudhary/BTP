import numpy as np
import matplotlib.pyplot as plt

# create an environment for the agent to interact with
class env(object):
    def __init__(self, mean, std):
        if len(mean)!=len(std):
            raise ValueError('mean and std must have the same length')
        self.mean = mean
        self.std = std
        self.k_arms = len(mean)
    
    def pull(self, arm):
        if arm not in range(self.k_arms):
            raise ValueError(f"arm {arm} must be in range of k_arms")
        return np.random.normal(self.mean[arm], self.std[arm])


# config
class config():
    arms = 16
    mean_dist = [i+1 for i in range(arms)]
    std = [1 for i in range(arms)]
    time = 2**13
    chng_arm = 2
    # chng_time = np.random.randint(0, time)
    # chng_time = 2**2
    new_mean = max(mean_dist) + 1
    new_mean_dist = [i+1 for i in range(arms)]
    new_mean_dist[chng_arm] = new_mean


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
    def __init__(self, config, chng_time, change = True):
        self.config = config
        self.time = config.time
        self.chng_arm = config.chng_arm
        self.chng_time = chng_time
        self.mean_dist = config.mean_dist
        self.new_mean_dist = config.new_mean_dist
        self.std = config.std
        self.change = change

    def act(self):
        env1 = env(self.mean_dist, self.std)
        env2 = env(self.new_mean_dist, self.std)
        arms = env1.k_arms
        change = self.change
        rounds = int(np.log2(arms))
        best_arms = np.array([i for i in range(arms)])
        arm_counts = np.zeros(arms)
        chng_time = self.chng_time
        t=0

        rewards = []
        cum_rewards = []
        for i in range(rounds):
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
            # select best half arms
            best_arms = np.argsort(arm_rewards)[::-1][:len(best_arms)//2]
        print(best_arms)
        return {"arms": arm_counts, "rewards": rewards, "cum_rewards": cum_rewards}


# Pure exploration where we change the mean of one arm
class PureExploration(object):
    def __init__(self, config, chng_time, change = True):
        self.config = config
        self.time = config.time
        self.chng_arm = config.chng_arm
        self.chng_time = chng_time
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
        print('Best arm is', np.argmax(arm_rewards))
        return {"arms": arm_counts, "rewards": rewards, "cum_rewards": cum_rewards}