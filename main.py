import numpy as np
import matplotlib.pyplot as plt
from models import env, plot_history, SeqHalf_with_change, PureExploration, config

if __name__ == "__main__":
    agent = SeqHalf_with_change(config)
    # agent = PureExploration(config)
    rew = agent.act(chng_arm=30,chng_time=200)
    # plot history
    plot_history(rew)

    # times, arms= agent.loop()
    # fig = plt.figure(figsize=[30,8])
    # plt.plot(arms, times, 'o')
    # plt.xlabel('Arm')
    # plt.ylabel('Time')
    # plt.show()