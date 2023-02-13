import numpy as np
import matplotlib.pyplot as plt
from models import env, plot_history, SeqHalf_with_change, PureExploration, config, analysis

if __name__ == "__main__":
    agent = SeqHalf_with_change(config)
    # agent = PureExploration(config)
    # chng_time= np.random.randint(0,config.time)
    # chng_arm = np.randon.randint(0,int(config.arms*top_percent))
    # rew = agent.act(chng_arm=30,chng_time=2046)
    # print(rew['probs'])
    # plot history
    # plot_history(rew)

    # times, arms= agent.loop()
    # fig = plt.figure(figsize=[30,8])
    # plt.plot(arms, times, 'o')
    # plt.xlabel('Arm')
    # plt.ylabel('Time')
    # plt.show()

    N, probabs_seq, probas_exp, probas_mix=[16, 32, 64, 128, 256, 512, 1024], [], [], []
    # iters = 40000
    # top_percent = 0.2
    # for n in N:
    #     config.arms=n
    #     config.time = 2**14
    #     print(config.arms, config.time)
    #     # print(config.mean_dist)
    #     agent = analysis(config)
    #     dic = agent.loop(top_percent, iters=iters)
    #     probabs_seq.append(dic['seq_half'])
    #     probas_exp.append(dic['pure_exp'])
    #     probas_mix.append(dic['mix'])
    #     print('pure exploration : ', probas_exp)
    #     print('seq half : ', probabs_seq)
    #     print('mix : ', probas_mix)

    # # plot curves in same figure
    # fig = plt.figure(figsize=[30,8])
    # # connect the points with lines
    # plt.plot(N, probabs_seq, 'o', label='SeqHalf', linestyle='dashed', linewidth = 3)
    # plt.plot(N, probas_exp, 'o', label='PureExploration', linestyle='dashed', linewidth = 3)
    # plt.plot(N, probas_mix, 'o', label='Mix', linestyle='dashed', linewidth = 3)
    # plt.xlabel('Number of arms')
    # plt.ylabel('Probability of choosing the best arm')
    # plt.legend()
    # plt.show()

    fig = plt.figure(figsize=[30,8])
    # # connect the points with lines
    
    # iters = 10000
    # plt.plot(N, [0.8764, 0.7803, 0.6778, 0.5874, 0.5116, 0.3739, 0.2681], 'o', label='T = 8192', linestyle='dashed', linewidth = 3)
    # plt.plot(N, [0.8762, 0.7872, 0.6802, 0.5854, 0.5234, 0.4372, 0.3323], 'o', label='T = 16384', linestyle='dashed', linewidth = 3)
    # plt.plot(N, [0.8735, 0.7773, 0.6829, 0.5969, 0.5278, 0.4627, 0.3862], 'o', label='T = 32768', linestyle='dashed', linewidth = 3)
    plt.plot(N, [0.8768, 0.7796, 0.6916, 0.5968, 0.5356, 0.4699, 0.4027], 'o', label='T = 65536, 10000 itterations', linestyle='dashed', linewidth = 3)
    
    # T= 16384
    # plt.plot(N, [0.8762, 0.7872, 0.6802, 0.5854, 0.5234, 0.4372, 0.3323], 'o', label='10000 itterations', linestyle='dashed', linewidth = 3)
    # iters = 20000
    # plt.plot(N, [0.87205, 0.7838, 0.68535, 0.59405, 0.51995, 0.4378, 0.3386], 'o', label='20000 itterations', linestyle='dashed', linewidth = 3)
    # iters = 30000
    plt.plot(N, [0.87407, 0.78084, 0.68146, 0.59463, 0.52246, 0.43306, 0.34253], 'o', label='T = 16384, 30000 itterations', linestyle='dashed', linewidth = 3)
    # iters = 40000
    # plt.plot(N, [0.8744, 0.777325, 0.685875, 0.59105, 0.5226, 0.432125, 0.3368], 'o', label='40000 itterations', linestyle='dashed', linewidth = 3)
    plt.xlabel('Number of arms')
    plt.ylabel('Probability of choosing the best arm')
    plt.legend()
    plt.show()