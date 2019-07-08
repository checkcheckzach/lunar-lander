import lunarlander
import matplotlib.pyplot as plt


PARAMS1 = [
    {'l_rate': 0.001,  'gamma': 0.99, 'epsilon_decay': 0.998, 'min_epsilon': 0,  'max_epsilon': 1, 'buffer_zize': 5000,
     'batch_size': 32},
    {'l_rate': 0.0005, 'gamma': 0.99, 'epsilon_decay': 0.998, 'min_epsilon': 0, 'max_epsilon': 1, 'buffer_zize': 5000,
     'batch_size': 32},
    {'l_rate': 0.0003, 'gamma': 0.99, 'epsilon_decay': 0.998, 'min_epsilon': 0, 'max_epsilon': 1, 'buffer_zize': 5000,
     'batch_size': 32},
    {'l_rate': 0.0001, 'gamma': 0.99, 'epsilon_decay': 0.998, 'min_epsilon': 0, 'max_epsilon': 1, 'buffer_zize': 5000,
     'batch_size': 32}
]


if __name__ == "__main__":
    scoreLRate = {0.001: [], 0.0005: [],0.0003: [], 0.0001: []}
    lrate = [0.001, 0.0005,0.0003,0.0001]
    ep = 1500
    for para in PARAMS1:
        exp = lunarlander.Experiment(episode_num=ep, train_mode=True, para=para)
        exp.run()
        if para['l_rate'] == 0.001:
            scoreLRate[0.001]= exp.rep
        if para['l_rate'] == 0.0005:
            scoreLRate[0.0005]= exp.rep
        if para['l_rate'] == 0.0003:
            scoreLRate[0.0003]= exp.rep
        if para['l_rate'] == 0.0001:
            scoreLRate[0.0001]= exp.rep


    epsex = range(0, ep)
    for i in lrate:
        if i == 0.001:
            plt.plot(epsex, scoreLRate[i], 'r-', label='α = {}'.format(i))
        if i == 0.0005:
            plt.plot(epsex, scoreLRate[i], 'b-', label='α = {}'.format(i))
        if i == 0.0003:
            plt.plot(epsex, scoreLRate[i], 'g-', label='α = {}'.format(i))
        if i == 0.0001:
            plt.plot(epsex, scoreLRate[i], 'k-', label='α = {}'.format(i))

    #plt.xlim([0,ep+10])
    plt.legend()
    plt.ylabel('Score',size=20)
    plt.xlabel('Episode',size=20)
    plt.title('Alpha Search with other parameters fixed',size=18)
    plt.show()