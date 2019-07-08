import lunarlander
import matplotlib.pyplot as plt


PARAMS2 = [
    {'l_rate': 0.0001,  'gamma': 0.9, 'epsilon_decay': 0.998, 'min_epsilon': 0,  'max_epsilon': 1, 'buffer_zize': 5000,
     'batch_size': 32},
    {'l_rate': 0.0001,  'gamma': 0.94, 'epsilon_decay': 0.998, 'min_epsilon': 0,  'max_epsilon': 1, 'buffer_zize': 5000,
     'batch_size': 32},
    {'l_rate': 0.0001, 'gamma': 0.99, 'epsilon_decay': 0.998, 'min_epsilon': 0, 'max_epsilon': 1, 'buffer_zize': 5000,
     'batch_size': 32},
    {'l_rate': 0.0001, 'gamma': 0.999, 'epsilon_decay': 0.998, 'min_epsilon': 0, 'max_epsilon': 1, 'buffer_zize': 5000,
     'batch_size': 32}
]

if __name__ == "__main__":
    scorega = {0.9: [], 0.94: [],0.99: [], 0.999: []}
    ga = [0.9, 0.94,0.99,0.999]
    ep = 1500
    for para in PARAMS2:
        exp = lunarlander.Experiment(episode_num=ep, train_mode=True, para=para)
        exp.run()
        if para['gamma'] == 0.9:
            scorega[0.9]= exp.rep
        if para['gamma'] == 0.94:
            scorega[0.94]= exp.rep
        if para['gamma'] == 0.99:
            scorega[0.99]= exp.rep
        if para['gamma'] == 0.999:
            scorega[0.999]= exp.rep


    epsex = range(0, ep)
    for i in ga:
        if i == 0.9:
            plt.plot(epsex, scorega[i], 'r-', label='gamma = {}'.format(i))
        if i == 0.94:
            plt.plot(epsex, scorega[i], 'b-', label='gamma= {}'.format(i))
        if i == 0.99:
            plt.plot(epsex, scorega[i], 'g-', label='gamma = {}'.format(i))
        if i == 0.999:
            plt.plot(epsex, scorega[i], 'k-', label='gamma = {}'.format(i))

    #plt.xlim([0,ep+10])
    plt.legend()
    plt.ylabel('Score',size=20)
    plt.xlabel('Episode',size=20)
    plt.title('Gamma Search with other parameters fixed',size=14)
    plt.show()