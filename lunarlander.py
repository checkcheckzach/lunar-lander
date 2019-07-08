
import gym
import agent
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import time



bparams = {'l_rate': 0.0001,  'gamma': 0.99, 'epsilon_decay': 0.998, 'min_epsilon': 0,  'max_epsilon': 1, 'buffer_zize': 5000,
     'batch_size': 32}

class Experiment:
    def __init__(self, environment='LunarLander-v2', episode_num=2500, train_mode=True, para=bparams):
        self.env = gym.make(environment)
        self.episode_num = episode_num
        self.reward_avg = deque([], maxlen=100)
        self.agent = agent.DDQNAgent(self.env, para['l_rate'], para['gamma'], para['epsilon_decay'],
                                     para['min_epsilon'],  para['max_epsilon'], para['buffer_zize'], para['batch_size'])
        self.train_mode = train_mode
        self.rep = []

    def run(self):
        train_weight = "lunar-lander_solved.h5"
        agent = self.agent
        env = self.env
        n_s = env.observation_space.shape[0]
        train_mode = self.train_mode
        if not train_mode:
            agent.models.online_model.load_weights('./lunar-lander_solved.h5')
            self.episode_num = 99
            self.agent.models.cur_epsilon = 0

        epsp = []
        rep = []
        avgp = 0

        stTime = time.time()
        for cur_episode in range(self.episode_num):
            cur_reward = 0
            s = env.reset()
            s = np.reshape(s, [1, n_s])
            done = False
            while not done:
                a = agent.get_Action(s)
                #print("action: " + str(a))
                s_prime, r, done, nouse = env.step(a)
                s_prime = np.reshape(s_prime, [1, n_s])
                if train_mode:
                    agent.add_memory(s,a, r, s_prime,done)
                cur_reward +=r
                s = s_prime
                if train_mode and len(agent.replay.memory) > agent.batch_size:
                    agent.learn()

            self.reward_avg.append(cur_reward)

            if train_mode:
                agent.update_target_model()

            if agent.cur_epsilon > agent.min_epsilon:
                agent.cur_epsilon *= agent.epsilon_decay
            avg_reward = np.average(self.reward_avg)

            avgp = avg_reward

            if not train_mode:
                print("Episode Nr. {} \nCurrent Reward: {} \nAverage 100 Reward: {} ".format(cur_episode, cur_reward, avg_reward))
                epsp.append(cur_episode)
                rep.append(cur_reward)

            if train_mode:
                epsp.append(cur_episode)
                rep.append(cur_reward)

            if train_mode and cur_episode % 100 == 0:
                print("Episode Nr. {} \nCurrent Reward: {} \nAverage 100 Reward: {}".format(cur_episode, cur_reward, avg_reward))
                curTime = time.time()
                print('Elapsed time : {} mins\n'.format((curTime - stTime)/60.0))


            if train_mode and cur_episode == (self.episode_num-1):
                agent.save(train_weight)
                print("Episode Nr. {} \nCurrent Reward: {} \nAverage 100 Reward: {}".format(cur_episode, cur_reward,
                                                                                            avg_reward))
                curTime = time.time()
                print('Elapsed time : {} mins\n'.format((curTime - stTime) / 60.0))
                #print("Problem Solved")
                self.rep = rep
                break

        if not train_mode:
            plt.figure(figsize=(10, 6))
            plt.plot(epsp, rep, 'ko-')
            plt.ylabel('Score', size=20)
            plt.xlabel('Episode', size=20)
            plt.title('Trained Agent 100 trials Performance', size=18)
            plt.plot([0, 100], [avgp, avgp], 'b-')
            plt.show()
        else:
            plt.figure(figsize=(10, 6))
            plt.plot(epsp, rep, 'k-')
            plt.ylabel('Score', size=20)
            plt.xlabel('Episode', size=20)
            plt.title('Training Learning Curve', size=18)
            plt.show()


if __name__ == "__main__":
    exp = Experiment(episode_num=2500, train_mode=True, para=bparams)
    exp.run()