
import replay
import model
import numpy as np
import random

class DDQNAgent:
    def __init__(self, env, l_rate, gamma, epsilon_decay, min_epsilon,
                 max_epsilon, buffer_zize, batch_size):
        self.env = env
        self.replay = replay.Relay(buffer_zize)
        self.n_s = env.observation_space.shape[0]
        self.n_a = env.action_space.n
        self.batch_size = batch_size
        self.full_memory = False
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.cur_epsilon = max_epsilon
        self.gamma = gamma
        self.l_rate = l_rate
        self.models = model.Model(self.n_s, self.n_a, 128, self.l_rate)


    def get_Action(self, s):
        if np.random.rand() < self.cur_epsilon:
            #print("cur_epsilon: " + str(self.cur_epsilon))
            action = self.env.action_space.sample()
            #print("r_action: " + str(action))
            return action
        else:
            pred = self.models.predict(s)
            action = np.argmax(pred[0])
            #print("p_action: " + str(action))
            return action

    def add_memory(self, s, a, r, s_prime, done):
        self.replay.store_exp((s, a, r, s_prime, done))

    def update_target_model(self):
        self.models.update_target_model()

    def learn(self):
        batch = random.sample(self.replay.memory, self.batch_size)
        batch = np.array(batch)
        y = np.copy(batch[:, 2])
        no_term_inds = np.where(batch[:, 4] == False)


        if len(no_term_inds[0]) > 0:
            p_sprime_target = self.models.target_model.predict(np.vstack(batch[:, 3]))
            p_sprime_online = self.models.online_model.predict(np.vstack(batch[:, 3]))
            y[no_term_inds] += np.multiply(self.gamma, p_sprime_target[no_term_inds, np.argmax(p_sprime_online[no_term_inds, :][0],axis=1)][0])

        a = np.array(batch[:, 1], dtype=int)
        y_t = self.models.online_model.predict(np.vstack(batch[:, 0]))
        y_t[range(self.batch_size), a] = y
        self.models.online_model.fit(np.vstack(batch[:, 0]), y_t, epochs=1, verbose=0)

    def save(self,name):
        self.models.online_model.save_weights(name)

