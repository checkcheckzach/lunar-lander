

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Model:

    def __init__(self, states_num, actions_num,
                 hidden_size, l_rate):
        self.hizzen_size = hidden_size
        self.l_rate = l_rate
        self.states_num = states_num
        self.actions_num = actions_num
        self.online_model = self.create_model(states_num, actions_num,
                 hidden_size, l_rate)
        self.target_model = self.create_model(states_num, actions_num,
                 hidden_size, l_rate)

    def create_model(self, states_num, actions_num, hidden_size, l_rate):
        model = Sequential()
        model.add(Dense(hidden_size, input_dim=states_num, activation='relu'))
        model.add(Dense(hidden_size, activation='relu'))
        model.add(Dense(actions_num, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.l_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.online_model.get_weights())

    def predict(self, state):
        return self.online_model.predict(state)