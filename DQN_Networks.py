import tensorflow as tf
import random
import numpy as np
from keras import Sequential, layers
from collections import deque, namedtuple

MAX_CAPACITY = 1000
MIN_REPLAY_MEMORY_SIZE = 500
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQN:
    def __init__(self, state_size, action_size):
        self.Q_model = self.build_model(state_size, action_size)
        self.target_model = self.build_model(state_size, action_size)
        self.target_model.set_weights(self.Q_model.get_weights())
        self.replay_buffer = deque(maxlen=MAX_CAPACITY)
        self.gamma = 0.95

    def build_model(self, state_size, action_size):
        Q_network = Sequential()
        Q_network.add(layers.Dense(units=8, activation='relu', input_dim=2))
        Q_network.add(layers.Dense(units=action_size, activation='linear'))
        return Q_network

    def add_memory(self, state, action, reward, next_state, done):
        if (len(self.replay_buffer) == self.replay_buffer.maxlen):
            self.replay_buffer.clear()
            self.replay_buffer.append(Experience(state, action, reward, next_state, done))
        else:
            self.replay_buffer.append(Experience(state, action, reward, next_state, done))

    def predict_q_value(self, state, model):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        predicted_q_value = model(state)
        return predicted_q_value

    def train_network(self):
        experience_sample = random.sample(self.replay_buffer, MIN_REPLAY_MEMORY_SIZE)

        states = [experience.state for experience in experience_sample]
        rewards = [experience.reward for experience in experience_sample]
        actions = [experience.action for experience in experience_sample]
        next_states = [experience.next_state for experience in experience_sample]
        done = [experience.done for experience in experience_sample]
        states = tf.concat(states, axis=0)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        q_pred = self.predict_q_value(states, self.Q_model).numpy()
        q_next = self.predict_q_value(next_states, self.target_model)
        for i in range(MIN_REPLAY_MEMORY_SIZE):

            if (done[i]):
                q_pred[actions[i]] = rewards[i]
            else:
                q_max = np.max(q_next[i].numpy())
                q_pred[actions[i]] = rewards[i] + self.gamma * q_max

        q_pred = tf.convert_to_tensor(q_pred, dtype=tf.float32)
        self.Q_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.Q_model.fit(states, q_pred, epochs=1, verbose=0)

        self.target_model.set_weights(self.Q_model.get_weights())