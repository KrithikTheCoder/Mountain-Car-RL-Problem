import random
import numpy as np
import tensorflow as tf
import gym
from DQN_Networks import DQN, MIN_REPLAY_MEMORY_SIZE

EPISODES = 2000
EPSILON = 0.99
MIN_EPSILON = 0.1
DECAY_RATE = 0.995
SHOW_EVERY = 10
result_dict = {'ep': [], 'reward': []}
env = gym.make('MountainCar-v0')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

dqn_network = DQN(state_size, action_size)

for episode in range(EPISODES):
    done = False
    total_reward = 0
    it = 0
    curr_state= env.reset()
    while not done and it <= 200:
        curr_state = np.reshape(curr_state, (1, -1))
        curr_state = tf.convert_to_tensor(curr_state, dtype=tf.float32)
        if random.random() < EPSILON:
            q_predict = dqn_network.predict_q_value(curr_state, dqn_network.Q_model)
            # print(f"q_predict: {q_predict}")
            q_action = tf.argmax(q_predict, axis=1).numpy()[0]

        else:
            q_action = random.randint(0, action_size - 1)
        if (it % 100 == 0):
            print(f"Q action:{q_action}")
        next_state, reward, done, info= env.step(q_action)
        total_reward += reward

        dqn_network.add_memory(curr_state, q_action, reward, next_state, done)
        curr_state = next_state
        if (next_state[0] >= env.goal_position): print(f"We made it in Episode: {episode}")
        if (len(dqn_network.replay_buffer) > MIN_REPLAY_MEMORY_SIZE):
            dqn_network.train_network()
        it += 1
    if EPSILON <= MIN_EPSILON:
        EPSILON = EPSILON * DECAY_RATE
    result_dict['reward'].append(total_reward)
    result_dict['ep'].append(episode)
    print(f"Episode:{episode} Results:{total_reward}")

env.close()
