# Mountain Car using Deep Q-Network (DQN)

## Overview

This project trains a Deep Q-Network (DQN) to solve the **MountainCar-v0** environment from OpenAI Gym, aiming to drive the car to the goal using reinforcement learning.

## Files

```
DQN_Networks.py  # DQN model implementation
DQN_Agent_Train.py          # Training and environment interaction
README.md        # Project documentation
```

## Setup

Install dependencies:

```sh
pip install numpy tensorflow keras gym
```

Run training:

```sh
python DQN_Agent_Train.py 
```

## DQN Model

- Feedforward neural network with:
  - Input layer (state size)
  - Hidden layer (8 neurons, ReLU activation)
  - Output layer (action size)
- Uses a **target network** for stable training.

## Training

1. **Epsilon-greedy policy** for action selection.
2. Stores experiences in a **replay buffer**.
3. Trains Q-network once the buffer is sufficiently filled.
4. Updates target network periodically.

## Hyperparameters

- **Episodes:** 2000
- **Gamma:** 0.95
- **Epsilon Decay:** 0.995
- **Min Epsilon:** 0.1
- **Replay Buffer:** 1000

## Expected Output

Logs Q-values and rewards. Success message:

```sh
We made it in Episode: X
```

## Improvements

- More hidden layers
- Prioritized experience replay
- Double DQN

## References

- [OpenAI Gym](https://www.gymlibrary.dev/)
- [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

