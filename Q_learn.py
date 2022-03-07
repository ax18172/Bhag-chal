import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.models import load_model
import io
import os
import collections
from collections import deque
from bag_chal_main import board
import random
from bag_chal_main import run_environment, GOAT_AI, TIGER_AI, TIGER, GOAT, score__and_game_check, memory, board, \
    tiger_score, goat_score, max_number_of_goats_on_the_board, grid_matrix

state_size = np.zeros((3, 3))
action_state_tiger = np.zeros((1, 2))
batch_size = 32


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        self.state_size = np.reshape(self.state_size, (1, 9))
        print(self.state_size.shape)
        model = Sequential()
        model.add(Dense(9, input_shape=(self.state_size.shape[0], self.state_size.shape[1]), activation="relu"))
        model.add(Dense(18, activation="relu"))
        model.add(Dense(36, activation="relu"))
        model.add(Dense(18, activation="relu"))
        model.add(Dense(9, activation="relu"))
        model.add(Dense(2, activation="softmax"))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act_decision(self):
        if random.random() <= self.epsilon:
            return "random"
        else:
            return "neural network"

    def replay(self, batch_size):
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


agent = DQNAgent(state_size, action_state_tiger)
number_of_simulations = 1000

for simulation in range(number_of_simulations):
    board = np.zeros((3, 3))
    tiger_score = 0
    goat_score = 0
    tiger = TIGER(2, 2)
    for timestep in range(20):
        decision = agent.act_decision()
        if decision == "random":
            run_environment(1, False, None, None, tiger)
        elif decision == "neural network":
            state = board
            current_state = board.copy()
            tiger_dx, tiger_dy = agent.model.predict(current_state)
            run_environment(1, True, tiger_dx, tiger_dy, tiger)
    if len(memory) > batch_size:
        agent.replay(batch_size)
