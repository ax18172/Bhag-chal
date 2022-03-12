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
from bag_chal_main import GOAT_AI, TIGER_AI, TIGER, tiger_score_check, goat_score_check, memory, \
 \
    max_number_of_goats_on_the_board, goat_move

state_size = np.zeros((3, 3))
action_size_tiger = 16
batch_size = 32


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.5
        self.epsilon = 0.8
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        self.state_size = np.reshape(self.state_size, (1, 9))
        model = Sequential()
        model.add(Dense(9, input_shape=(self.state_size.shape[1],), activation="relu"))
        model.add(Dense(18, activation="relu"))
        model.add(Dense(9, activation="linear"))
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
            if not done:
                # print(next_state)
                prediction_for_next_state = self.model.predict(np.reshape(next_state, (1, 9)))[0]
                target_q_value = reward + self.gamma * np.amax(prediction_for_next_state)
            else:
                target_q_value = reward
            q_value_nn_prediction = self.model.predict(np.reshape(state, (1, 9)))
            np.reshape(q_value_nn_prediction, (3, 3))
            q_value_nn_prediction[0][action[0], action[1]] = target_q_value
            self.model.fit(np.reshape(state, (1, 9)), q_value_nn_prediction, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


agent = DQNAgent(state_size, action_size_tiger)
number_of_simulations = 10000

for simulation in range(number_of_simulations):
    board_dimension = 3
    board = np.zeros((board_dimension, board_dimension))
    goat_coord = []
    goats = []
    maximum_number_of_timesteps = 30
    eaten_goats = 0
    tiger = TIGER(2, 2, board)
    tiger_ai = TIGER_AI(tiger)
    goat_ai = GOAT_AI(max_number_of_goats_on_the_board)
    avialable_goats = max_number_of_goats_on_the_board
    decision = agent.act_decision()
    for timestep in range(maximum_number_of_timesteps):
        current_state = np.zeros((board_dimension, board_dimension))
        action = (0, 0)
        next_state = np.zeros((board_dimension, board_dimension))
        if avialable_goats > 0:
            board, goats, goat_coord = goat_move(board, goat_ai, goats, "placing", goat_coord)
            avialable_goats -= 1
        else:
            board, goats, goat_coord = goat_move(board, goat_ai, goats, "moving", goat_coord)
        print("goat is placed\n", board)
        done, tiger_reward = goat_score_check(tiger_ai, board, goat_coord)
        if not done:
            eaten_goats = eaten_goats
            if decision == "random":
                current_state, next_state, goat_coord, goats, action = tiger_ai.make_a_move([], board, goat_coord,
                                                                                            goats)
                print("random tiger moved\n", board)
            elif decision == "neural network":
                current_state = board.copy()
                q_values = agent.model.predict(np.reshape(current_state, (1, 9)))
                current_state, next_state, goat_coord, goats, action = tiger_ai.make_a_move(q_values, board, goat_coord,
                                                                                            goats)
                print("neural tiger moved\n", board, "q\n", q_values)
            done, tiger_reward, eaten_goats = tiger_score_check(tiger_ai, eaten_goats)
            memory.append((current_state, action, tiger_reward, next_state, done))
        else:
            break
        if memory[-1][-1]:
            break
