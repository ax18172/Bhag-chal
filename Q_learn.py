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
from bag_chal_main import run_environment, GOAT_AI, TIGER_AI, TIGER, GOAT, tiger_score_check, goat_score_check, memory, \
 \
    max_number_of_goats_on_the_board, grid_matrix, eaten_goats

state_size = np.zeros((3, 3))
action_size_tiger = 16
batch_size = 32
possible_moves = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, 2), (2, 0), (0, -2),
                  (-2, 0), (2, 2), (2, -2),
                  (-2, 2), (-2, -2)]


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
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
        model.add(Dense(16, activation="linear"))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act_decision(self):
        if random.random() <= self.epsilon:
            return "random"
        else:
            return "neural network"

    def replay(self, batch_size):
        minibatch = random.sample(memory, batch_size)
        # print(memory)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                prediction_for_next_state = self.model.predict(np.reshape(next_state, (1, 9)))[0]
                highest_q_value_bellman_prediction = reward + self.gamma * np.amax(prediction_for_next_state)
            else:
                highest_q_value_bellman_prediction = reward
            # print(action)
            action_index = possible_moves.index(action)
            q_value_nn_prediction = self.model.predict(np.reshape(state, (1, 9)))
            q_value_nn_prediction[0][action_index] = highest_q_value_bellman_prediction
            self.model.fit(np.reshape(state, (1, 9)), q_value_nn_prediction, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


agent = DQNAgent(state_size, action_size_tiger)
number_of_simulations = 100

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
        # print(board)
        if decision == "random":
            print("normal first", board)
            board, goat_coord, goats, tiger, eaten_goats, tiger_ai, goat_ai, avialable_goats = run_environment(board,
                                                                                                               tiger,
                                                                                                               goat_coord,
                                                                                                               goats,
                                                                                                               False,
                                                                                                               None,
                                                                                                               None,
                                                                                                               maximum_number_of_timesteps,
                                                                                                               timestep,
                                                                                                               board_dimension,
                                                                                                               eaten_goats,
                                                                                                               tiger_ai,
                                                                                                               goat_ai,
                                                                                                               avialable_goats)
            print("normal second", board)
        elif decision == "neural network":
            print("neural first", board)
            # print(board)
            current_state = board.copy()
            # print(current_state)
            q_values = agent.model.predict(np.reshape(current_state, (1, 9)))
            move_q_value = int(np.argmax(q_values))
            tiger_dx, tiger_dy = possible_moves[move_q_value]
            print(tiger_dx, tiger_dy)
            board, goat_coord, goats, tiger, eaten_goats, tiger_ai, goat_ai, avialable_goats = run_environment(board,
                                                                                                               tiger,
                                                                                                               goat_coord,
                                                                                                               goats,
                                                                                                               True,
                                                                                                               tiger_dx,
                                                                                                               tiger_dy,
                                                                                                               maximum_number_of_timesteps,
                                                                                                               timestep,
                                                                                                               board_dimension,
                                                                                                               eaten_goats,
                                                                                                               tiger_ai,
                                                                                                               goat_ai,
                                                                                                                                                                                                                avialable_goats)
            print("neural second", board)
        if memory[-1][-1]:
            break
        if len(memory) > batch_size:
            agent.replay(batch_size)
