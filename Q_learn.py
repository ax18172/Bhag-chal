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
from matplotlib import pyplot as plt
import io
import os
import collections
from collections import deque
from bag_chal_main import board
import random
import time
from bag_chal_main import GOAT_AI, TIGER_AI, TIGER, tiger_score_check, goat_score_check, memory, \
 \
    max_number_of_goats_on_the_board, goat_move

state_size = np.zeros((3, 3))
action_size_tiger = 16
batch_size = 32
Q_values = []
Maximum_Q_values = []
number_of_timesteps_to_win = []
number_of_timesteps_to_lose = []
times_per_simulation = []
start_point = 0


def memory_scan(start_point, batch_size):
    counter = 0
    result = 0
    game_iteration = 0
    for mem in range(start_point, start_point + batch_size, 1):
        if memory[mem][-2]:
            game_iteration = mem + 1
            game_has_ended = False
            while not game_has_ended:
                if game_iteration >= len(memory):
                    break
                result = memory[game_iteration][-2]
                if not result:
                    counter += 1
                    game_iteration += 1
                else:
                    if memory[game_iteration][2] == 1000:
                        number_of_timesteps_to_win.append(counter)
                    elif memory[game_iteration][2] == -1000:
                        number_of_timesteps_to_lose.append(counter)
                    counter = 0
                    game_has_ended = True
    return start_point + batch_size
    # return start_point + batch_size


def plotting_difference(list1, list2, plotting_differences, title, x_axis, y_axis, name):
    differences = []
    print(list1)
    if plotting_differences:
        for i in range(len(list1)):
            diff = list1[i] - list2[i]
            differences.append(diff)
    else:
        differences = list1
    plt.plot(differences)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.savefig(name)
    plt.show()

    return differences


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.6
        self.epsilon = 0.8
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.009
        self.tau = 0.125
        self.model = self.build_model()
        self.target_model = self.build_model()

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
        for state, action, reward, next_state, done, test_probability_matrix in minibatch:
            if not done:
                # print("reward: ", reward)
                prediction_for_next_state = self.target_model.predict(np.reshape(next_state, (1, 9)))[0]
                target_q_value = reward + self.gamma * np.amax(prediction_for_next_state)
            else:
                target_q_value = reward
            q_value_nn_prediction = self.target_model.predict(np.reshape(state, (1, 9)))
            q_value_nn_prediction = np.reshape(q_value_nn_prediction, (3, 3))
            q_value_nn_prediction[action[0], action[1]] = target_q_value
            q_value_nn_prediction = np.reshape(q_value_nn_prediction, (1, 9))
            self.model.fit(np.reshape(state, (1, 9)), q_value_nn_prediction, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)


agent = DQNAgent(state_size, action_size_tiger)
number_of_simulations = 705

for simulation in range(number_of_simulations):
    print("simulation number: ", simulation, "/",
          number_of_simulations)
    # print("memory:\n", memory)
    # print("Q_values test:\n", Q_values)
    # print("Maximum Q_values:\n", Maximum_Q_values)
    board_dimension = 3
    board = np.zeros((board_dimension, board_dimension))
    goat_coord = []
    goats = []
    maximum_number_of_timesteps = 20
    eaten_goats = 0
    tiger = TIGER(2, 2, board)
    tiger_ai = TIGER_AI(tiger)
    goat_ai = GOAT_AI(max_number_of_goats_on_the_board)
    avialable_goats = max_number_of_goats_on_the_board
    decision = agent.act_decision()
    times_per_simulation.append(time.process_time)
    if (simulation + 1) % 100 == 0:
        plotting_difference(Maximum_Q_values, Q_values, True,
                            'Graph showing difference between maximum Q-value \n and Q value of eating a goat',
                            'number of cases where the goat can be eaten', 'difference between Q-values',
                            'line_plot.pdf')
        plotting_difference(number_of_timesteps_to_win, None, False, "steps it takes for the tiger to win",
                            "iterations", "number of moves", 'win_steps.pdf')
        plotting_difference(number_of_timesteps_to_lose, None, False, "steps it takes for the tiger to lose",
                            "iterations", "number of moves", "lose_steps.pdf")
        #print('time=', sum(times_per_simulation))

    for timestep in range(maximum_number_of_timesteps):
        current_state = np.zeros((board_dimension, board_dimension))
        action = (0, 0)
        next_state = np.zeros((board_dimension, board_dimension))
        if avialable_goats > 0:
            board, goats, goat_coord = goat_move(board, goat_ai, goats, "placing", goat_coord)
            avialable_goats -= 1
        else:
            board, goats, goat_coord = goat_move(board, goat_ai, goats, "moving", goat_coord)
        # print("goat is placed\n", board)
        done, tiger_reward = goat_score_check(tiger_ai, board, goat_coord)
        if not done:
            eaten_goats = eaten_goats
            if decision == "random":
                current_state, next_state, goat_coord, goats, action, test_probability_matrix = tiger_ai.make_a_move([],
                                                                                                                     board,
                                                                                                                     goat_coord,
                                                                                                                     goats)
                # print("random tiger moved\n", board)
            elif decision == "neural network":
                current_state = board.copy()
                q_values = agent.target_model.predict(np.reshape(current_state, (1, 9)))
                # print("Q-values:\n", q_values)
                current_state, next_state, goat_coord, goats, action, test_probability_matrix = tiger_ai.make_a_move(
                    q_values, board, goat_coord,
                    goats)
                # print("neural tiger moved\n", board)
                if np.amax(test_probability_matrix) > 1:
                    test_probability_matrix = np.reshape(test_probability_matrix, (1, 9))
                    for probability in test_probability_matrix[0]:
                        if probability > 1:
                            index = np.where(test_probability_matrix == probability)
                            # q_values_array = q_values.copy()
                            # print('array',q_values_array)
                            # print('list',q_values_list)
                            # print("index ", index[1][0])
                            # print("q values ", q_values)
                            q_value = q_values[0, index[1][0]]
                            max_q_value = np.amax(q_values)
                            Q_values.append(q_value)
                            Maximum_Q_values.append(max_q_value)
                            break
            done, tiger_reward, eaten_goats = tiger_score_check(tiger_ai, eaten_goats)
            memory.append((current_state, action, tiger_reward, next_state, done, test_probability_matrix))

        else:
            break
        if memory[-1][-2]:
            break
        if len(memory) > batch_size:
            agent.replay(batch_size)
            if (len(memory) + 1) % batch_size == 0:
                agent.target_train()
                start_point = memory_scan(start_point, 15)
                agent.save("trial-{}.model".format(len(memory)))
                print("won", number_of_timesteps_to_win)
                print("lost", number_of_timesteps_to_lose)
#print('time = ',sum(times_per_simulation))