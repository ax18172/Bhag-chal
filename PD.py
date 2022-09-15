import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras.backend as K
from collections import deque
import random
from bag_chal_main import GOAT_AI, TIGER_AI, TIGER, tiger_score_check, goat_score_check, \
 \
    max_number_of_goats_on_the_board, run_episode

state_size = np.zeros((3, 3))
action_size_goats = 90

class Agent():
    def __init__(self,board_dimension, action_size):
        self.board_dimension = board_dimension
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.005
        self.model = self.build_model()
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []



    def store_transition(self,state,action,reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)


    def build_model(self):
        model = Sequential()
        model.add(Dense(self.board_dimension**2, input_shape=(self.board_dimension**2,), activation="relu"))
        model.add(Dense(45, activation="relu"))
        model.add(Dense(90, activation="relu"))
        model.add(Dense(180, activation="relu"))
        model.add(Dense(self.action_size, activation="softmax"))
        def custom_loss(y_true,y_pred):
            edited_softmax_output = K.clip(y_pred,1e-8,-1e-8)
            log_probability = y_true*K.log(edited_softmax_output)
            return K.sum(-log_probability)
        model.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def choose_action(self,state):
       state = np.reshape(state,(1,self.board_dimension**2))
       probabilities = self.model.predict(state)
       return probabilities[0]

    def learn(self):
        G = np.zeros((len(self.reward_memory),1))
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t,len(self.reward_memory)):
                G_sum += self.reward_memory[k]*discount
                discount *= self.gamma
            G[t] = G_sum
        for index,action in enumerate(self.action_memory):
                action_index= np.where(action == 1)
                action[action_index] = G[index]
        self.action_memory = np.array(self.action_memory)
        self.model.fit(self.state_memory,self.action_memory[0])
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def load(self, name):
            self.model.load_weights(name)

    def save(self, name):
            self.model.save_weights(name)

agent = Agent(3,90)
number_of_episodes = 10000
for episode in range(number_of_episodes):
    run_episode(3,7,(2,2),40,agent)
    agent.learn()

        


