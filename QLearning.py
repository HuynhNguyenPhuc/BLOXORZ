from GameWorld import *

from copy import deepcopy
import random
import numpy as np
from collections import deque
from tqdm import tqdm

from Model import LSTM_Model
from keras.utils import pad_sequences
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))

def manhattanDistance(item1, item2):
    return abs(item1[0] - item2[0]) + abs(item1[1] - item2[1])


def checkValid(board, key):
    try:
        test = board[key]
        if test == 0:
            return False
        return True
    except:
        return False


class QLearning:
    class State:
        def __init__(self, board, location, button, parent=None, length=0):
            self.board = board
            self.location = location
            self.button = button
            self.parent = parent
            self.length = length

        def checkValidLocation(self):
            if checkValid(self.board, self.location[0]) and checkValid(
                self.board, self.location[1]
            ):
                return True
            return False

        def nextStateList(self):
            location = self.location
            x1 = location[0][0]
            y1 = location[0][1]
            x2 = location[1][0]
            y2 = location[1][1]
            if location[0] == location[1]:
                listLocation = [
                    ((x1, y1 - 2), (x1, y1 - 1)),
                    ((x1 + 1, y1), (x1 + 2, y1)),
                    ((x1, y1 + 1), (x1, y1 + 2)),
                    ((x1 - 2, y1), (x1 - 1, y1)),
                ]
            elif manhattanDistance(location[0], location[1]) == 1:
                if x1 == x2:
                    if y1 > y2:
                        y1, y2 = y2, y1
                    listLocation = [
                        ((x1, y1 - 1), (x1, y1 - 1)),
                        ((x1 + 1, y1), (x1 + 1, y2)),
                        ((x1, y2 + 1), (x1, y2 + 1)),
                        ((x1 - 1, y1), (x1 - 1, y2)),
                    ]
                else:
                    if x1 > x2:
                        x1, x2 = x2, x1
                    listLocation = [
                        ((x1, y1 - 1), (x2, y1 - 1)),
                        ((x2 + 1, y1), (x2 + 1, y1)),
                        ((x1, y1 + 1), (x2, y1 + 1)),
                        ((x1 - 1, y1), (x1 - 1, y1)),
                    ]
            else:
                listLocation = [
                    ((x1 + 1, y1), (x2, y2)),
                    ((x1, y1 + 1), (x2, y2)),
                    ((x1 - 1, y1), (x2, y2)),
                    ((x1, y1 - 1), (x2, y2)),
                    ((x1, y1), (x2 + 1, y2)),
                    ((x1, y1), (x2, y2 + 1)),
                    ((x1, y1), (x2 - 1, y2)),
                    ((x1, y1), (x2, y2 - 1)),
                ]
            return listLocation

        def __buttonTrigger(self, index):
            for i in self.button[index][1]:
                if checkValid(self.board, i):
                    self.board[i] = 0
                else:
                    self.board[i] = 2

        def __changeStatus(self, newLocation):
            newState = self.__class__(
                deepcopy(self.board),
                newLocation,
                self.button,
                self,
                int(self.length + 1),
            )
            if newLocation[0] == newLocation[1]:
                if (
                    checkValid(self.button, newLocation[0])
                    and self.button[newLocation[0]][0] != 3
                ):
                    newState.__buttonTrigger(newLocation[0])
            else:
                if (
                    checkValid(self.button, newLocation[0])
                    and self.button[newLocation[0]][0] == 1
                ):
                    if newLocation[0] != self.location[0]:
                        newState.__buttonTrigger(newLocation[0])
                if (
                    checkValid(self.button, newLocation[1])
                    and self.button[newLocation[1]][0] == 1
                ):
                    if newLocation[1] != self.location[1]:
                        newState.__buttonTrigger(newLocation[1])
            return newState

        def move(self, action):
            if action >= len(self.nextStateList()):
                return None
            newLocation = self.nextStateList()[action]
            newState = self.__changeStatus(newLocation)
            if newState.checkValidLocation():
                if newState.location[0] == newState.location[1]:
                    if (
                        checkValid(self.board, newState.location[0])
                        and self.board[newState.location[0]] == 1
                    ):
                        return None
                    if checkValid(self.button, newState.location[0]):
                        if self.button[newState.location[0]][0] == 3:
                            newState.location = tuple(
                                self.button[newState.location[0]][1]
                            )
                return newState
            return None

        def isGoalState(self):
            if self.board[self.location[0]] == 3 and self.board[self.location[1]] == 3:
                return True
            return False

        def hash(self):
            return pad_sequences([(self.location[0][0], self.location[0][1], self.location[1][0], self.location[1][1]) + tuple(self.board.values())], maxlen = 512, padding = 'post')

    def __init__(self, cube, board):
        self.board = board.map
        self.firstCube = cube.firstCube
        self.secondCube = cube.secondCube
        self.button = board.buttonList
        self.boardTemp = deepcopy(self.board)
        self.memory = deque(maxlen=10000)
        self.res = []
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.lr = 0.001
        self.reset_every = 50
        self.n_iter = 1
        
        for i in self.button.values():
            if (i[0] != 3):
                for j in i[1]:
                    value = self.boardTemp.get(j, 0)
                    if value == 0:
                        self.boardTemp[j] = 0
        self.model = self.__model()
        self.target_model = self.__model()
        self.target_model.set_weights(self.model.get_weights())
        try:
            self.model.load_weights("model.weights.h5")
            self.target_model.load_weights("model.weights.h5")
        except:
            pass

    def __model(self):
        model = LSTM_Model(512, 4, 256)
        model.compile(loss=huber_loss, optimizer=Adam(learning_rate=0.0025))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            return random.randint(0, len(state.nextStateList()) - 1)
        return np.argmax(self.model.predict(state.hash(), verbose = 0))
    
    def trainExperienceReplay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []
        
        if self.n_iter % self.reset_every == 0:
            self.target_model.set_weights(self.model.get_weights())

        for state, action, reward, next_state, done in mini_batch:
            if done or next_state is None:
                target = reward
            else:
                target = reward + self.gamma * self.target_model.predict(next_state.hash(), verbose = 0)[0][np.argmax(self.model.predict(next_state.hash(), verbose = 0)[0])]

            q_values = self.model.predict(state.hash(), verbose = 0)
            q_values[0][action] = target

            X_train.append(state.hash()[0])
            y_train.append(q_values[0])

        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1, verbose=0
        ).history["loss"][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def learn(self, num_episodes = 100, batch_size=32):
        avg_loss = []

        episode = 0
        with tf.device("/GPU:0"):
            for episode in range(num_episodes):
                state = self.State(deepcopy(self.boardTemp), (self.firstCube, self.secondCube), self.button)
                done = False
                while not done:
                    action = self.act(state)
                    next_state = state.move(action)
                    if next_state is None:
                        reward = -100
                    elif next_state.isGoalState():
                        reward = 100
                        done = True
                    else:
                        reward = 1

                    self.remember(state, action, reward, next_state, done)
                    if len(self.memory) > batch_size:
                        loss = self.trainExperienceReplay(batch_size)
                        avg_loss.append(loss)
                        self.n_iter += 1

                    if next_state is not None:
                        state = next_state

                self.model.save_weights("model.weights.h5")

    def solve(self):
        state = self.State(deepcopy(self.boardTemp), (self.firstCube, self.secondCube), self.button)
        done = False
        loop = 0
        self.res.append([self.firstCube, self.secondCube])
        while (not done) and loop < 100:
            q_values = self.model.predict(state.hash(), verbose = 0)
            action = np.argmax(q_values)
            next_state = state.move(action)
            while next_state is None:
                np.put(q_values[0], action, -100)
                action = np.argmax(q_values)
                next_state = state.move(action)
            self.res.append(list(next_state.location))
            state = next_state
            if state.isGoalState():
                done = True
            loop+=1
        return self.res