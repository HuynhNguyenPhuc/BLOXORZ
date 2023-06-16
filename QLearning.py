from GameWorld import *
from copy import deepcopy
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.metrics import Recall, Precision


NUM_TESTCASE = '1' # Choose testcase to run

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
            return (self.location[0][0], self.location[0][1], self.location[1][0], self.location[1][1]) + tuple(self.board.values())

    def __init__(self, cube, board):
        self.board = board.map
        self.firstCube = cube.firstCube
        self.secondCube = cube.secondCube
        self.button = board.buttonList
        self.boardTemp = deepcopy(self.board)
        self.res = []
        
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        
        for i in self.button.values():
            if (i[0] != 3):
                for j in i[1]:
                    value = self.boardTemp.get(j, 0)
                    if value == 0:
                        self.boardTemp[j] = 0
        self.model = self.__model()
        try:
            self.first = False
            self.weights_model = np.load("model/model_"+NUM_TESTCASE+".npy")
            self.decode(self.weights_model)
        except:
            self.first = True
            pass

    def __model(self):
        model = Sequential()
        model.add(Input(shape = (4 + len(self.boardTemp.values()), )))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(4, activation='linear'))
        model.compile(loss = "binary_crossentropy", optimizer = Adam(learning_rate=0.001), metrics = ["accuracy", Recall(), Precision()])
        return model

    def encode(self):
        weights = []
        for layer in self.model.layers:
            layer_weight = layer.get_weights()
            weights = np.append(weights,layer_weight[0])
            weights = np.append(weights,layer_weight[1])
        return np.array(weights,dtype=np.float64).flatten()

    def decode(self, input):
        weights = []
        chromosome = np.copy(input)
        for layer in self.model.layers:
            layer_weights = layer.get_weights()
            weights_shape = layer_weights[0].shape
            bias_shape = layer_weights[1].shape
            weights_len = len(layer_weights[0].flatten())
            bias_len = len(layer_weights[1].flatten())
            new_weights = np.reshape(chromosome[:weights_len],weights_shape)
            new_bias = np.reshape(chromosome[weights_len:weights_len+bias_len],
                                  bias_shape)
            chromosome = chromosome[weights_len+bias_len:]
            weights.append(new_weights)
            weights.append(new_bias)
        self.model.set_weights(np.array(weights,dtype=object))

    

    def act(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            return random.randint(0, len(state.nextStateList()) - 1)
        return np.argmax(self.model.predict(np.array([state.hash()])))

    def learn(self, num_episodes=1):
        for episode in range(num_episodes):
            state = self.State(deepcopy(self.boardTemp), (self.firstCube, self.secondCube), self.button)
            done = False
            while not done:
                action = self.act(state)
                next_state = state.move(action)
                if next_state is None:
                    next_state = state
                if next_state.isGoalState():
                    reward = 100
                    done = True
                else:
                    reward = -1
                if self.first:
                    target_f = np.array([(0,0,0,0)])
                    np.put(target_f[0], action, reward)
                    self.first = False
                else:
                    target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state.hash()])))
                    target_f = self.model.predict(np.array([state.hash()]))
                    np.put(target_f[0], action, target)
                self.model.fit(np.array([state.hash()]), target_f, epochs=1, verbose=0)
                state = next_state
                if done:
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.weights_model = self.encode()
            np.save("model/model_"+NUM_TESTCASE+".npy", self.weights_model)
            print("Episode " + str(episode) + " end!")
        print("Training completed!")

    def solve(self):
        # self.learn()
        state = self.State(deepcopy(self.boardTemp), (self.firstCube, self.secondCube), self.button)
        done = False
        loop = 0
        self.res.append([self.firstCube, self.secondCube])
        while (not done) and loop < 50:
            q_values = self.model.predict(np.array([state.hash()]))
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
