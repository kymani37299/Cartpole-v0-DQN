from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random
import numpy as np

class DQNAgent:

	def __init__(self,input_size,output_size):
		self.no_inputs = input_size
		self.no_outputs = output_size

		self.memory = deque(maxlen=2000)

		self.gamma = 0.9  # Discount rate

		self.epsilon = 1.0 # Exploration rate
		self.epsilon_decay = 0.995
		self.epsilon_min = 0.01

		self.alpha = 0.01 # Learning rate
		self.model = self._build_model()

	def _build_model(self):
		nn_height = 24

		model = Sequential()
		model.add(Dense(nn_height,input_dim = self.no_inputs,activation='relu'))
		model.add(Dense(nn_height,activation='relu'))
		model.add(Dense(self.no_outputs,activation='linear'))

		model.compile(loss='mse',optimizer=Adam(lr=self.alpha))

		return model

	def remember(self,state,action,reward,next_state,done):
		self.memory.append((state,action,reward,next_state,done))

	def act(self,state):
		if np.random.rand() < self.epsilon:
			return random.randrange(self.no_outputs)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])

	def replay(self,batch_size):
		minibatch = random.sample(self.memory,batch_size)
		for state , action , reward, next_state, done in minibatch:
			target = reward	
			if not done:
				target = (reward + self.gamma* np.amax(self.model.predict(next_state)[0]))
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state,target_f,epochs=1,verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load(self,name):
		self.model.load_weights(name)

	def save(self,name):
		self.model.save_weights(name)