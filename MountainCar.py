import gym
from DQN import DQNAgent
import numpy as np
from collections import deque

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

env = gym.make('MountainCar-v0')

batch_size = 32

n_episodes = 1000
game_time = 199

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size , action_size)


for i in range(n_episodes):
	
	state = env.reset()
	state = np.reshape(state,[1,state_size])
	speed = deque(maxlen=20)
	for time in range(game_time):
		if i % 100 == 0:
			env.render()
		action = agent.act(state)
		next_state, _ , _, _ = env.step(action)
		last_speed = sum(speed)/len(speed) if len(speed) != 0 else 0
		curr_v = abs(next_state[1]*10)
		speed.append(curr_v)
		curr_speed = sum(speed)/len(speed)
		reward = abs(last_speed-curr_speed)
		next_state = np.reshape(next_state,[1,state_size])
		if(np.random.rand() < 0.3):
			agent.remember(state,action,reward,next_state,False)
		state = next_state

	print("{}/{}".format(i,n_episodes))

	if(len(agent.memory) > batch_size):
		agent.replay(batch_size)

	if i % 100 == 0:
		agent.save("weights_mountaincar_{}.hdf5".format(i))