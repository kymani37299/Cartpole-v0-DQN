import gym
from DQN import DQNAgent
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

env = gym.make('CartPole-v1')

batch_size = 32

n_episodes = 500

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size , action_size)
done = False

for i in range(n_episodes):
	state = env.reset()
	state = np.reshape(state,[1,state_size])
	for time in range(5000):
		if(i % 50 == 0):
			env.render()
		action = agent.act(state)
		next_state,reward,done, _ = env.step(action)
		reward = reward if not done else -10
		next_state = np.reshape(next_state,[1,state_size])
		if(np.random.rand() < 0.3):
			agent.remember(state,action,reward,next_state,done)
		state = next_state
		if done:
			print("episode: {}/{} , score: {}".format(i,n_episodes,time))
			break
	if(len(agent.memory) > batch_size):
		agent.replay(batch_size)

	if i % 100 == 0:
		agent.save("weights_cartpole_{}.hdf5".format(i))