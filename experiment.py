import gym
import random
import math
import numpy as np
import keras.models as models
import keras.layers as layers
import keras.optimizers as opt
import csv

GAME_NAME = 'LunarLander-v2'
SAMPLE_BATCH_SIZE = 64
EPISODE_NUM = 500
MODEL_NAME = "LunerLander-v2_DQN-FT.h5"

class Game:
	def __init__(self, game):
		self.game = game
		self.env = gym.make(game)
		self.step = 1
		self.episode_id = 0
		self.tot_reward = [0 for i in range(EPISODE_NUM)]
		self.epi_length = [0 for i in range(EPISODE_NUM)]

	def playEpisode(self, agent):
		observation = self.env.reset()
		while True:
			# self.env.render()
			
			action = agent.selectAct(observation)
			
			observation_s_next, reward, done, info = self.env.step(action)
			observation = observation_s_next
			self.tot_reward[self.episode_id] += reward

			if done:
				print ("total reward:", self.tot_reward[self.episode_id])
				self.epi_length[self.episode_id] = self.step
				self.episode_id += 1
				break

			self.step += 1

class Agent:
	def __init__(self, observation_dim, action_num):
		self.observation_dim = observation_dim
		self.action_num = action_num

		self.experience = Experience(observation_dim, action_num)

	def selectAct(self, observation):
		# seed = random.uniform(0, 1)
		# if seed <= self.epsilon:
		self.action = np.argmax(self.experience.getQ(observation, 'single_mode'))
		return self.action

class Experience:

	def __init__(self, observation_dim, action_num):
		self.observation_dim = observation_dim
		self.action_num = action_num

		self.q_nn = models.load_model(MODEL_NAME)

	def getQ(self, observation, mode):
		if mode == 'batch_mode':
			q = self.q_nn.predict(observation)
		if mode == 'single_mode':
			obs = observation.reshape(1, self.observation_dim)
			q = self.q_nn.predict(obs)
			q = q.flatten()
		return q

def main():
    # env = gym.make('LunarLander-v2')
    # s = env.observation_space
    # a = env.action_space
    # print('observation_space:', s)
    # print('action_space:', a)

    game = Game(GAME_NAME)

    observation_dim = game.env.observation_space.shape[0]
    action_num = game.env.action_space.n

    agent = Agent(observation_dim, action_num)

    for i in range(EPISODE_NUM):
    	print(">>>>>>> START TO PLAY NEW EPISODE:", i)
    	game.playEpisode(agent)

    with open("Experiment_DQN-FT_rewards.csv", "w", newline="") as reward_csv:
    	wr = csv.writer(reward_csv, quoting=csv.QUOTE_ALL)
    	wr.writerow(game.tot_reward)
    with open("Experiment_DQN-FT_EpiLen.csv", "w", newline="") as epiLen_csv:
    	wr = csv.writer(epiLen_csv, quoting=csv.QUOTE_ALL)
    	wr.writerow(game.epi_length)

if __name__ == "__main__":
    main()