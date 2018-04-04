import gym
import random
import math
import numpy as np
import keras.models as models
import keras.layers as layers
import keras.optimizers as opt
import csv

EPSILON_MAX = 1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.000025

GAMMA = 0.99

EXP_CAP = 500000
GAME_NAME = 'LunarLander-v2'

SAMPLE_BATCH_SIZE = 64
LEARNING_RATE = 0.00025
EPISODE_NUM = 1500
TARGETQ_UPDATE_RATE = 600

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
			if done:
				observation_s_next = None
			memo = (observation, action, reward, observation_s_next)
			
			agent.update(memo, self.step)

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
		self.epsilon = EPSILON_MAX
		self.observation_dim = observation_dim
		self.action_num = action_num

		self.experience = Experience(observation_dim, action_num)

	def selectAct(self, observation):
		# seed = random.uniform(0, 1)
		# if seed <= self.epsilon:
		if random.random() < self.epsilon:
			self.action = random.choice(range(self.action_num))
		else:
			self.action = np.argmax(self.experience.getQ(observation, 'single_mode'))
		return self.action

	def update(self, memo, step):
		# cache newly experienced rewards and observations
		self.experience.memory.append(memo)
		if len(self.experience.memory) > self.experience.cap:
			self.experience.memory.pop(0)

		# decrease exploration probability
		self.epsilon = EPSILON_MIN + (EPSILON_MAX-EPSILON_MIN) * math.exp(-EPSILON_DECAY * step)
		# if step % 10 == 0:
		# 	print (self.epsilon)
		
		## update the QNN (experience replay)
		# random sample a batch of past transitions for updating Q_nn
		sample_batch = self.experience.sample()
		# print("size of sample batch:", type(sample_batch),len(sample_batch))
		# print("size of sample batch:", type(sample_batch[0][0]), len(sample_batch[0][0]))
		obs_s_batch = np.array([batch[0] for batch in sample_batch])
		zero_obs = np.zeros(self.observation_dim)
		obs_s_next_batch = np.array([(zero_obs if batch[3] is None else \
							batch[3]) for batch in sample_batch])
		
		q_s_batch = self.experience.getQ(obs_s_batch, 'batch_mode')

		if step % TARGETQ_UPDATE_RATE == 0:
			if_update = True
		else:
			if_update = False
		q_s_next_batch = self.experience.getTargetQ(obs_s_next_batch, if_update)

		obs_batch = np.zeros((len(sample_batch), self.observation_dim))
		target_batch = np.zeros((len(sample_batch), self.action_num))
		for i, batch in enumerate(sample_batch):
			obs_s = batch[0]
			reward_s = batch[2]
			action_s = batch[1]
			obs_s_next = batch[3]
			q_s_next = q_s_next_batch[i]

			target = q_s_batch[i]
			if obs_s_next is None:
				target[action_s] = reward_s
			else:
				target[action_s] = reward_s + GAMMA*np.amax(q_s_next)

			obs_batch[i] = obs_s
			target_batch[i] = target

		self.experience.trainQ_nn(obs_batch, target_batch)

class Experience:
	memory = []

	def __init__(self, observation_dim, action_num):
		self.cap = EXP_CAP
		self.observation_dim = observation_dim
		self.action_num = action_num

		self.q_nn = self.initQ_nn()
		self.q_nn_target = self.initQ_nn()

	def sample(self):
		if len(self.memory) < SAMPLE_BATCH_SIZE:
			sample_batch = random.sample(self.memory, len(self.memory)) 
		else:
			sample_batch = random.sample(self.memory, SAMPLE_BATCH_SIZE)

		return sample_batch

	def initQ_nn(self):
		q_nn = models.Sequential()
		q_nn.add(layers.Dense(output_dim=64, activation='relu', \
			input_dim=self.observation_dim))
		q_nn.add(layers.Dense(output_dim=64, activation='relu'))
		q_nn.add(layers.Dense(output_dim=self.action_num, activation='relu'))
		opt_method = opt.RMSprop(lr=LEARNING_RATE)
		q_nn.compile(loss='mse', optimizer=opt_method)

		return q_nn

	def trainQ_nn(self, obs_batch, target_batch, epoch=1, verbose=0):
		self.q_nn.fit(obs_batch, target_batch, batch_size=SAMPLE_BATCH_SIZE, \
			nb_epoch=epoch, verbose=verbose)

	def getQ(self, observation, mode):
		if mode == 'batch_mode':
			q = self.q_nn.predict(observation)
		if mode == 'single_mode':
			obs = observation.reshape(1, self.observation_dim)
			q = self.q_nn.predict(obs)
			q = q.flatten()
		return q

	def getTargetQ(self, observation, if_update):
		if if_update:
			self.q_nn_target.set_weights(self.q_nn.get_weights())
			# self.q_nn_target = self.q_nn

		q = self.q_nn_target.predict(observation)
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
    	print("epsilon:", agent.epsilon)
    	game.playEpisode(agent)

    agent.experience.q_nn.save("LunerLander-v2_DQN-FT.h5")

    with open("LunerLander-v2_DQN-FT_Rewards.csv", "w", newline="") as reward_csv:
    	wr = csv.writer(reward_csv, quoting=csv.QUOTE_ALL)
    	wr.writerow(game.tot_reward)
    with open("LunerLander-v2_DQN-FT_EpiLen.csv", "w", newline="") as epiLen_csv:
    	wr = csv.writer(epiLen_csv, quoting=csv.QUOTE_ALL)
    	wr.writerow(game.epi_length)

if __name__ == "__main__":
    main()