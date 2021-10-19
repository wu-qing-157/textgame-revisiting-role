import numpy as np
from logger import log
import pickle

# state = [0]

class SimpleEnv:
	def __init__(self, envs):
		self.envs = envs
		self._dones = [False] * len(envs)
		self.num_envs = len(envs)
	def step(self, actions):
		results = []
		for i, (done, env, action) in enumerate(zip(self._dones, self.envs, actions)):
			log(f'{i} Step {action}')
			# with open('state.pickle', 'wb') as f:
				# state[0] = env.env.get_state()
				# pickle.dump(env.env.get_state(), f)
			if done:
				ob, info = env.reset()
				reward = 0
				self._dones[i] = False
			else:
				# print(action)
				ob, reward, self._dones[i], info = env.step(action)
			results.append((ob, reward, self._dones[i], info))
		obs, rewards, dones, infos = zip(*results)
		return np.stack(obs), np.stack(rewards), np.stack(dones), infos
	def reset(self):
		results = [env.reset() for env in self.envs]
		obs, infos = zip(*results)
		return np.stack(obs), infos
	def get_end_scores(self):
		results = [env.get_end_scores(last=100) for env in self.envs]
		return np.stack(results)
