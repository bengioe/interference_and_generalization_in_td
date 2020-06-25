import time
import sys
import torch
import numpy as np
import copy
import pickle
import gzip
import hashlib
import os.path
from tqdm import tqdm
from backpack import backpack, extend
from backpack.extensions import BatchGrad

if '--cpu' in sys.argv:
    device = torch.device('cpu')
else:
    device = torch.device('cuda')



class WindowEnv:
    def __init__(self, X, Y, window=8):
        self.X = X
        self.Y = Y
        self.window = window
        self.max_steps = 20
        self.step_reward = 0.1

    def reset(self, seed):
        self.current_seed = seed
        self.current_y = self.Y[seed]
        self.steps_taken = 0
        self.visible_image = torch.zeros((4, 32, 32), device=device)
        self.image = self.X[seed]
        self.agent_pos = [16, 16]
        self._draw()
        return self.visible_image

    def _draw(self):
        u,v = self.agent_pos
        w = self.window
        t,l,b,r = max(u-w//2,0),max(v-w//2,0) , min(u+w//2,32), min(v+w//2,32)
        self.visible_image[1:, t:b, l:r] = self.image[:, t:b, l:r]
        self.visible_image[0] = 0
        self.visible_image[0, t:b, l:r] = 1

    def step(self, action):
        cls, mov = action
        reward = 1 if cls == self.current_y else -self.step_reward
        done = 1 if cls == self.current_y else 0
        if mov == 0:
            self.agent_pos[0] = max(self.agent_pos[0] - self.window, 0)
        if mov == 1:
            self.agent_pos[0] = min(self.agent_pos[0] + self.window, 31)
        if mov == 2:
            self.agent_pos[1] = max(self.agent_pos[1] - self.window, 0)
        if mov == 3:
            self.agent_pos[1] = min(self.agent_pos[1] + self.window, 31)
        self._draw()
        self.steps_taken += 1
        return self.visible_image, reward, done or self.steps_taken >= self.max_steps, 0


class WindowEnvBatch:
    def __init__(self, window=8):
        self.window = window
        self.max_steps = 20
        self.step_reward = 0.1

    def reset(self, X, Y):
        n = len(X)
        self.current_y = Y.data.cpu().numpy()
        self.steps_taken = 0
        self.visible_image = torch.zeros((X.shape[0], 4, 32,32), device=device)
        self.image = X
        self.agent_pos = np.ones((n, 2), dtype='int32') * 16
        self._draw()
        self.dones = np.zeros((n,))
        self.correct_answers = 0
        self.acc_reward = 0
        return self.visible_image + 0

    def _draw(self):
        w = self.window
        t0 = time.time()
        for i,(u,v) in enumerate(self.agent_pos):
            t,l,b,r = max(u-w//2,0),max(v-w//2,0) , min(u+w//2,32), min(v+w//2,32)
            self.visible_image[i,1:, t:b, l:r] = self.image[i,:, t:b, l:r]
            self.visible_image[i,0] = 0
            self.visible_image[i,0, t:b, l:r] = 1
        t1 = time.time()

    def step(self, action):
        c = action[0] == self.current_y
        r = c - (1-c)*self.step_reward
        self.agent_pos[:, 0] -= (action[1] == 0) * self.window
        self.agent_pos[:, 0] += (action[1] == 1) * self.window
        self.agent_pos[:, 1] -= (action[1] == 2) * self.window
        self.agent_pos[:, 1] += (action[1] == 3) * self.window
        self.agent_pos = np.clip(self.agent_pos, 0, 31)
        self._draw()
        self.steps_taken += 1
        self.correct_answers += ((1-self.dones) * c).sum()
        self.acc_reward += ((1-self.dones) * r).sum() / len(c)
        self.dones = self.dones + (1-self.dones) * c
        if self.steps_taken > self.max_steps:
            self.dones[:] = 1
        return self.visible_image + 0, r, self.dones, 0




