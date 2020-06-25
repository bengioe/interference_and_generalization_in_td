import gym
import numpy as np
import cv2

class AtariEnv:
    def __init__(self, env='ms_pacman'):
        env = ''.join(i.capitalize() for i in env.split('_'))
        self.env = gym.make(f"{env}NoFrameskip-v4")
        self._obs_buffer = np.zeros((2, 84, 84), dtype=np.uint8)
        self.last = []
        self.env_name = env
        self.enumber = 0
        self.num_actions = self.env.action_space.n
        self.last_action = None

    def process(self, x):
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(x, (84, 84), interpolation=cv2.INTER_LINEAR)
        return np.uint8(x)

    def getRAM(self):
        return self.last_ram

    def reset(self):
        self.last = np.uint8(
            [self.process(self.env.reset())] * 4)
        self.enumber += 1
        self.last_action = None
        return self.last

    def step(self, a):
        tr = 0
        true_reward = 0
        for i in range(4):  # Frameskip of 4
            if np.random.uniform(0,1) < 0.25 and self.last_action is not None:
              # Sticky actions
              taken_action = self.last_action
            else:
              taken_action = a
            s, r, d, info = self.env.step(taken_action)
            self.last_action = taken_action
            tr += np.sign(r)
            true_reward += r
            self._obs_buffer[i % 2] = self.process(s)
            if d:
                break
        #self.last_ram = self.env._get_ram()
        self.last[0:3] = self.last[1:4]  # Feed last 4 "frames"
        # max of last 2 frames
        self.last[3] = self._obs_buffer.max(0)
        return self.last, tr, d, true_reward
