from collections import defaultdict
import gc
import gzip
import inspect
import os
import os.path
import sys
import time

#import cv2
import gym
import numpy as np
import pickle

import neural_network as nn
from neural_network import tf, tint
from replay_buffer import ReplayBufferV2, LambdaReturn
from envs import AtariEnv


import six
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=2.5e-4, help="Learning rate", type=float)
parser.add_argument("--run", default=0, help="run", type=int)
parser.add_argument("--mbsize", default=32, help="Minibatch size", type=int)
parser.add_argument("--buffer_size", default=500000, help="Replay buffer size",type=int)
parser.add_argument("--clone_interval", default=10000, type=int)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--Lambda", default=1e-4, type=float)
parser.add_argument("--opt", default='adam')
parser.add_argument("--env_name", default='ms_pacman')
parser.add_argument("--device", default='cuda', help="device")


def main():
  device = torch.device(ARGS.device)
  nn.set_device(device)
  results = {
      "episode": [],
      "measure": [],
      "parameters": [],
  }


  hps = {
      "opt": ARGS.opt,
      "env_name": ARGS.env_name,
      "lr": ARGS.learning_rate,
      "weight_decay": ARGS.weight_decay,
      "run": ARGS.run,
  }

  nhid = hps.get("nhid", 32)
  gamma = hps.get("gamma", 0.99)
  mbsize = ARGS.mbsize
  weight_decay = hps.get("weight_decay", 0)
  sample_near = hps.get("sample_near", "both")
  slice_size = hps.get("slice_size", 0)
  env_name = hps.get("env_name", "ms_pacman")

  clone_interval = ARGS.clone_interval
  reset_on_clone = hps.get("reset_on_clone", False)
  reset_opt_on_clone = hps.get("reset_opt_on_clone", False)
  max_clones = hps.get("max_clones", 2)
  replay_type = hps.get("replay_type", "normal")  # normal, prioritized
  final_epsilon = hps.get("final_epsilon", 0.05)
  num_exploration_steps = hps.get("num_exploration_steps", 500_000)
  Lambda = ARGS.Lambda

  lr = hps.get("lr", 1e-4)
  num_iterations = hps.get("num_iterations", 10_000_000)
  buffer_size = ARGS.buffer_size

  seed = hps.get("run", 0) + 1_642_559  # A large prime number
  hps["_seed"] = seed
  torch.manual_seed(seed)
  np.random.seed(seed)
  rng = np.random.RandomState(seed)

  env = AtariEnv(env_name)
  num_act = env.num_actions

  # Define model

  _Qarch, theta_q, Qf, _Qsemi = nn.build(
      nn.conv2d(4, nhid, 8, stride=4),  # Input is 84x84
      nn.conv2d(nhid, nhid * 2, 4, stride=2),
      nn.conv2d(nhid * 2, nhid * 2, 3),
      nn.flatten(),
      nn.hidden(nhid * 2 * 12 * 12, nhid * 16),
      nn.linear(nhid * 16, num_act),
  )

  def make_opt():
    if hps.get("opt", "sgd") == "sgd":
      return torch.optim.SGD(theta_q, lr, weight_decay=weight_decay)
    elif hps["opt"] == "msgd":
      return torch.optim.SGD(
          theta_q,
          lr,
          momentum=hps.get("beta", 0.99),
          weight_decay=weight_decay)
    elif hps["opt"] == "rmsprop":
      return torch.optim.RMSprop(theta_q, lr, weight_decay=weight_decay)
    elif hps["opt"] == "adam":
      return torch.optim.Adam(theta_q, lr, weight_decay=weight_decay)
    else:
      raise ValueError(hps["opt"])

  opt = make_opt()
  clone_theta_q = lambda: [i.detach().clone() for i in theta_q]

  def copy_theta_q_to_target():
    for i in range(len(theta_q)):
      frozen_theta_q[i] = theta_q[i].detach().clone()

  # Define loss
  def sl1(a, b):
    d = a - b
    u = abs(d)
    s = d**2
    m = (u < s).float()
    return u * m + s * (1 - m)

  td = lambda x: sl1(
      x.r + (1 - x.t.float()) * gamma * Qf(x.sp, past_theta[0]).max(1)[0].detach(),
      Qf(x.s, theta_q)[np.arange(len(x.a)), x.a.long()],
  )

  tdQL = lambda x: sl1(
      Qf(x.s, theta_q)[np.arange(len(x.a)), x.a.long()],
      x.lg)

  mc = lambda x: sl1(
      Qf(x.s, theta_q).max(1)[0], x.g)

  past_theta = [clone_theta_q()]

  replay_buffer = ReplayBufferV2(seed, buffer_size,
                                 lambda s: Qf(s, theta_q),
                                 lambda s: Qf(s, past_theta[0]).max(1)[0],
                                 Lambda, gamma)

  total_reward = 0
  last_end = 0
  num_fill = buffer_size // 2
  num_measure = 500
  _t0 = t0 = t1 = t2 = t3 = t4 = time.time()
  tm0 = tm1 = tm2 = tm3 = time.time()
  ema_loss = 0
  last_rewards = [0]

  measure = Measures(
      theta_q, {
          "td": td,
          "tdQL": tdQL,
          "mc": mc,
      }, replay_buffer, results["measure"], 32)

  obs = env.reset()
  for it in range(num_fill):
    action = rng.randint(0, num_act)
    obsp, r, done, info = env.step(action)
    replay_buffer.add(obs, action, r, done)

    obs = obsp
    if done:
      print(it)
      obs = env.reset()


  for it in range(num_iterations):
    do_measure = not it % num_measure
    eta = (time.time() - _t0) / (it + 1) * (num_iterations - it) / 60
    if it and it % 100_000 == 0 or it == num_iterations - 1:
      ps = {str(i): p.data.cpu().numpy() for i, p in enumerate(theta_q)}
      ps.update({"step": it})
      results["parameters"].append(ps)

    if it < num_exploration_steps:
      epsilon = 1 - (it / num_exploration_steps) * (1 - final_epsilon)
    else:
      epsilon = final_epsilon

    if rng.uniform(0, 1) < epsilon:
      action = rng.randint(0, num_act)
    else:
      action = Qf(tf(obs / 255.0).unsqueeze(0)).argmax().item()

    obsp, r, done, info = env.step(action)
    total_reward += r
    replay_buffer.add(obs, action, r, done)

    obs = obsp
    if done:
      obs = env.reset()
      results["episode"].append({
          "end": it,
          "start": last_end,
          "total_reward": total_reward
      })
      last_end = it
      last_rewards = [total_reward] + last_rewards[:10]
      total_reward = 0

    sample = replay_buffer.sample(mbsize)
    with torch.no_grad():
      v_before = Qf(sample.s, theta_q).detach()

    loss = tdQL(sample)

    if do_measure:
      tm0 = time.time()
      measure.pre(sample)
      tm1 = time.time()
    loss = loss.mean()
    loss.backward()
    opt.step()
    opt.zero_grad()

    with torch.no_grad():
      v_after = Qf(sample.s, theta_q).detach()
    replay_buffer.compute_value_difference(sample, v_before, v_after)

    if do_measure:
      tm2 = time.time()
      measure.post()
      tm3 = time.time()
    t4 = time.time()
    if it and clone_interval and it % clone_interval == 0:
      past_theta = [clone_theta_q()] #+ past_theta[:max_clones - 1]
      replay_buffer.recompute_lambda_returns()

    #exp_results["loss"].append(loss.item())
    ema_loss = 0.999 * ema_loss + 0.001 * loss.item()
  with open(f'results/lambda_{run}.pkl', 'wb') as f:
    pickle.dump(results, f)


class Measures:

  def __init__(self, params, losses, replay_buffer, results, mbsize):
    self.p = params
    self.losses = losses
    self.rb = replay_buffer
    self.mbsize = mbsize
    self.rs = results

  def pre(self, sample):
    near_s, self.near_pmask = self.rb.slice_near(sample, 30)
    self._samples = {
        "sample": sample,
        "other": self.rb.sample(self.mbsize),
        "near": near_s,
    }
    self._cache = {}
    for loss_name, loss in self.losses.items():
      for item_name, item in self._samples.items():
        with torch.no_grad():
          self._cache[f'{item_name}_{loss_name}_pre'] = loss(item)

  def post(self):
    r = {
        "vdiff_acc": self.rb.vdiff_acc + 0,
        "vdiff_cnt": self.rb.vdiff_cnt + 0,
        'near_pmask': self.near_pmask.data.cpu().numpy(),
    }
    self.rb.vdiff_acc *= 0
    self.rb.vdiff_cnt *= 0
    for loss_name, loss in self.losses.items():
      for item_name, item in self._samples.items():
        k = f'{item_name}_{loss_name}'
        with torch.no_grad():
          self._cache[f'{k}_post'] = (loss(item))
          r[f'{k}_gain'] = (self._cache[f'{k}_pre'] -
                            self._cache[f'{k}_post']).cpu().data.numpy()
        r[k] = self._cache[f'{k}_post'].cpu().data.numpy()
    self.rs.append(r)



if __name__ == "__main__":
  ARGS = parser.parse_args()
  main()
