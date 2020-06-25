from collections import defaultdict
import gc
import gzip
import inspect
import os
import os.path
import sys
import time

import gym
import numpy as np
import pickle

import .neural_network as nn
from neural_network import tf, tint
from replay_buffer import ReplayBuffer, PrioritizedExperienceReplay
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
parser.add_argument("--opt", default='adam')
parser.add_argument("--env_name", default='ms_pacman')
parser.add_argument("--device", default='cuda', help="device")
parser.add_argument("--checkpoint", default='results/41.pkl', help="checkpoint file",type=str)

def main():
  results = {
      "results": [],
      "measure_reg": [],
      "measure_td": [],
      "measure_mc": [],
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
  mbsize = hps.get("mbsize", 32)
  weight_decay = hps.get("weight_decay", 0)
  sample_near = hps.get("sample_near", "both")
  slice_size = hps.get("slice_size", 0)
  env_name = hps.get("env_name", "ms_pacman")

  clone_interval = hps.get("clone_interval", 10_000)
  reset_on_clone = hps.get("reset_on_clone", False)
  reset_opt_on_clone = hps.get("reset_opt_on_clone", False)
  max_clones = hps.get("max_clones", 2)
  target = hps.get("target", "last")  # self, last, clones
  replay_type = hps.get("replay_type", "normal")  # normal, prioritized
  final_epsilon = hps.get("final_epsilon", 0.05)
  num_exploration_steps = hps.get("num_exploration_steps", 500_000)

  lr = hps.get("lr", 1e-4)
  num_iterations = hps.get("num_iterations", 10_000_000)
  buffer_size = hps.get("buffer_size", 250_000)

  seed = hps.get("run", 0) + 1_642_559  # A large prime number
  hps["_seed"] = seed
  torch.manual_seed(seed)
  np.random.seed(seed)
  rng = np.random.RandomState(seed)

  env = AtariEnv(env_name)
  num_act = env.num_actions

  def make_opt(theta):
    if hps.get("opt", "sgd") == "sgd":
      return torch.optim.SGD(theta, lr, weight_decay=weight_decay)
    elif hps["opt"] == "msgd":
      return torch.optim.SGD(
          theta, lr, momentum=hps.get("beta", 0.99), weight_decay=weight_decay)
    elif hps["opt"] == "rmsprop":
      return torch.optim.RMSprop(theta, lr, weight_decay=weight_decay)
    elif hps["opt"] == "adam":
      return torch.optim.Adam(theta, lr, weight_decay=weight_decay)
    else:
      raise ValueError(hps["opt"])

  # Define model

  _Qarch, theta_q, Qf, _Qsemi = nn.build(
      nn.conv2d(4, nhid, 8, stride=4),  # Input is 84x84
      nn.conv2d(nhid, nhid * 2, 4, stride=2),
      nn.conv2d(nhid * 2, nhid * 2, 3),
      nn.flatten(),
      nn.hidden(nhid * 2 * 12 * 12, nhid * 16),
      nn.linear(nhid * 16, num_act),
  )
  clone_theta_q = lambda: [i.detach().clone().requires_grad_() for i in theta_q]

  # Pretrained parameters
  theta_target = load_parameters_from_checkpoint()
  # (Same) Random parameters
  theta_regress = clone_theta_q()
  theta_qlearn = clone_theta_q()
  theta_mc = clone_theta_q()
  opt_regress = make_opt(theta_regress)
  opt_qlearn = make_opt(theta_qlearn)
  opt_mc = make_opt(theta_mc)

  # Define loss
  def sl1(a, b):
    d = a - b
    u = abs(d)
    s = d**2
    m = (u < s).float()
    return u * m + s * (1 - m)

  td = lambda s, a, r, sp, t, w, tw=theta_q: sl1(
      r + (1 - t.float()) * gamma * Qf(sp, tw).max(1)[0].detach(),
      Qf(s, w)[np.arange(len(a)), a.long()],
  )

  obs = env.reset()

  replay_buffer = ReplayBuffer(seed, buffer_size, near_strategy=sample_near)

  total_reward = 0
  last_end = 0
  num_fill = buffer_size
  num_measure = 500
  _t0 = t0 = t1 = t2 = t3 = t4 = time.time()
  tm0 = tm1 = tm2 = tm3 = time.time()
  ema_loss = 0
  last_rewards = [0]

  print("Filling buffer")
  epsilon = final_epsilon

  replay_buffer.new_episode(obs, env.enumber % 2)
  while replay_buffer.idx < replay_buffer.size - 10:
    if rng.uniform(0, 1) < epsilon:
      action = rng.randint(0, num_act)
    else:
      action = Qf(tf(obs / 255.0).unsqueeze(0), theta_target).argmax().item()
    obsp, r, done, info = env.step(action)
    replay_buffer.add(obs, action, r, done, env.enumber % 2)

    obs = obsp
    if done:
      obs = env.reset()
      replay_buffer.new_episode(obs, env.enumber % 2)
  # Remove last episode from replay buffer, as it didn't end
  it = replay_buffer.idx
  curp = replay_buffer.p[it]
  while replay_buffer.p[it] == curp:
    replay_buffer._sumtree.set(it, 0)
    it -= 1
  print(f'went from {replay_buffer.idx} to {it} when deleting states')

  print("Computing returns")
  replay_buffer.compute_values(lambda s: Qf(s, theta_regress), num_act)
  replay_buffer.compute_returns(gamma)
  replay_buffer.compute_reward_distances()
  print("Training regressions")
  losses_reg, losses_td, losses_mc = [], [], []

  loss_reg_f = lambda x, w: sl1(Qf(x[0], w), Qf(x[0], theta_target))
  loss_td_f = lambda x, w: td(*x[:-1], w, theta_target)
  loss_mc_f = lambda x, w: sl1(
      Qf(x[0], w)[np.arange(len(x[1])), x[1].long()], replay_buffer.g[x[-1]])

  losses = {
      "reg": loss_reg_f,
      "td": loss_td_f,
      "mc": loss_mc_f,
  }

  measure_reg = Measures(theta_regress, losses, replay_buffer,
                         results["measure_reg"], mbsize)
  measure_mc = Measures(theta_mc, losses, replay_buffer,
                        results["measure_mc"], mbsize)
  measure_td = Measures(theta_qlearn, losses, replay_buffer,
                        results["measure_td"], mbsize)

  for i in range(100_000):
    sample = replay_buffer.sample(mbsize)
    replay_buffer.compute_value_difference(sample, Qf(sample[0], theta_regress))

    if i and not i % num_measure:
      measure_reg.pre(sample)
      measure_mc.pre(sample)
      measure_td.pre(sample)

    loss_reg = loss_reg_f(sample, theta_regress).mean()
    loss_reg.backward()
    losses_reg.append(loss_reg.item())
    opt_regress.step()
    opt_regress.zero_grad()

    loss_td = loss_td_f(sample, theta_qlearn).mean()
    loss_td.backward()
    losses_td.append(loss_td.item())
    opt_qlearn.step()
    opt_qlearn.zero_grad()

    loss_mc = loss_mc_f(sample, theta_mc).mean()
    loss_mc.backward()
    losses_mc.append(loss_mc.item())
    opt_mc.step()
    opt_mc.zero_grad()

    replay_buffer.update_values(sample, Qf(sample[0], theta_regress))
    if i and not i % num_measure:
      measure_reg.post()
      measure_td.post()
      measure_mc.post()

    if not i % 1000:
      print(i, loss_reg.item(), loss_td.item(), loss_mc.item())

  results["results"].append({
      "losses_reg": np.float32(losses_reg),
      "losses_td": np.float32(losses_td),
      "losses_mc": np.float32(losses_mc)
  })

  path = f'results/regress_{ARGS.run}.pkl'
  with open(path, "wb") as f:
    pickle.dump(results, f)
  print(f"Done in {(time.time()-_t0)/60:.2f}m")


class Measures:

  def __init__(self, params, losses, replay_buffer, results, mbsize):
    self.p = params
    self.losses = losses
    self.rb = replay_buffer
    self.mbsize = mbsize
    self.rs = results

  def pre(self, sample):
    self._sampleidx = sample[-1]
    near_s, self.near_pmask = self.rb.slice_near(self._sampleidx, 30)
    self._samples = {
        "sample": sample,
        "other": self.rb.sample(self.mbsize),
        "near": near_s,
    }
    self._cache = {}
    for loss_name, loss in self.losses.items():
      for item_name, item in self._samples.items():
        with torch.no_grad():
          self._cache[f'{item_name}_{loss_name}_pre'] = loss(item, self.p)

  def post(self):
    r = {
        "vdiff_acc": self.rb.vdiff_acc + 0,
        "vdiff_cnt": self.rb.vdiff_cnt + 0,
        'rdist': self.rb.rdist[self._sampleidx].data.cpu().numpy(),
        'g': self.rb.g[self._sampleidx].data.cpu().numpy(),
        'near_pmask': self.near_pmask.data.cpu().numpy(),
    }
    self.rb.vdiff_acc *= 0
    self.rb.vdiff_cnt *= 0
    for loss_name, loss in self.losses.items():
      for item_name, item in self._samples.items():
        k = f'{item_name}_{loss_name}'
        with torch.no_grad():
          self._cache[f'{k}_post'] = (loss(item, self.p))
          r[f'{k}_gain'] = (self._cache[f'{k}_pre'] -
                            self._cache[f'{k}_post']).cpu().data.numpy()
        r[k] = self._cache[f'{k}_post'].cpu().data.numpy()
    self.rs.append(r)

def load_parameters_from_checkpoint():
  data = pickle.load(open(ARGS.checkpoint, 'rb'))
  return [tf(data[str(i)]) for i in range(10)]


if __name__ == "__main__":
  ARGS = parser.parse_args()
  device = torch.device(ARGS.device)
  nn.set_device(device)
  main()
