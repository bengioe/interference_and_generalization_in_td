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

import neural_network as nn
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



def main():
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
  start_step = ARGS.start_step
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

  td = lambda s, a, r, sp, t, w, tw=theta_q: sl1(
      r + (1 - t.float()) * gamma * Qf(sp, tw).max(1)[0].detach(),
      Qf(s, w)[np.arange(len(a)), a.long()],
  )

  obs = env.reset()

  if replay_type == "normal":
    replay_buffer = ReplayBuffer(seed, buffer_size, near_strategy=sample_near)
  elif replay_type == "prioritized":
    replay_buffer = PrioritizedExperienceReplay(
        seed, buffer_size, near_strategy=sample_near)

  total_reward = 0
  last_end = 0
  num_fill = 200000
  num_measure = 500
  _t0 = t0 = t1 = t2 = t3 = t4 = time.time()
  tm0 = tm1 = tm2 = tm3 = time.time()
  ema_loss = 0
  last_rewards = [0]

  measure = Measures()
  print("Filling buffer")

  if start_step < num_exploration_steps:
    epsilon = 1 - (start_step / num_exploration_steps) * (1 - final_epsilon)
  else:
    epsilon = final_epsilon

  for it in range(num_fill):
    if start_step == 0:
      action = rng.randint(0, num_act)
    else:
      if rng.uniform(0, 1) < epsilon:
        action = rng.randint(0, num_act)
      else:
        action = Qf(tf(obs / 255.0).unsqueeze(0)).argmax().item()
    obsp, r, done, info = env.step(action)
    replay_buffer.add(obs, action, r, done, env.enumber % 2)
    if replay_type == "prioritized":
      replay_buffer.set_last_priority(
          td(
              tf(obs / 255.0).unsqueeze(0),
              tint([action]),
              r,
              tf(obsp / 255.0).unsqueeze(0),
              tf([done]),
              theta_q,
              theta_q,
          ))

    obs = obsp
    if done:
      obs = env.reset()

  past_theta = [clone_theta_q()]

  for it in range(start_step, num_iterations):
    do_measure = not it % num_measure
    eta = (time.time() - _t0) / (it + 1) * (num_iterations - it) / 60
    if it and it % 100_000 == 0 or it == num_iterations - 1:
      ps = {str(i): p.data.cpu().numpy() for i, p in enumerate(theta_q)}
      ps.update({"step": it})
      results["parameters"].append(ps)

    if it % 10_000 == 0:
      print(
          it,
          f"{(t1 - t0)*1000:.2f}, {(t2 - t1)*1000:.2f}, {(t3 - t2)*1000:.2f}, {(t4 - t3)*1000:.2f},",
          f"{(tm1 - tm0)*1000:.2f}, {(tm3 - tm2)*1000:.2f},",
          f"{int(eta//60):2d}h{int(eta%60):02d}m left",
          f":: {ema_loss:.5f}, last 10 rewards: {np.mean(last_rewards):.2f}",
      )

    t0 = time.time()
    if it < num_exploration_steps:
      epsilon = 1 - (it / num_exploration_steps) * (1 - final_epsilon)
    else:
      epsilon = final_epsilon

    if rng.uniform(0, 1) < epsilon:
      action = rng.randint(0, num_act)
    else:
      action = Qf(tf(obs / 255.0).unsqueeze(0)).argmax().item()
    t1 = time.time()
    obsp, r, done, info = env.step(action)
    total_reward += r
    replay_buffer.add(obs, action, r, done, env.enumber % 2)
    if replay_type == "prioritized":
      replay_buffer.set_last_priority(
          td(
              tf(obs / 255.0).unsqueeze(0),
              tint([action]),
              r,
              tf(obsp / 255.0).unsqueeze(0),
              tf([done]),
              theta_q,
              theta_q,
          ))

    obs = obsp
    if done:
      obs = env.reset()
      results["episode"].append({
          "end": it,
          "start": last_end,
          "total_reward": total_reward
      })
      #exp_results["episodes"].append((it - last_end, total_reward))
      last_end = it
      last_rewards = [total_reward] + last_rewards[:10]
      total_reward = 0

    t2 = time.time()
    *sample, idx = replay_buffer.sample(mbsize)
    if slice_size > 0:
      *sample, idx = replay_buffer.slice_near(idx, slice_size, exclude_0=False)
    t3 = time.time()

    target_ws = past_theta[0]
    loss = td(*sample, theta_q, target_ws)

    if replay_type == "prioritized":
      replay_buffer.set_prioties_at(loss.cpu().data.numpy(),
                                    idx.cpu().data.numpy())

    if do_measure:
      tm0 = time.time()
      measure.pre()
      tm1 = time.time()
    loss = loss.mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    if do_measure:
      tm2 = time.time()
      measure.post()
      tm3 = time.time()
    t4 = time.time()
    if it and clone_interval and it % clone_interval == 0:
      past_theta = [clone_theta_q()] + past_theta[:max_clones - 1]
      if reset_on_clone:
        for p in theta_q[-2:]:
          nn.init_weight(p)
      if reset_opt_on_clone:
        opt = make_opt()

    #exp_results["loss"].append(loss.item())
    ema_loss = 0.999 * ema_loss + 0.001 * loss.item()
    if do_measure:
      measure.log(results["measure"])
  with open(f'results/{run}.pkl', 'wb') as f:
    pickle.dump(results, f)
  print(f"Done in {(time.time()-_t0)/60:.2f}m")


class Measures:

  def pre(self):
    # No I'm not proud
    e = inspect.currentframe().f_back.f_locals
    self._td = lambda *x: e["td"](*x, e["theta_q"], e["target_ws"])
    with torch.no_grad():
      self.other_samples = e["replay_buffer"].sample(e["mbsize"])[:-1]
      self.other_td_before = self._td(*self.other_samples)
      self.near_samples = e["replay_buffer"].slice_near(e["idx"], 30)[:-1]
      self.near_td_before = self._td(*self.near_samples)
      self.sample_td_before = e["loss"]

  def post(self):
    e = inspect.currentframe().f_back.f_locals  # Still not proud
    self.other_td_after = self._td(*self.other_samples)
    self.other_td_gain = ((self.other_td_before -
                           self.other_td_after).cpu().data.numpy())
    self.other_td_gain_avg = self.other_td_gain.mean().item()

    self.near_td_after = self._td(*self.near_samples)
    self.near_td_gain = ((self.near_td_before -
                          self.near_td_after).cpu().data.numpy())
    self.near_td_gain_avg = self.near_td_gain.mean().item()

    self.sample_td_after = self._td(*e["sample"])
    self.sample_td_gain = ((self.sample_td_before -
                            self.sample_td_after).cpu().data.numpy())
    self.sample_td_gain_avg = self.sample_td_gain.mean().item()

  def log(self, rs):
    e = inspect.currentframe().f_back.f_locals  # Don't do this at home
    rs.append({
        "td_error": self.sample_td_before.cpu().data.numpy(),
        "other_td_gain": self.other_td_gain,
        "other_td_gain_avg": self.other_td_gain_avg,
        "near_td_gain": self.near_td_gain,
        "near_td_gain_avg": self.near_td_gain_avg,
        "sample_td_gain": self.sample_td_gain,
        "sample_td_gain_avg": self.sample_td_gain_avg,
        "idx": e["idx"].cpu().data.numpy(),
        "step": e["it"],
    })

if __name__ == "__main__":
  ARGS = parser.parse_args()
  device = torch.device(ARGS.device)
  nn.set_device(device)
  main()
