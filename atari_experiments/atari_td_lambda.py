import math
import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

import neural_network as mm
from neural_network import tf, tint
from replay_buffer import ReplayBuffer
from envs import AtariEnv
from rainbow import DQN

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=2.5e-4, help="Learning rate", type=float)
parser.add_argument("--run", default=0, help="run", type=int)
parser.add_argument("--mbsize", default=32, help="Minibatch size", type=int)
parser.add_argument("--other_mbsize", default=2048, help="Minibatch size", type=int)
parser.add_argument("--buffer_size", default=500000, help="Replay buffer size",type=int)
parser.add_argument("--clone_interval", default=10000, type=int)
parser.add_argument("--num_rand_classes", default=10, type=int)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--opt", default='adam')
parser.add_argument("--loss_func", default='qlearn')
parser.add_argument("--env_name", default='ms_pacman')
parser.add_argument("--device", default='cuda', help="device")


class odict(dict):

  def __getattr__(self, x):
    return self[x]



def sl1(a, b):
  """Smooth L1 distance"""
  d = a - b
  u = abs(d)
  s = d**2
  m = (u < s).float()
  return u * m + s * (1 - m)


def make_opt(opt, theta, lr, weight_decay):
  if opt == "sgd":
    return torch.optim.SGD(theta, lr, weight_decay=weight_decay)
  elif opt == "msgd":
    return torch.optim.SGD(theta, lr, momentum=0.9, weight_decay=weight_decay)
  elif opt == "rmsprop":
    return torch.optim.RMSprop(theta, lr, weight_decay=weight_decay)
  elif opt == "adam":
    return torch.optim.Adam(theta, lr, weight_decay=weight_decay)
  else:
    raise ValueError(opt)


def fill_buffer_with_expert(env, replay_buffer):
  model_path = f"rainbow_atari_models/{ARGS.env_name}.pth"
  with open(model_path, "rb") as f:
    m = torch.load(f)

  device = mm.get_device()
  dqn = DQN(
      odict({
          "history_length": 4,
          "hidden_size": 256,
          "architecture": "data-efficient",
          "atoms": 51,
          "noisy_std": 0.1,
          "V_min": -10,
          "V_max": 10,
          "device": device,
      }), env.num_actions)
  dqn.load_state_dict(m)
  dqn.eval()
  dqn.to(device)

  rand_classes = np.zeros(replay_buffer.size)
  ram2class = {}
  totr = 0
  obs = env.reset()
  replay_buffer.new_episode(obs, env.enumber % 2)
  it = 0
  while replay_buffer.idx < replay_buffer.size - 10:
    action = dqn.act_e_greedy(
        torch.tensor(obs).float().to(device) / 255, epsilon=0.01)
    obs_ram = env.getRAM().tostring()
    if obs_ram not in ram2class:
      ram2class[obs_ram] = np.random.randint(0, ARGS.num_rand_classes)
    rand_classes[replay_buffer.idx] = ram2class[obs_ram]
    obsp, r, done, tr = env.step(action)
    replay_buffer.add(obs, action, r, done, env.enumber % 2)
    obs = obsp
    totr += tr
    if done:
      totr = 0
      obs = env.reset()
      replay_buffer.new_episode(obs, env.enumber % 2)
    it += 1

  # Remove last episode from replay buffer, as it didn't end
  it = replay_buffer.idx
  curp = replay_buffer.p[it]
  while replay_buffer.p[it] == curp:
    replay_buffer._sumtree.set(it, 0)
    it -= 1
  print(f'went from {replay_buffer.idx} to {it} when deleting states')
  return rand_classes


def main():
  device = torch.device(ARGS.device)
  mm.set_device(device)
  results = {
      "measure": [],
      "parameters": [],
  }

  seed = ARGS.run + 1_642_559  # A large prime number
  torch.manual_seed(seed)
  np.random.seed(seed)
  rng = np.random.RandomState(seed)
  env = AtariEnv(ARGS.env_name)
  mbsize = ARGS.mbsize
  Lambda = ARGS.Lambda
  nhid = 32
  num_measure = 1000
  gamma = 0.99
  clone_interval = ARGS.clone_interval
  num_iterations = ARGS.num_iterations

  num_Q_outputs = 1
  # Model
  _Qarch, theta_q, Qf, _Qsemi = mm.build(
      mm.conv2d(4, nhid, 8, stride=4),  # Input is 84x84
      mm.conv2d(nhid, nhid * 2, 4, stride=2),
      mm.conv2d(nhid * 2, nhid * 2, 3),
      mm.flatten(),
      mm.hidden(nhid * 2 * 12 * 12, nhid * 16),
      mm.linear(nhid * 16, num_Q_outputs),
  )
  clone_theta_q = lambda: [i.detach().clone() for i in theta_q]
  theta_target = clone_theta_q()
  opt = make_opt(ARGS.opt, theta_q, ARGS.learning_rate, ARGS.weight_decay)

  # Replay Buffer
  replay_buffer = ReplayBuffer(seed, ARGS.buffer_size)

  # Losses
  td = lambda s, a, r, sp, t, idx, w, tw: sl1(
      r + (1 - t.float()) * gamma * Qf(sp, tw)[:, 0].detach(),
      Qf(s, w)[:, 0],
  )

  tdL = lambda s, a, r, sp, t, idx, w, tw: sl1(
      Qf(s, w)[:, 0], replay_buffer.LG[idx])

  mc = lambda s, a, r, sp, t, idx, w, tw: sl1(
      Qf(s, w)[:, 0], replay_buffer.g[idx])


  # Define metrics
  measure = Measures(
      theta_q, {
          "td": lambda x, w: td(*x, w, theta_target),
          "tdL": lambda x, w: tdL(*x, w, theta_target),
          "mc": lambda x, w: mc(*x, w, theta_target),
      }, replay_buffer, results["measure"], 32)


  # Get expert trajectories
  rand_classes = fill_buffer_with_expert(env, replay_buffer)
  # Compute initial values
  replay_buffer.compute_values(lambda s: Qf(s, theta_q), num_Q_outputs)
  replay_buffer.compute_returns(gamma)
  replay_buffer.compute_reward_distances()
  replay_buffer.compute_episode_boundaries()
  replay_buffer.compute_lambda_returns(lambda s: Qf(s, theta_q), Lambda, gamma)

  # Run policy evaluation
  for it in range(num_iterations):
    do_measure = not it % num_measure
    sample = replay_buffer.sample(mbsize)

    if do_measure:
      measure.pre(sample)
    replay_buffer.compute_value_difference(sample, Qf(sample[0], theta_q))

    loss = tdL(*sample, theta_q, theta_target)
    loss = loss.mean()
    loss.backward()
    opt.step()
    opt.zero_grad()

    replay_buffer.update_values(sample, Qf(sample[0], theta_q))
    if do_measure:
      measure.post()

    if it and clone_interval and it % clone_interval == 0:
      theta_target = clone_theta_q()
      replay_buffer.compute_lambda_returns(lambda s: Qf(s, theta_q), Lambda, gamma)

    if it and it % clone_interval == 0 or it == num_iterations - 1:
      ps = {str(i): p.data.cpu().numpy() for i, p in enumerate(theta_q)}
      ps.update({"step": it})
      results["parameters"].append(ps)

  with open(f'results/td_lambda_{run}.pkl', 'wb') as f:
    pickle.dump(results, f)


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


if __name__ == "__main__":
  ARGS = parser.parse_args()
  main()
