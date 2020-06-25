import math
import torch
from torch.nn import functional as F

import argparse
import numpy as np
import pickle

import neural_network as nn
from neural_network import tf, tint
from replay_buffer import ReplayBuffer
from envs import AtariEnv
from ram_annotations import atari_dict

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=2.5e-4, help="Learning rate", type=float)
parser.add_argument("--run", default=0, help="run", type=int)
parser.add_argument("--mbsize", default=32, help="Minibatch size", type=int)
parser.add_argument("--buffer_size", default=100000, help="Replay buffer size",type=int)
parser.add_argument("--checkpoint", default='results/41.pkl', help="checkpoint file",type=str)
parser.add_argument("--expert_is_self", default=1, help="is expert params", type=int)
parser.add_argument("--loss_func", default='td', help="target loss")
parser.add_argument("--device", default='cuda', help="device")



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


def load_parameters_from_checkpoint():
  data = pickle.load(open(ARGS.checkpoint, 'rb'))
  return [tf(data[str(i)]) for i in range(10)]


def fill_buffer_with_expert(replay_buffer, env_name, epsilon=0.01):
  mbsize = ARGS.mbsize
  envs = [AtariEnv(env_name) for i in range(mbsize)]
  num_act = envs[0].num_actions

  nhid = 32
  _, theta_q, Qf, _ = nn.build(
      nn.conv2d(4, nhid, 8, stride=4),  # Input is 84x84
      nn.conv2d(nhid, nhid * 2, 4, stride=2),
      nn.conv2d(nhid * 2, nhid * 2, 3),
      nn.flatten(),
      nn.hidden(nhid * 2 * 12 * 12, nhid * 16),
      nn.linear(nhid * 16, num_act),
  )

  theta_q_trained = load_parameters_from_checkpoint()
  if ARGS.expert_is_self:
    theta_expert = theta_q_trained
  else:
    expert_id = {
        'ms_pacman':457, 'asterix':403, 'seaquest':428}[env_name]
    with open(f'checkpoints/dqn_model_{expert_id}.pkl',
              "rb") as f:
      theta_expert = pickle.load(f)
    theta_expert = [tf(i) for i in theta_expert]

  obs = [i.reset() for i in envs]
  trajs = [list() for i in range(mbsize)]
  enumbers = list(range(mbsize))
  replay_buffer.ram = torch.zeros([replay_buffer.size, 128],
                                  dtype=torch.uint8,
                                  device=replay_buffer.device)

  while True:
    mbobs = tf(obs) / 255
    greedy_actions = Qf(mbobs, theta_expert).argmax(1)
    random_actions = np.random.randint(0, num_act, mbsize)
    actions = [
        j if np.random.random() < epsilon else i
        for i, j in zip(greedy_actions, random_actions)
    ]
    for i, (e, a) in enumerate(zip(envs, actions)):
      obsp, r, done, _ = e.step(a)
      trajs[i].append([obs[i], int(a), float(r), int(done), e.getRAM() + 0])
      obs[i] = obsp
        if replay_buffer.idx + len(trajs[i]) + 4 >= replay_buffer.size:
          # We're done!
          return Qf, theta_q_trained
        replay_buffer.new_episode(trajs[i][0][0], enumbers[i] % 2)
        for s, a, r, d, ram in trajs[i]:
          replay_buffer.ram[replay_buffer.idx] = tint(ram)
          replay_buffer.add(s, a, r, d, enumbers[i] % 2)

        trajs[i] = []
        obs[i] = envs[i].reset()
        enumbers[i] = max(enumbers) + 1



def main():
  gamma = 0.99
  hps = pickle.load(open(ARGS.checkpoint, 'rb'))['hps']
  env_name = hps["env_name"]
  if 'Lambda' in hps:
    Lambda = hps['Lambda']
  else:
    Lambda = 0

  device = torch.device(ARGS.device)
  nn.set_device(device)
  replay_buffer = ReplayBuffer(ARGS.run, ARGS.buffer_size)
  Qf, theta_q = fill_buffer_with_expert(replay_buffer, env_name)
  for p in theta_q:
    p.requires_grad = True
  if Lambda > 0:
    replay_buffer.compute_episode_boundaries()
    replay_buffer.compute_lambda_returns(lambda s: Qf(s, theta_q), Lambda, gamma)

  td__ = lambda s, a, r, sp, t, idx, w, tw: sl1(
      r + (1 - t.float()) * gamma * Qf(sp, tw).max(1)[0].detach(),
      Qf(s, w)[np.arange(len(a)), a.long()],
  )

  td = lambda s, a, r, sp, t, idx, w, tw: Qf(s, w).max(1)[0]

  tdL = lambda s, a, r, sp, t, idx, w, tw: sl1(
      Qf(s, w)[:, 0], replay_buffer.LG[idx])

  loss_func = {
      'td': td, 'tdL': tdL}[ARGS.loss_func]

  opt = torch.optim.SGD(theta_q, 1)

  def grad_sim(inp, grad):
    dot = sum([(p.grad * gp).sum() for p, gp in zip(inp, grad)])
    nA = torch.sqrt(sum([(p.grad**2).sum() for p, gp in zip(inp, grad)]))
    nB = torch.sqrt(sum([(gp**2).sum() for p, gp in zip(inp, grad)]))
    return (dot / (nA * nB)).item()

  relevant_features = np.int32(
      sorted(list(atari_dict[env_name.replace("_", "")].values())))
  sims = []
  ram_sims = []
  for i in range(2000):
    sim = []
    *sample, idx = replay_buffer.sample(1)
    loss = loss_func(*sample, idx, theta_q, theta_q).mean()
    loss.backward()
    g0 = [p.grad + 0 for p in theta_q]
    for j in range(-30, 31):
      opt.zero_grad()
      loss = loss_func(*replay_buffer.get(idx + j), theta_q, theta_q).mean()
      loss.backward()
      sim.append(grad_sim(theta_q, g0))
    sims.append(np.float32(sim))
    for j in range(200):
      opt.zero_grad()
      *sample_j, idx_j = replay_buffer.sample(1)
      loss = loss_func(*sample_j, idx_j, theta_q, theta_q).mean()
      loss.backward()
      ram_sims.append(
          (grad_sim(theta_q, g0),
           abs(replay_buffer.ram[idx[0]][relevant_features].float() -
               replay_buffer.ram[idx_j[0]][relevant_features].float()).mean()))
    opt.zero_grad()
  ram_sims = np.float32(
      ram_sims)  #np.histogram(np.float32(ram_sim), 100, (-1, 1))

  # Compute "True" gradient
  grads = [i.detach() * 0 for i in theta_q]
  N = 0
  for samples in replay_buffer.in_order_iterate(ARGS.mbsize * 8):
    loss = loss_func(*samples, theta_q, theta_q).mean()
    loss.backward()
    N += samples[0].shape[0]
    for p, gp in zip(theta_q, grads):
      gp.data.add_(p.grad)
    opt.zero_grad()

  dots = []
  i = 0
  for sample in replay_buffer.in_order_iterate(1):
    loss = loss_func(*sample, theta_q, theta_q).mean()
    loss.backward()
    dots.append(grad_sim(theta_q, grads))
    opt.zero_grad()
    i += 1
  histo = np.histogram(dots, 100, (-1, 1))

  results = {
      "grads": [i.cpu().data.numpy() for i in grads],
      "sims": np.float32(sims),
      "histo": histo,
      "ram_sims": ram_sims,
  }

  path = f'results/grads_{ARGS.checkpoint}.pkl'
  with open(path, "wb") as f:
    pickle.dump(results, f)


if __name__ == "__main__":
  ARGS = parser.parse_args()
  main()
