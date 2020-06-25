import argparse
import math
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from backpack import backpack, extend
from backpack.extensions import BatchGrad
import copy
import gc
import pickle

import neural_network as mm
from neural_network import tf, tint
from replay_buffer import ReplayBufferV2
from envs import AtariEnv
from ram_annotations import atari_dict
from rainbow import DQN

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=2.5e-4, help="Learning rate", type=float)
parser.add_argument("--run", default=0, help="run", type=int)
parser.add_argument("--mbsize", default=32, help="Minibatch size", type=int)
parser.add_argument("--other_mbsize", default=32, help="Measures Minibatch size", type=int)
parser.add_argument("--buffer_size", default=50000, help="Replay buffer size",type=int)
parser.add_argument("--clone_interval", default=10000, type=int)
parser.add_argument("--num_iterations", default=5000, type=int)
parser.add_argument("--num_rand_classes", default=10, type=int)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--opt", default='adam')
parser.add_argument("--loss_func", default='qlearn')
parser.add_argument("--env_name", default='ms_pacman')
parser.add_argument("--device", default='cuda', help="device")
parser.add_argument("--comment", default='')


def gpu_usage():
  size = 0
  for obj in gc.get_objects():
    try:
      if torch.is_tensor(obj):
        size += np.prod(obj.size()) if len(obj.size()) else 0
      elif (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        size += np.prod(obj.data.size()) if len(obj.data.size()) else 0
    except Exception as e:
      pass
  print('gpu alloc', size, size // 1024, size // (1024 ** 2))


class SumLoss(torch.nn.Module):
    def __init__(self):
        super(SumLoss, self).__init__()
    def forward(self, input):
        return input.sum()

sumloss = extend(SumLoss())


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
    raise ValueError(hps["opt"])


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
  relevant_features = np.int32(
      sorted(list(atari_dict[env.env_name.replace("_", "").lower()].values())))

  totr = 0
  obs = env.reset()
  #for it in range(replay_buffer.size):
  it = 0
  while not replay_buffer.hit_max:
    action = dqn.act_e_greedy(
        torch.tensor(obs).float().to(device) / 255, epsilon=0.01)
    #obs_ram = env.getRAM()
    obsp, r, done, tr = env.step(action)
    replay_buffer.add(obs, action, r, done)
    #, ram_info=obs_ram[relevant_features])
    obs = obsp
    totr += tr
    if done:
      print("Done episode %d reward %d"%(totr, replay_buffer.current_size))
      totr = 0
      obs = env.reset()
    it += 1
  return dqn

def init_weights(m):
  if isinstance(m, torch.nn.Linear):
    k = np.sqrt(6 / (np.sum(m.weight.shape)))
    m.weight.data.uniform_(-k, k)
    m.bias.data.fill_(0)
  if isinstance(m, torch.nn.Conv2d):
    u,v,w,h = m.weight.shape
    k = np.sqrt(6 / (w*h*u + w*h*v))
    m.weight.data.uniform_(-k, k)
    m.bias.data.fill_(0)


def main():
  device = torch.device(ARGS.device)
  mm.set_device(device)
  results = {
      "measure": [],
      "parameters": [],
      "args": ARGS,
  }

  seed = ARGS.run + 1_642_559  # A large prime number
  torch.manual_seed(seed)
  np.random.seed(seed)
  rng = np.random.RandomState(seed)
  env = AtariEnv(ARGS.env_name)
  mbsize = ARGS.mbsize
  nhid = 32
  num_measure = 50
  gamma = 0.99
  clone_interval = ARGS.clone_interval
  num_iterations = ARGS.num_iterations

  num_Q_outputs = env.num_actions if ARGS.loss_func != 'rand' else ARGS.num_rand_classes
  # Model
  act = torch.nn.LeakyReLU()
  Qf = torch.nn.Sequential(torch.nn.Conv2d(4, nhid, 8, stride=4, padding=4), act,
                           torch.nn.Conv2d(nhid, nhid*2, 4, stride=2,padding=2), act,
                           torch.nn.Conv2d(nhid*2, nhid*2, 3,padding=1), act,
                           torch.nn.Flatten(),
                           torch.nn.Linear(nhid*2*12*12, nhid*16), act,
                           torch.nn.Linear(nhid*16, num_Q_outputs))
  Qf.to(device)
  Qf.apply(init_weights)
  Qf_target = copy.deepcopy(Qf)
  Qf = extend(Qf)

  opt = make_opt(ARGS.opt, Qf.parameters(), ARGS.learning_rate, ARGS.weight_decay)

  # Replay Buffer
  replay_buffer = ReplayBufferV2(seed, ARGS.buffer_size,
                                 value_callback=lambda s: Qf(s),
                                 Lambda=0)

  td = lambda x: sl1(
      expert.get_Q(x.s)[np.arange(len(x.a)), x.a.long()] * 0.1,
      Qf(x.s)[np.arange(len(x.a)), x.a.long()],
  )
  sarsa = lambda x: sl1(
      x.r + ((1 - x.t.float())
             * gamma
             * Qf_target(x.sp)[np.arange(len(x.ap)), x.ap.long()].detach()),
      Qf(x.s)[np.arange(len(x.a)), x.a.long()],
  )


  mc = lambda x: sl1(
      Qf(x.s).max(1)[0], x.g)


  if ARGS.loss_func == 'rand':
    raise ValueError('fixme Qf')

    def rand_nll(s, a, r, sp, t, idx, w, tw):
      return F.cross_entropy(Qf(s, w), tint(rand_classes[idx]), reduce=False)
    def rand_acc(s, a, r, sp, t, idx, w, tw):
      return (Qf(s, w).argmax(1) != tint(rand_classes[idx])).float()

    # Define metrics
    measure = Measures(
        theta_q, {
            "rand_nll": lambda x, w: rand_nll(*x, w, theta_target),
            "rand_acc": lambda x, w: rand_acc(*x, w, theta_target),
        }, replay_buffer, results["measure"], 32)

    loss_func = rand_nll
  else:
    # Define metrics
    measure = Measures(
        list(Qf.parameters()), {
            "td": td,
            "func": lambda x: Qf(x.s).max(1).values,
            #"sarsa": sarsa,
            #"mc": mc,
        }, replay_buffer, results["measure"], 32,
        lambda x: Qf(x.s),
        Qf)

    loss_func = {
        "sarsa": sarsa,
        "qlearn": td,
        "mc": mc,
    }[ARGS.loss_func]


  # Get expert trajectories
  expert = fill_buffer_with_expert(env, replay_buffer)

  # Run policy evaluation
  for it in tqdm(range(num_iterations), smoothing=0):
    do_measure = not it % num_measure
    sample = replay_buffer.sample(mbsize)

    if do_measure:
      measure.pre(sample)
    #v_before = Qf(sample[0])

    opt.zero_grad()
    loss = loss_func(sample)
    loss = loss.mean()
    loss.backward()
    opt.step()

    #replay_buffer.update_values(sample, v_before, Qf(sample[0], theta_q))
    if do_measure:
      measure.post()

    if it and clone_interval and it % clone_interval == 0:
      if ARGS.loss_func not in ['mc', 'rand']:
        Qf_target = copy.deepcopy(Qf)

    if it and it % clone_interval == 0 and False or it == num_iterations - 1:
      ps = {str(i): p.data.cpu().numpy() for i, p in enumerate(Qf.parameters())}
      ps.update({"step": it})
      results["parameters"].append(ps)

  with open(f'results/rainbow_regress_{ARGS.run}.pkl', 'wb') as f:
    pickle.dump(results, f)


def interferences(loss, params):
  with backpack(BatchGrad()):
    sumloss(loss).backward()
  grads = torch.cat([i.grad_batch.reshape((loss.shape[0], -1)) for i in params], 1)
  all_dots = []
  for i in range(len(grads)-1):
    all_dots.append((grads[i][None, :] @ grads[i+1:].t())[0])
  return torch.cat(all_dots, 0).cpu().data.numpy()

def self_interferences(loss, params, t=False):
  with backpack(BatchGrad()):
    sumloss(loss).backward()
  grads = torch.cat([i.grad_batch.reshape((loss.shape[0], -1)) for i in params], 1)
  if t:
    return (grads ** 2).sum(1)
  return (grads ** 2).sum(1).cpu().data.numpy()

def cross_interferences(lossA, lossB, params, t=False):
  """lossA and B must be callbacks, because for backpack to work it
  seems there must only be one computation graph associated with the
  parameters at a time!
  Or like it checks the most recent one only? I'm confused
  """
  with backpack(BatchGrad()):
    sumloss(lossA()).backward()
  gradsA = torch.cat([i.grad_batch.reshape((i.grad_batch.shape[0], -1)) for i in params], 1)
  with backpack(BatchGrad()):
    sumloss(lossB()).backward()
  gradsB = torch.cat([i.grad_batch.reshape((i.grad_batch.shape[0], -1)) for i in params], 1)
  x = gradsA @ gradsB.t()
  gc.collect()
  if t:
    return x
  return x.cpu().data.numpy()

def cross_interferences_diag(lossA, lossB, params, t=False):
  """lossA and B must be callbacks, because for backpack to work it
  seems there must only be one computation graph associated with the
  parameters at a time!
  Or like it checks the most recent one only? I'm confused
  """
  with backpack(BatchGrad()):
    sumloss(lossA()).backward()
  gradsA = torch.cat([i.grad_batch.reshape((i.grad_batch.shape[0], -1)) for i in params], 1)
  with backpack(BatchGrad()):
    sumloss(lossB()).backward()
  gradsB = torch.cat([i.grad_batch.reshape((i.grad_batch.shape[0], -1)) for i in params], 1)
  x = (gradsA * gradsB).sum(1)
  gc.collect()
  if t: return x
  return x.cpu().data.numpy()

def td_interf_grad_terms(A, B, Qf):
  """
  \rho'_{TD} =
     \delta_{B}^{2}g_{AB}(g_{AB}-\gamma g_{A'B})
   + \delta_{A}\delta_{B}g_{AB}(g_{BB}-\gamma g_{B'B})
   - \delta_{A} \delta_{B}^{2} \nabla_{\theta}^{f(B)}H_{\theta}^{f(A)}
               (\nabla_{\theta}^{f(B)}+\nabla_{\theta}^{f(A)})

  \rho'_{reg} =
     g^{2}\delta_{B}^{2}
     +2\delta_{A}\delta_{B}g_{AB}g_{BB}
      -\delta_{A}\delta_{B}^{2}\nabla_{\theta}^{f(B)}H_{\theta}^{f(A)}
               (\nabla_{\theta}^{f(B)}+\nabla_{\theta}^{f(A)})
  """
  theta = list(Qf.parameters())
  fA = Qf(A.s)[np.arange(len(A.a)), A.a.long()]
  fB = Qf(B.s)[np.arange(len(B.a)), B.a.long()]
  fAp = Qf(A.sp).max(1).values
  fBp = Qf(B.sp).max(1).values
  delta_A = fA - A.r - 0.99 * fAp
  delta_B = fB - B.r - 0.99 * fBp

  g_AB = cross_interferences(lambda: Qf(A.s)[np.arange(len(A.a)), A.a.long()],
                             lambda: Qf(B.s)[np.arange(len(B.a)), B.a.long()], theta, t=True)
  g_BB = self_interferences(Qf(B.s)[np.arange(len(B.a)), B.a.long()], theta, t=True)

  term0 = delta_B[None, :]**2 * g_AB**2
  term1 = delta_A[:, None] * delta_B[None, :] * g_AB * g_BB[None, :]
  term2 = delta_A[:, None] * delta_B[None, :]**2

  torch.cuda.empty_cache()

  S = lambda l: torch.cat([i.flatten() if i is not None else torch.zeros_like(p)
                           for i,p in zip(l, theta)])
  inner_grads = torch.zeros_like(term2)
  for i in tqdm(range(len(A.s)), leave=False):
    for j in range(len(B.s)):
      # compute VfB (HfA VfB + HfB VfA)
      dfadt = S(torch.autograd.grad(Qf(A.s[i][None,:])[0, A.a[i].long()], theta, create_graph=True))
      dfbdt = S(torch.autograd.grad(Qf(B.s[j][None,:])[0, B.a[j].long()], theta, create_graph=True))
      HfA = S(torch.autograd.grad(dfadt @ dfbdt.detach(), theta,
                                  create_graph=True, allow_unused=True))
      HfB = S(torch.autograd.grad(dfadt.detach() @ dfbdt, theta,
                                  create_graph=True, allow_unused=True))
      inner_grads[i, j] = ((HfA + HfB) @ dfbdt).detach()

  term2 = term2 * inner_grads
  return list(map(lambda x: x.data.cpu().numpy(), [term0, term1, term2]))

class Measures:

  def __init__(self, params, losses, replay_buffer, results, mbsize, value_callback, Qf):
    self.p = params
    self.losses = losses
    self.rb = replay_buffer
    self.mbsize = mbsize
    self.other_mbsize = ARGS.other_mbsize
    self.value_callback = value_callback
    self.Qf = Qf
    self.rs = results

  def pre(self, sample):
    near_s, self.near_pmask = self.rb.slice_near(sample, 10)
    self._samples = {
        "sample": sample,
        "other": self.rb.sample(self.other_mbsize),
        "near": near_s,
    }
    self.r = {}
    if 0:
      with torch.no_grad():
        v_other = self.value_callback(self._samples['other'])
        v_sample = self.value_callback(self._samples['sample'])
        self.value_dists = abs(v_other[None, :, :] - v_sample[:, None, :]).mean(2)
    self._cache = {}
    for loss_name, loss in self.losses.items():
      for item_name, item in self._samples.items():
        l = loss(item)
        self._cache[f'{item_name}_{loss_name}_pre'] = l
        if item_name != 'near':
          self.r[f'{item_name}_{loss_name}_int'] = interferences(l, self.p)
      self.r[f'sample_x_other_{loss_name}_int'] = cross_interferences(
        lambda: loss(self._samples['sample']),
        lambda: loss(self._samples['other']), self.p)
      self.r[f'sample_x_near_{loss_name}_int'] = cross_interferences(
        lambda: loss(self._samples['sample'].slice(0, 4)),
        lambda: loss(self._samples['near'].slice(0, 4)), self.p)

    rhop_terms = td_interf_grad_terms(self._samples['other'], self._samples['sample'], self.Qf)
    for i in range(3):
      self.r[f'sample_x_other_rhop_{i}'] = rhop_terms[i]

  def post(self):
    #sample_ram = self.rb.getRAM(self._samples['sample']).float()
    #other_ram = self.rb.getRAM(self._samples['other']).float()
    #ram_dists = abs(sample_ram[:,None,:] - other_ram[None,:,:]).sum(2)
    self.r.update({
        #"vdiff_acc": self.rb.vdiff_acc + 0,
        #"vdiff_cnt": self.rb.vdiff_cnt + 0,
        'near_pmask': self.near_pmask.data.cpu().numpy(),
        #'ram_dists': ram_dists.data.cpu().numpy(),
        #'value_dists': self.value_dists.data.cpu().numpy(),
    })
    #self.rb.vdiff_acc *= 0
    #self.rb.vdiff_cnt *= 0
    for loss_name, loss in self.losses.items():
      for item_name, item in self._samples.items():
        k = f'{item_name}_{loss_name}'
        with torch.no_grad():
          self._cache[f'{k}_post'] = (loss(item))
          self.r[f'{k}_gain'] = (self._cache[f'{k}_pre'] -
                            self._cache[f'{k}_post']).cpu().data.numpy()
        self.r[k] = self._cache[f'{k}_post'].cpu().data.numpy()
    self.rs.append(self.r)


if __name__ == "__main__":
  ARGS = parser.parse_args()
  main()
