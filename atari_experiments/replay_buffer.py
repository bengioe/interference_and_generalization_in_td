import gc

import torch
import numpy as np
from collections import namedtuple

from neural_network import tf, tint, get_device

#Minibatch =
class Minibatch(namedtuple('Minibatch', ['s', 'a', 'r', 'sp', 't', 'g',
                                         'lg', 'ap', 'idx', 'eidx'])):
  def slice(self, start, stop):
    return Minibatch(self.s[start:stop],
                     self.a[start:stop],
                     self.r[start:stop],
                     self.sp[start:stop],
                     self.t[start:stop],
                     self.g[start:stop],
                     self.lg[start:stop],
                     self.ap[start:stop],
                     self.idx[start:stop],
                     self.eidx[start:stop])

class LambdaReturn:

  def __init__(self, Lambda, gamma, max_len=4096):
    self.Lambda = Lambda
    self.gamma = gamma
    self.max_len = max_len
    self.Lt = tf(np.zeros((max_len, max_len), dtype="float32"))
    self.gt = tf(np.zeros((max_len, max_len), dtype="float32"))
    for i in range(max_len):
      self.gt[i, i:] = gamma**torch.arange(max_len - i)
      self.Lt[i, i:] = Lambda**torch.arange(max_len - i)

  def __call__(self, r, v_sp):
    T = r.shape[0]
    if T > self.max_len: # This almost never occurs, tbf
      t = self.max_len
      return torch.cat([self(r[:t], v_sp[:t]),
                        self(r[t:], v_sp[t:])])
    alive = np.ones(T)
    alive[-1] = 0
    alive = tf(alive)
    n_step_rewards = torch.cumsum(r[None, :] * self.gt[:T, :T], 1)
    n_step_returns = alive[None, :] * (
        n_step_rewards + self.gamma * v_sp[None, :] * self.gt[:T, :T])
    weighted_n_step_returns = (1 - self.Lambda) * (
        n_step_returns * self.Lt[:T, :T]).sum(1)
    weighted_mc_returns = self.Lt[:T, T] * self.Lambda * n_step_rewards[:, -1]
    lambda_target = weighted_n_step_returns + weighted_mc_returns
    return lambda_target



class Episode:
  def __init__(self, gamma):
    self.s = []
    self.a = []
    self.r = []
    self.t = []
    self.ram = []
    self.gamma = gamma
    self.is_over = False
    self.device = get_device()

  def add(self,s,a,r,t,ram_info=None):
    self.s.append(s)
    self.a.append(a)
    self.r.append(r)
    self.t.append(t)
    if ram_info is not None:
      self.ram.append(ram_info)
    if len(self.s) == 1: # is this the first frame?
      # Add padding frames
      for i in range(3):
        self.add(s,a,r,t,ram_info)

  def end(self):
    self.length = len(self.s)
    self.add(self.s[-1], self.a[-1], 0, 1,
             self.ram[-1] if len(self.ram) else None)
    self.s = torch.stack(self.s)
    self.a = torch.tensor(self.a, dtype=torch.uint8, device=self.device)
    self.r = torch.tensor(self.r, dtype=torch.float32, device=self.device)
    self.t = torch.tensor(self.t, dtype=torch.uint8, device=self.device)
    self.lg = torch.zeros([self.length], dtype=torch.float32, device=self.device)
    if len(self.ram):
      self.ram = torch.tensor(self.ram, dtype=torch.uint8, device=self.device)
    self.is_over = True
    self.compute_returns()
    gc.collect()

  def i2y(self, idx):
    d = tint((-3, -2, -1, 0, 1))  # 4 state history slice + 1 for s'
    sidx = (idx.reshape((idx.shape[0], 1)) + d)
    return (
        self.s[sidx[:, :4]].float() / 255,
        self.a[idx],
        self.r[idx],
        self.s[sidx[:, 1:]].float() / 255,
        self.t[idx],
        self.g[idx],
        self.lg[idx],
        self.a[idx+1],
        idx,
    )

  def compute_returns(self):
    self.g = torch.zeros([self.length], dtype=torch.float32, device=self.device)
    g = 0
    for i in range(self.length - 1, -1, -1):
      self.g[i] = g = self.r[i] + self.gamma * g

  def compute_values(self, V):
    s = self.i2y(tint(np.arange(3, self.length-1)))[0]
    with torch.no_grad():
      v = V(s).detach()
    if not hasattr(self, 'v'):
      self.v = torch.zeros([self.length, v.shape[1]], dtype=torch.float32, device=self.device)
    self.v[3:self.length-1] = v

  def compute_lambda_return(self, V, lr):
    s = self.i2y(tint(np.arange(4, self.length)))[0]
    with torch.no_grad():
      vp = V(s).detach()
      self.lg[3:self.length-1] = lr(self.r[3:self.length-1], vp)

  def sample(self):
    return self.i2y(tint([np.random.randint(3,self.length-1)]))

  def update_value(self, idx, vb, va):
    old_v = self.v[idx]
    self.v[idx] = va
    return old_v - vb

  def slice_near(self, i, dist, exclude_0=True):
    ar = np.arange(-dist, dist + 1)
    if exclude_0:
      ar = ar[ar != 0]
    sidx = i + tint(ar)
    pmask = (sidx >= 3).float() * (sidx <= self.length - 2).float()
    sidx = sidx.clamp(3, self.length - 2)
    return (*self.i2y(sidx), pmask)

class ReplayBufferV2:
  def __init__(self, seed, size, value_callback=None, target_value_callback=None, Lambda=0.9, gamma=0.99, nbins=512):
    self.current_size = 0
    self.size = size
    self.device = get_device()
    self.rng = np.random.RandomState(seed)
    self.value_callback = value_callback
    self.target_value_callback = target_value_callback
    self.Lambda = Lambda
    self.gamma = gamma
    self.hit_max = False
    if self.Lambda > 0:
      self.lr = LambdaReturn(Lambda, gamma)
    self.current_episode = Episode(self.gamma)
    self.episodes = []
    self.vdiff_acc = np.zeros(nbins)
    self.vdiff_cnt = np.zeros(nbins)
    self.vdiff_bins = np.linspace(-2, 2, nbins-1)

  def compute_value_difference(self, sample, v_before, v_after):
    idxs, eidxs = sample[-2:]
    diff = np.float32([
        self.episodes[eidx].update_value(idx, vb, va).data.cpu().numpy()
        for idx, eidx, vb, va in zip(idxs, eidxs, v_before, v_after)]).flatten()
    bins = np.digitize(diff, self.vdiff_bins)
    self.vdiff_acc += np.bincount(bins, diff, minlength=self.vdiff_acc.shape[0])
    self.vdiff_cnt += np.bincount(bins, minlength=self.vdiff_acc.shape[0])

  def add(self, s,a,r,t, ram_info=None):
    self.current_episode.add(
        torch.tensor(s[-1], dtype=torch.uint8).to(self.device),
        a,
        r,
        t * 1,
        ram_info=ram_info)
    if t:
      self.current_episode.end()
      #self.current_episode.compute_values(self.value_callback)
      if self.Lambda > 0:
        self.current_episode.compute_lambda_return(
            self.target_value_callback, self.lr)
      self.current_size += self.current_episode.length
      self.episodes.append(self.current_episode)
      self.current_episode = Episode(self.gamma)
      while self.current_size > self.size:
        self.hit_max = True
        e = self.episodes.pop(0)
        self.current_size -= e.length

  def sample(self, n):
    eidx = self.rng.randint(0, len(self.episodes), n)
    data = [self.episodes[i].sample() for i in eidx]
    return Minibatch(*[torch.cat([d[i] for d in data])
                       for i in range(len(data[0]))], eidx)

  def recompute_lambda_returns(self):
    for e in self.episodes:
      e.compute_lambda_return(self.target_value_callback, self.lr)

  def getRAM(self, minibatch):
    data = [self.episodes[i].ram[j]
            for i, j in zip(minibatch.eidx, minibatch.idx)]
    return torch.stack(data)

  def slice_near(self, minibatch, n):
    data = [self.episodes[i].slice_near(j, n)
            for i, j in zip(minibatch.eidx, minibatch.idx)]
    eidx = np.repeat(minibatch.eidx, data[0][0].shape[0])
    pmask = torch.cat([d[-1] for d in data])
    x = Minibatch(*[torch.cat([d[i] for d in data])
                    for i in range(len(data[0]) - 1)], eidx), pmask
    return x




class ReplayBuffer:

  def __init__(self, seed, size, iwidth=84, near_strategy="both", extras=[]):
    self.rng = np.random.RandomState(seed)
    self.size = size
    self.near_strategy = near_strategy
    self.device = device = get_device()
    # Storing s,a,r,done,episode parity
    self.s = torch.zeros([size, iwidth, iwidth],
                         dtype=torch.uint8,
                         device=device)
    self.a = torch.zeros([size], dtype=torch.uint8, device=device)
    self.r = torch.zeros([size], dtype=torch.float32, device=device)
    self.t = torch.zeros([size], dtype=torch.uint8, device=device)
    self.p = torch.zeros([size], dtype=torch.uint8, device=device)
    self.idx = 0
    self.last_idx = 0
    self.maxidx = 0
    self.is_filling = True
    self._sumtree = SumTree(self.rng, size)
    self._sumtree = SumTree(self.rng, size)

  def compute_returns(self, gamma):
    if not hasattr(self, "g"):
      self.g = torch.zeros([self.size], dtype=torch.float32, device=self.device)
    g = 0
    for i in range(self.maxidx - 1, -1, -1):
      self.g[i] = g = self.r[i] + gamma * g
      if self.t[i]:
        g = 0

  def compute_reward_distances(self):
    if not hasattr(self, "rdist"):
      self.rdist = torch.zeros([self.size], dtype=torch.float32, device=self.device)

    d = 0
    for i in range(self.maxidx - 1, -1, -1):
      self.rdist[i] = d
      d += 1
      if self.r[i] != 0:
        d = 0


  def compute_values(self, fun, num_act, mbsize=128, nbins=256):
    if not hasattr(self, "last_v"):
      self.last_v = torch.zeros([self.size, num_act],
                                dtype=torch.float32,
                                device=self.device)
      self.vdiff_acc = np.zeros(nbins)
      self.vdiff_cnt = np.zeros(nbins)
      self.vdiff_bins = np.linspace(-1, 1, nbins-1)

    d = tint((-3, -2, -1, 0))  # 4 state history slice
    idx_0 = tint(np.arange(mbsize)) + 3
    idx_s = idx_0.reshape((-1, 1)) + d
    for i in range(int(np.ceil((self.maxidx - 4) / mbsize))):
      islice = idx_s + i * mbsize
      iar = idx_0 + i * mbsize
      if (i + 1) * mbsize >= self.maxidx - 2:
        islice = islice[:self.maxidx - i * mbsize - 2]
        iar = iar[:self.maxidx - i * mbsize - 2]
      s = self.s[islice].float().div_(255)
      with torch.no_grad():
        self.last_v[iar] = fun(s)
      if not i % 100:
        gc.collect()


  def compute_value_difference(self, sample, value):
    idxs = sample[-1]
    last_vs = self.last_v[idxs]
    diff = (last_vs - value).data.cpu().numpy().flatten()
    bins = np.digitize(diff, self.vdiff_bins)
    self.vdiff_acc += np.bincount(bins, diff, minlength=self.vdiff_acc.shape[0])
    self.vdiff_cnt += np.bincount(bins, minlength=self.vdiff_acc.shape[0])

  def update_values(self, sample, value):
    self.last_v[sample[-1]] = value.data


  def compute_episode_boundaries(self):
    self.episodes = []
    i = 0
    while self._sumtree.getp(i) == 0:
      i += 1
    print('start of first episode sampleable state', i)
    start = i
    while i < self.maxidx:
      if self.t[i]:
        self.episodes.append((start, i + 1))
        i += 1
        while i < self.maxidx and self._sumtree.getp(i) == 0:
          i += 1
        start = i
      i += 1
#    if i - start > 0:
#      self.episodes.append((start, i - 1))

  def compute_lambda_returns(self, fun, Lambda, gamma):
    if not hasattr(self, 'LR'):
      self.LR = LambdaReturn(Lambda, gamma)
      self.LG = torch.zeros([self.size], dtype=torch.float32, device=self.device)
    i = 0
    for start, end in self.episodes:
      s = self._idx2xy(tint(np.arange(start + 1, end)))[0]
      with torch.no_grad():
        vp = fun(s)[:, 0].detach()
        vp = torch.cat([vp, torch.zeros((1,), device=self.device)])
      self.LG[start:end] = self.LR(
          self.r[start:end],
          vp)
      i += 1

  def add(self, s, a, r, t, p, sampleable=1):
    self.s[self.idx] = torch.tensor(s[-1], dtype=torch.uint8).to(self.device)
    self.a[self.idx] = a
    self.r[self.idx] = r
    self.t[self.idx] = t * 1
    self.p[self.idx] = p
    self._sumtree.set(self.idx, sampleable)
    self.last_idx = self.idx
    self.idx += 1
    if self.idx >= self.size:
      self.idx = 0
      self.is_filling = False
    if self.is_filling:
      self.maxidx += 1
    if t:
      self.add(s,0,0,0,p,0) # pad end of episode with 1 unsampleable state

  def new_episode(self, s, p):
    # pad beginning of episode with 3 unsampleable states
    for i in range(3):
      self.add(s, 0, 0, 0, p, 0)

  def _idx2xy(self, idx, sidx=None):
    if sidx is None:
      d = tint((-3, -2, -1, 0, 1))  # 4 state history slice + 1 for s'
      sidx = (idx.reshape((idx.shape[0], 1)) + d) % self.maxidx
    return (
        self.s[sidx[:, :4]].float() / 255,
        self.a[idx],
        self.r[idx],
        self.s[sidx[:, 1:]].float() / 255,
        self.t[idx],
        idx,
    )

  def sample(self, n):
    idx = self._sumtree.stratified_sample(n)
    return self._idx2xy(idx)

  def slice_near(self, idx, dist=10, exclude_0=True):
    ar = np.arange(-dist, dist + 1)
    if exclude_0:
      ar = ar[ar != 0]
    sidx = (idx.reshape((-1, 1)) + tint(ar))
    p = self.p[idx]
    ps = self.p[sidx]
    pmask = (p[:, None] == ps).reshape((-1,)).float()
    sidx = sidx.reshape((-1,)) # clamp??
    return self._idx2xy(sidx), pmask

  def get(self, idx):
    return self._idx2xy(idx)

  def in_order_iterate(self, mbsize, until=None):
    if until is None:
      until = self.size
    valid_indices = np.arange(self.size)[self._sumtree.levels[-1] > 0]
    it = 0
    end = 0
    while end < valid_indices.shape[0]:
      end = min(it + mbsize, valid_indices.shape[0])
      if end > until:
        break
      yield self.get(tint(valid_indices[it:end]))
      it += mbsize


class SumTree:

  def __init__(self, rng, size):
    self.rng = rng
    self.nlevels = int(np.ceil(np.log(size) / np.log(2))) + 1
    self.size = size
    self.levels = []
    for i in range(self.nlevels):
      self.levels.append(np.zeros(min(2**i, size), dtype="float32"))

  def sample(self, q=None):
    q = self.rng.random() if q is None else q
    q *= self.levels[0][0]
    s = 0
    for i in range(1, self.nlevels):
      s *= 2
      if self.levels[i][s] < q and self.levels[i][s + 1] > 0:
        q -= self.levels[i][s]
        s += 1
    return s

  def stratified_sample(self, n):
    # As per Schaul et al. (2015)
    return tint([
        self.sample((i + q) / n)
        for i, q in enumerate(self.rng.uniform(0, 1, n))
    ])

  def set(self, idx, p):
    delta = p - self.levels[-1][idx]
    for i in range(self.nlevels - 1, -1, -1):
      self.levels[i][idx] += delta
      idx //= 2

  def getp(self, idx):
    return self.levels[-1][idx]


class PrioritizedExperienceReplay(ReplayBuffer):

  def __init__(self, *a, **kw):
    super().__init__(*a, **kw)
    self.sumtree = SumTree(self.rng, self.size)

  def sample(self, n, near=None, near_dist=5):
    if near is None:
      #try:
      idx = self.sumtree.stratified_sample(n).clamp(4, self.maxidx - 2)
    #except Exception as e:
    #    import pdb
    #    pdb.set_trace()
    else:
      raise ValueError("`near` argument incompatible with this class")
    return self._idx2xy(*self._fix_idx(idx))

  def set_last_priority(self, p):
    """sets the unnormalized priority of the last added example"""
    self.sumtree.set(self.last_idx, p)

  def set_prioties_at(self, ps, idx):
    self.sumtree.set(idx, ps)
    #for i, p in zip(idx, ps):
    #    self.sumtree.set(i, p)


if __name__ == "__main__":

  N = 1000000
  s = SumTree(np.random, 100)
  for i in range(100):
    s.set(i, np.random.random())
  s.sample(1)
  s.sample(1.2)
  x = [s.sample() for i in range(N)]
  u = np.unique(x, return_counts=True)[1] / N
  true_u = s.levels[-1] / np.sum(s.levels[-1])
  print(np.max(abs(u - true_u)))
  assert np.max(abs(u - true_u)) < 1e-3
