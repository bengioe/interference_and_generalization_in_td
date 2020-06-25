import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


_default_act = F.leaky_relu
_device_holder = ["cpu"]


def init_weight(w):
  """set weight data in-place"""
  shape = w.shape
  if len(shape) == 4:
    i, o, u, v = shape
    k = np.sqrt(6 / (i * u * v + o * u * v))
    w.data.uniform_(-k, k)
  elif len(shape) == 2:
    k = np.sqrt(6 / sum(shape))
    w.data.uniform_(-k, k)
  elif len(shape) == 1:
    w.data.zero_()


def make_weight(shape):
  w = torch.empty(shape, dtype=torch.float, requires_grad=True)
  init_weight(w)
  w.data = w.data.to(_device_holder[0])
  return w


def layer(*s):
  """Returns a decorator that makes weights according to `s`."""
  make_weights = lambda a: [make_weight([a[j] for j in i]) for i in s]
  return lambda l: (lambda *a, **k:
                    (lambda x, w: l(x, w, **k), make_weights(a)))



def set_device(which):
  _device_holder[0] = which


def get_device():
  return _device_holder[0]


tf = lambda x: torch.tensor(x).float().to(_device_holder[0])
tint = lambda x: torch.tensor(x).long().to(_device_holder[0])


@layer((1, 0, 2, 2), (1,))
def conv2d(x, W, **k):
  w, b = next(W), next(W)
  a = k.pop("act", _default_act)
  return a(F.conv2d(x, w, b, **{"padding": w.shape[2] // 2, **k}))


@layer((1, 0), (1,))
def hidden(x, w):
  return _default_act(F.linear(x, next(w), next(w)))


@layer((1, 0), (1,))
def linear(x, w):
  return F.linear(x, next(w), next(w))


@layer()
def leaky_relu(x, w):
  return F.leaky_relu(x)


@layer()
def tanh(x, w):
  return torch.tanh(x)


@layer()
def callback(x, w, **k):
  return k["l"](x)


@layer()
def softmax(x, w):
  return torch.nn.functional.softmax(x, -1)


@layer()
def mean_pool_2d(x, w):
  return x.mean(dim=2).mean(dim=2)


@layer()
def dropout(x, w, **k):
  rate = k.pop("rate", 0.5)
  p = (torch.empty_like(x).uniform_(0, 1).to(_device_holder[0]) > rate).float()
  return x * p


@layer()
def flatten(x, w, **k):
  return x.reshape((x.shape[0], np.prod(x.shape[1:])))


def build(*layers):
  arch, ws = list(zip(*layers))
  ws = [i for l in ws for i in l]
  applyf = lambda x, wi, arch: reduce(lambda u, f: f(u, wi), arch, x)
  semiapply_ = lambda x, wi, n: (
      applyf(x, wi, arch[:n]),
      lambda x: applyf(x, wi, arch[n:]),
  )
  semiapply = lambda x, w=ws, n=len(arch): semiapply_(x, iter(w), n)
  model = lambda x, w=ws: applyf(x, iter(w), arch)
  return arch, ws, model, semiapply


def gradient_step(loss, weights, alpha, detach=True):
  grads = torch.autograd.grad(
      loss, weights, create_graph=True, allow_unused=True)
  return [
      p - alpha * (gp.detach() if detach else gp) if gp is not None else p
      for p, gp in zip(weights, grads)
  ]


def one_hot(y, n=10):
  return torch.zeros((y.shape[0], n),
                     device=y.device).scatter(1, y.unsqueeze(1), 1)
