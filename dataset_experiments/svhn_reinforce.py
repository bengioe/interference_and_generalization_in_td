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
import torch.nn.functional as F
from torch.distributions import Categorical

class SumLoss(torch.nn.Module):
    def __init__(self):
        super(SumLoss, self).__init__()
    def forward(self, input):
        return input.sum()

sumloss = extend(SumLoss())

if '--cpu' in sys.argv:
    device = torch.device('cpu')
else:
    device = torch.device('cuda')

data_root = 'svhn/'
save_root = 'results/'

train_x = np.load(f'{data_root}/trX.npy').reshape((-1, 32, 32, 3)).transpose(0, 3, 1, 2)
train_y = np.load(f'{data_root}/trY.npy').flatten() - 1
test_x = np.load(f'{data_root}/teX.npy').reshape((-1, 32, 32, 3)).transpose(0, 3, 1, 2)
test_y = np.load(f'{data_root}/teY.npy').flatten() - 1

_train_x = torch.tensor(train_x).float().to(device)
_train_y = torch.tensor(train_y).long().to(device)
test_x = torch.tensor(test_x).float().to(device)
test_y = torch.tensor(test_y).long().to(device)

from cifar_agent_2 import CifarWindowEnv, CifarWindowEnvBatch



def run_exp(meta_seed, nhid, nlayers, n_train_seeds):
    torch.manual_seed(meta_seed)
    np.random.seed(meta_seed)
    gamma = 0.9

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

    env = CifarWindowEnv(_train_x, _train_y)
    test_env = CifarWindowEnvBatch()
    env.step_reward = 0.05
    test_env.step_reward = 0.05

    ##nhid = 32
    act = torch.nn.LeakyReLU()
    #act = torch.nn.Tanh()
    model = torch.nn.Sequential(*([torch.nn.Conv2d(4, nhid, 5, stride=2), act,
                                 torch.nn.Conv2d(nhid, nhid*2, 3), act,
                                 torch.nn.Conv2d(nhid*2, nhid*4, 3), act] +
                                sum([[torch.nn.Conv2d(nhid*4, nhid*4, 3, padding=1), act]
                                     for i in range(nlayers)], []) +
                                [torch.nn.Flatten(),
                                 torch.nn.Linear(nhid*4*10*10, nhid*4), act,
                                 torch.nn.Linear(nhid*4, 14)]))
    model.to(device)
    model.apply(init_weights)
    if 1:
        model = extend(model)
    opt = torch.optim.Adam(model.parameters(), 1e-5)#, weight_decay=1e-5)
    #opt = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9)

    def run_test(X, Y, dataacc=None):
        obs = test_env.reset(X, Y)
        accr = np.zeros(len(X))
        for i in range(test_env.max_steps):
            o = model(obs)
            cls_act = Categorical(logits=o[:, :10]).sample()
            mov_act = Categorical(logits=o[:, 10:]).sample()
            actions = torch.stack([cls_act, mov_act]).data.cpu().numpy()
            obs, r, done, _ = test_env.step(actions)
            accr += r
            if dataacc is not None:
                dataacc.append(obs[np.random.randint(0, len(obs))])
                dataacc.append(obs[np.random.randint(0, len(obs))])
            if done.all():
                break
        return test_env.correct_answers / len(X), test_env.acc_reward

    train_perf = []
    test_perf = []
    all_dots = []
    all_dots_test = []
    tds = []
    qs = []
    xent = torch.nn.CrossEntropyLoss()
    tau = 0.1

    n_rp = 1000
    rp_s = torch.zeros((n_rp, 4, 32, 32), device=device)
    rp_a = torch.zeros((n_rp, 2), device=device, dtype=torch.long)
    rp_g = torch.zeros((n_rp,), device=device)
    rp_idx = 0
    rp_fill = 0

    obs = env.reset(np.random.randint(0, n_train_seeds))
    ntest = 128
    epsilon = 0.9
    ep_reward = 0
    ep_rewards = []
    ep_start = 0
    for i in tqdm(range(200001)):
        epsilon = 0.9 * (1 - min(i, 100000) / 100000) + 0.05
        if not i % 1000:
            t0 = time.time()
            dataacc = []
            with torch.no_grad():
                train_perf.append(run_test(_train_x[:min(ntest, n_train_seeds)], _train_y[:min(ntest, n_train_seeds)]))
                test_perf.append(run_test(test_x[:ntest], test_y[:ntest], dataacc=dataacc))
            print(train_perf[-2:], test_perf[-2:], np.mean(ep_rewards[-50:]), len(ep_rewards))
            if 1:
                t1 = time.time()
                s = rp_s[:128]
                if 1:
                    loss = sumloss(model(s).mean())
                    with backpack(BatchGrad()):
                        loss.backward()
                    all_grads = torch.cat([i.grad_batch.reshape((s.shape[0], -1)) for i in model.parameters()], 1)
                else:
                    all_grads = []
                    for k in range(len(s)):
                        Qsa = model(s[k][None, :])
                        grads = torch.autograd.grad(Qsa.max(), model.parameters())
                        fg = torch.cat([i.reshape((-1,)) for i in grads])
                        all_grads.append(fg)
                dots = []
                for k in range(len(s)):
                    for j in range(k+1, len(s)):
                        dots.append(all_grads[k].dot(all_grads[j]).item())
                all_dots.append(np.float32(dots).mean())
                opt.zero_grad()
                s = torch.stack(dataacc[:128])
                loss = sumloss(model(s).mean())
                with backpack(BatchGrad()):
                    loss.backward()
                all_grads = torch.cat([i.grad_batch.reshape((s.shape[0], -1)) for i in model.parameters()], 1)
                dots = []
                for k in range(len(s)):
                    for j in range(k+1, len(s)):
                        dots.append(all_grads[k].dot(all_grads[j]).item())
                all_dots_test.append(np.float32(dots).mean())
                opt.zero_grad()
                if i and 0:
                    print(i,
                      (cls_pi * torch.log(cls_pi)).mean().item(),
                      (mov_pi * torch.log(mov_pi)).mean().item())
                    print(cls_pi[0].data.cpu().numpy())

        o = model(obs[None, :])
        cls_act = Categorical(logits=o[0, :10]).sample().item()
        #cls_act = env.current_y
        mov_act = Categorical(logits=o[0, 10:]).sample().item()
        action = np.int32([cls_act, mov_act])

        #if np.random.uniform(0,1) < 0.4:
        #    action = env.current_y

        obsp, r, done, _ = env.step(action)
        rp_s[rp_idx] = obs
        rp_a[rp_idx] = torch.tensor(action)
        rp_idx = (rp_idx + 1) % rp_s.shape[0]
        rp_fill += 1
        ep_reward += r
        obs = obsp
        if done:
            rp_g[ep_start:i] = ep_reward
            ep_rewards.append(ep_reward)
            ep_reward = 0
            ep_start = i
            obs = env.reset(np.random.randint(0, n_train_seeds))
            if rp_idx > 250:
                rp_at = rp_idx
                rp_idx = 0
                rp_fill = 0
                s = rp_s[:rp_at]
                a = rp_a[:rp_at]
                g = rp_g[:rp_at]
                o = model(s)
                cls_pi = F.softmax(o[:, :10], 1).clamp(min=1e-5)
                mov_pi = F.softmax(o[:, 10:], 1).clamp(min=1e-5)
                cls_prob = torch.log(cls_pi[torch.arange(len(a)), a[:, 0]])
                mov_prob = torch.log(mov_pi[torch.arange(len(a)), a[:, 1]])
                #import pdb; pdb.set_trace()
                loss = -(g * (cls_prob + mov_prob)).mean()
                loss += 1e-4 * (
                    (cls_pi * torch.log(cls_pi)).sum(1).mean() +
                    (mov_pi * torch.log(mov_pi)).sum(1).mean())
                loss.backward()
                #for p in model.parameters():
                #    p.grad.data.clamp_(-1, 1)
                opt.step()
                opt.zero_grad()




    s = rp_s[:200]
    all_grads = []
    for i in range(len(s)):
        Qsa = model(s[i][None, :])
        grads = torch.autograd.grad(Qsa.max(), model.parameters())
        fg = torch.cat([i.reshape((-1,)) for i in grads], 0)
        all_grads.append(fg)
    dots = []
    cosd = []
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            dots.append(all_grads[i].dot(all_grads[j]).item())
            cosd.append(all_grads[i].dot(all_grads[j]).item() / (np.sqrt((all_grads[i]**2).sum().item()) *
                                                                 np.sqrt((all_grads[j]**2).sum().item())))
    dots = np.float32(dots)
    print(np.mean(train_perf[-5:]), np.mean(test_perf[-5:]))#, all_dots[-5:])

    return {'dots': dots,
            'cosd': cosd,
            'all_dots': all_dots,
            'all_dots_test': all_dots_test,
            'train_perf': train_perf,
            'test_perf': test_perf,
    }



def main():
    for nhid in [8,16,32]:#
        for nlayers in [0,1,2,3]:
            for n_train_seeds in [20, 100, 500, 1000, 5000, 10000, 50000]:#[4,8,16,32,64,128]:
                for meta_seed in [1,2,3]:
                    cfg = {'nhid': nhid,
                           'nlayers': nlayers,
                           'n_train_seeds': n_train_seeds,
                           'meta_seed': meta_seed,
                           'what':'svhn-reinforce-2'}
                    h = hashlib.sha1(bytes(str(sorted(cfg.items())), 'utf8')).hexdigest()
                    path = f'{save_root}/{h}.pkl.gz'
                    if os.path.exists(path):
                        continue
                    print(cfg)
                    open(path,'w').write('touch')
                    results = run_exp(meta_seed, nhid, nlayers, n_train_seeds)
                    with gzip.open(path, 'wb') as f:
                        pickle.dump((cfg, results), f)

if __name__ == '__main__':
    main()
