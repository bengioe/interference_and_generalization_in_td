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

from env import WindowEnv, WindowEnvBatch


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

    env = WindowEnv(_train_x, _train_y)
    test_env = WindowEnvBatch()
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
    target = copy.deepcopy(model)
    if 1:
        model = extend(model)
    opt = torch.optim.Adam(model.parameters(), 2.5e-4)#, weight_decay=1e-5)

    n_rp_test = 128
    rpt_s = torch.zeros((n_rp_test, 4, 32, 32), device=device)
    rpt_a = torch.zeros((n_rp_test, 2), device=device, dtype=torch.long)
    rpt_r = torch.zeros((n_rp_test,), device=device)
    rpt_z = torch.zeros((n_rp_test, 4, 32, 32), device=device)
    rpt_t = torch.zeros((n_rp_test,), device=device)
    rpt_idx = [0]

    def run_test(X, Y, dataacc=False):
        obs = test_env.reset(X, Y)
        if dataacc: rpt_idx[0] = 0
        for i in range(test_env.max_steps):
            Qs = model(obs)
            actions = [Qs[:, :10].argmax(1).data.cpu().numpy(),
                       Qs[:, 10:].argmax(1).data.cpu().numpy()]
            obsp, r, done, _ = test_env.step(actions)
            for i in range(2):
                if dataacc and rpt_idx[0] < n_rp_test:
                    u = np.random.randint(0, len(obs))
                    rpt_s[rpt_idx[0]] = obs[u]
                    rpt_a[rpt_idx[0]] = torch.tensor([actions[0][u], actions[1][u]])
                    rpt_r[rpt_idx[0]] = r[u]
                    rpt_z[rpt_idx[0]] = obsp[u]
                    rpt_t[rpt_idx[0]] = 1 - done[u]
                    rpt_idx[0] += 1
            obs = obsp
            if done.all():
                break
        return test_env.correct_answers / len(X), test_env.acc_reward

    train_perf = []
    test_perf = []
    all_dots = []
    all_jdots = []
    all_dots_test = []
    all_cosd = []
    tds = []
    qs = []
    xent = torch.nn.CrossEntropyLoss()
    tau = 0.1

    n_rp = 100000
    rp_s = torch.zeros((n_rp, 4, 32, 32), device=device)
    rp_a = torch.zeros((n_rp, 2), device=device, dtype=torch.long)
    rp_r = torch.zeros((n_rp,), device=device)
    rp_z = torch.zeros((n_rp, 4, 32, 32), device=device)
    rp_t = torch.zeros((n_rp,), device=device)
    rp_idx = 0
    rp_fill = 0

    obs = env.reset(np.random.randint(0, n_train_seeds))
    ntest = 128
    epsilon = 0.9
    ep_reward = 0
    ep_rewards = []
    for i in tqdm(range(200001)):
        epsilon = 0.9 * (1 - min(i, 100000) / 100000) + 0.05
        if not i % 10000:
            t0 = time.time()
            with torch.no_grad():
                train_perf.append(run_test(_train_x[:min(ntest, n_train_seeds)], _train_y[:min(ntest, n_train_seeds)]))
                test_perf.append(run_test(test_x[:ntest], test_y[:ntest], dataacc=True))
            #print(train_perf[-2:], test_perf[-2:], np.mean(ep_rewards[-50:]), len(ep_rewards))
            if i:
                t1 = time.time()
                mbidx = np.random.randint(0, min(len(rp_s), rp_fill), 128)
                s = rp_s[mbidx]
                loss = sumloss(model(s).max(1).values)
                with backpack(BatchGrad()):
                    loss.backward()
                train_grads = torch.cat([i.grad_batch.reshape((s.shape[0], -1)) for i in model.parameters()], 1)
                opt.zero_grad()
                s = rpt_s[:rpt_idx[0]]
                loss = sumloss(model(s).max(1).values)
                with backpack(BatchGrad()):
                    loss.backward()
                test_grads = torch.cat([i.grad_batch.reshape((s.shape[0], -1)) for i in model.parameters()], 1)
                opt.zero_grad()

                trtr = []
                trte = []
                tete = []
                cosd = []
                for i in range(128):
                    for j in range(i+1, 128):
                        trtr.append(train_grads[i].dot(train_grads[j]).item())
                        cosd.append(trtr[-1] / (torch.sqrt((train_grads[i]**2).sum()) / torch.sqrt((train_grads[j]**2).sum())).item())
                    for j in range(rpt_idx[0]):
                        trte.append(train_grads[i].dot(test_grads[j]))
                for i in range(rpt_idx[0]):
                    for j in range(i+1, rpt_idx[0]):
                        tete.append(test_grads[i].dot(test_grads[j]).item())
                all_dots.append(list(map(np.float32, [trtr, trte, tete])))
                all_cosd.append(np.float32(cosd))


                s = rp_s[mbidx]
                a = rp_a[mbidx]
                r = rp_r[mbidx]
                z = rp_z[mbidx]
                t = rp_t[mbidx]
                with torch.no_grad():
                    Qp = target(z)
                    vp1 = Qp[:, :10].max(1).values
                    vp2 = Qp[:, 10:].max(1).values
                Q = model(s)
                v1 = Q[np.arange(len(a)), a[:, 0]]
                v2 = Q[np.arange(len(a)), a[:, 1] + 10]
                td1 = v1 - (r + gamma * vp1 * t)
                td2 = v2 - (r + gamma * vp2 * t)
                loss = torch.min(td1**2, abs(td1)) / 128
                loss += torch.min(td2**2, abs(td2)) / 128
                loss = sumloss(loss)
                with backpack(BatchGrad()):
                    loss.backward()
                train_grads = torch.cat([i.grad_batch.reshape((s.shape[0], -1)) for i in model.parameters()], 1)
                opt.zero_grad()


                s = rpt_s[:rpt_idx[0]]
                a = rpt_a[:rpt_idx[0]]
                r = rpt_r[:rpt_idx[0]]
                z = rpt_z[:rpt_idx[0]]
                t = rpt_t[:rpt_idx[0]]
                with torch.no_grad():
                    Qp = target(z)
                    vp1 = Qp[:, :10].max(1).values
                    vp2 = Qp[:, 10:].max(1).values
                Q = model(s)
                v1 = Q[np.arange(len(a)), a[:, 0]]
                v2 = Q[np.arange(len(a)), a[:, 1] + 10]
                td1 = v1 - (r + gamma * vp1 * t)
                td2 = v2 - (r + gamma * vp2 * t)
                loss = torch.min(td1**2, abs(td1)) / 128
                loss += torch.min(td2**2, abs(td2)) / 128
                loss = sumloss(loss)
                with backpack(BatchGrad()):
                    loss.backward()
                test_grads = torch.cat([i.grad_batch.reshape((s.shape[0], -1)) for i in model.parameters()], 1)
                opt.zero_grad()

                trtr = []
                trte = []
                tete = []
                for i in range(128):
                    for j in range(i+1, 128):
                        trtr.append(train_grads[i].dot(train_grads[j]).item())
                    for j in range(rpt_idx[0]):
                        trte.append(train_grads[i].dot(test_grads[j]))
                for i in range(rpt_idx[0]):
                    for j in range(i+1, rpt_idx[0]):
                        tete.append(test_grads[i].dot(test_grads[j]).item())
                all_jdots.append(list(map(np.float32, [trtr, trte, tete])))


        if np.random.uniform(0,1) < epsilon:
            action = [np.random.randint(0, 10),
                      np.random.randint(0, 4)]
        else:
            Qs = model(obs[None, :])[0]
            action = [Qs[:10].argmax().item(), Qs[10:].argmax().item()]

        #if np.random.uniform(0,1) < 0.4:
        #    action = env.current_y

        obsp, r, done, _ = env.step(action)
        rp_s[rp_idx] = obs
        rp_a[rp_idx] = torch.tensor(action)
        rp_r[rp_idx] = r
        rp_z[rp_idx] = obsp
        rp_t[rp_idx] = 1 - done
        rp_idx = (rp_idx + 1) % rp_s.shape[0]
        rp_fill += 1
        ep_reward += r
        obs = obsp
        if done:
            ep_rewards.append(ep_reward)
            ep_reward = 0
            obs = env.reset(np.random.randint(0, n_train_seeds))

        if i > 5000 and not i % 2:
            mbidx = np.random.randint(0, min(len(rp_s), rp_fill), 128)
            s = rp_s[mbidx]
            a = rp_a[mbidx]
            r = rp_r[mbidx]
            z = rp_z[mbidx]
            t = rp_t[mbidx]
            with torch.no_grad():
                Qp = target(z)
                vp1 = Qp[:, :10].max(1).values
                vp2 = Qp[:, 10:].max(1).values
            Q = model(s)
            v1 = Q[np.arange(len(a)), a[:, 0]]
            v2 = Q[np.arange(len(a)), a[:, 1] + 10]
            td1 = v1 - (r + gamma * vp1 * t)
            td2 = v2 - (r + gamma * vp2 * t)
            loss = torch.min(td1**2, abs(td1)).mean()
            loss += torch.min(td2**2, abs(td2)).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            for target_param, param in zip(target.parameters(), model.parameters()):
                target_param.data.copy_(tau * param + (1 - tau) * target_param)



    return {'all_dots': all_dots,
            'all_jdots': all_jdots,
            'all_cosd': all_cosd,
            'train_perf': train_perf,
            'test_perf': test_perf,
    }



def main():
    for nhid in [8, 16, 32]:#
        for nlayers in [1,0,2,3]:
            for n_train_seeds in [20, 100, 500, 1000, 5000, 10000, 50000]:#[4,8,16,32,64,128]:
                for meta_seed in [1,2,3]:
                    cfg = {'nhid': nhid,
                           'nlayers': nlayers,
                           'n_train_seeds': n_train_seeds,
                           'meta_seed': meta_seed,
                           'what':'svhn-agent-2'}
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
