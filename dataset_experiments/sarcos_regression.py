import torch
import numpy as np
import matplotlib.pyplot as pp
import copy
import pickle
import gzip
import hashlib
import os.path
import sklearn.datasets
from sklearn.datasets import load_boston
import scipy.io
from tqdm import tqdm

def run_exp(meta_seed, nhid, nlayers, n_train_seeds):
    torch.manual_seed(meta_seed)
    np.random.seed(meta_seed)
    gamma = 0.9
    ##nhid = 32
    act = torch.nn.LeakyReLU()
    #act = torch.nn.Tanh()
    model = torch.nn.Sequential(*([torch.nn.Linear(21, nhid), act] +
                                  sum([[torch.nn.Linear(nhid, nhid), act]
                                       for i in range(nlayers)],[]) +
                                  [torch.nn.Linear(nhid, 7)]))
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            k = np.sqrt(6 / (np.sum(m.weight.shape)))
            m.weight.data.uniform_(-k, k)
            m.bias.data.fill_(0)
    model.apply(init_weights)
    opt = torch.optim.Adam(model.parameters(), 1e-3)#, weight_decay=1e-5)

    data = scipy.io.loadmat('../../data/sarcos_inv.mat')
    train_x, train_y = data['sarcos_inv'][:, :21], data['sarcos_inv'][:, 21:]
    data = scipy.io.loadmat('../../data/sarcos_inv_test.mat')
    yvar = torch.tensor(train_y.var(0)).float()
    test_x, test_y = data['sarcos_inv_test'][:, :21], data['sarcos_inv_test'][:, 21:]
    train_x = torch.tensor(train_x[:n_train_seeds]).float()
    train_y = torch.tensor(train_y[:n_train_seeds]).float()# / yvar[None, :]
    test_x = torch.tensor(test_x).float()
    test_y = torch.tensor(test_y).float()

    #print(yvar)
    train_perf = []
    test_perf = []
    all_dots = []
    all_dots_test = []

    def compute_acc_mb(X, Y, mbs=1024):
        N = len(X)
        i = 0
        tot = 0
        totx = 0
        while i < N:
            x = X[i:i+mbs]
            y = Y[i:i+mbs]
            tot += ((model(x)-y)**2).mean(1).sum().item()
            totx += ((model(x)-y)**2 / yvar[None,:]).mean(1).sum().item()
            i += mbs
        return tot / N, totx / N

    for i in tqdm(range(5000), leave=False):
        if not i % 250:
            train_perf.append(compute_acc_mb(train_x, train_y))
            test_perf.append(compute_acc_mb(test_x, test_y))

            s = train_x[:128]
            all_grads = []
            for i in range(len(s)):
                Qsa = model(s[i][None, :])
                grads = torch.autograd.grad(Qsa.mean(), model.parameters())
                fg = np.concatenate([i.reshape((-1,)).data.numpy() for i in grads])
                all_grads.append(fg)
            dots = []
            cosd = []
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    dots.append(all_grads[i].dot(all_grads[j]))
            all_dots.append(np.float32(dots).mean())

            s = test_x[:128]
            all_grads = []
            for i in range(len(s)):
                Qsa = model(s[i][None, :])
                grads = torch.autograd.grad(Qsa.mean(), model.parameters())
                fg = np.concatenate([i.reshape((-1,)).data.numpy() for i in grads])
                all_grads.append(fg)
            dots = []
            cosd = []
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    dots.append(all_grads[i].dot(all_grads[j]))
            all_dots_test.append(np.float32(dots).mean())
        mbidx = np.random.randint(0, len(train_x), 32)
        x = train_x[mbidx]
        y = train_y[mbidx]
        pred = model(x)
        loss = ((pred - y)**2).mean(1).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()



    s = train_x[:128]
    all_grads = []
    for i in range(len(s)):
        Qsa = model(s[i][None, :])
        grads = torch.autograd.grad(Qsa.max(), model.parameters())
        fg = np.concatenate([i.reshape((-1,)).data.numpy() for i in grads])
        all_grads.append(fg)
    dots = []
    cosd = []
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            dots.append(all_grads[i].dot(all_grads[j]))
            cosd.append(all_grads[i].dot(all_grads[j]) / (np.sqrt((all_grads[i]**2).sum()) *
                                                          np.sqrt((all_grads[j]**2).sum())))
    dots = np.float32(dots)
    #print(train_perf[-5:], test_perf[-5:], all_dots[-5:])

    return {'dots': dots,
            'cosd': cosd,
            'all_dots': all_dots,
            'all_dots_test': all_dots_test,
            'train_perf': train_perf,
            'test_perf': test_perf,
    }



def main():

    cfgs = []
    for nhid in [16,32,64,128,256]:#
        for nlayers in [0,1,2,3]:
            for n_train_seeds in [20, 100, 500, 1000, 5000, 10000, 44484]:#[4,8,16,32,64,128]:
                for meta_seed in [0,1,2]:
                    cfg = {'nhid': nhid,
                           'n_train_seeds': n_train_seeds,
                           'meta_seed': meta_seed,
                           'nlayers': nlayers,
                           'what':'sarcos-2'}
                    cfgs.append(cfg)
    for cfg in tqdm(cfgs):
        #print(cfg)
        h = hashlib.sha1(bytes(str(sorted(cfg.items())), 'utf8')).hexdigest()
        path = f'results/{h}.pkl.gz'
        if os.path.exists(path):
            continue
        open(path, 'w').write('touch')
        results = run_exp(cfg['meta_seed'], cfg['nhid'], cfg['nlayers'], cfg['n_train_seeds'])
        with gzip.open(path, 'wb') as f:
            pickle.dump((cfg, results), f)


if __name__ == '__main__':
    main()
