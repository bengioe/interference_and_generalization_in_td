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

def run_exp(meta_seed, nhid, n_train_seeds):
    torch.manual_seed(meta_seed)
    np.random.seed(meta_seed)
    gamma = 0.9
    ##nhid = 32
    act = torch.nn.LeakyReLU()
    #act = torch.nn.Tanh()
    model = torch.nn.Sequential(torch.nn.Linear(8, nhid), act,
                                torch.nn.Linear(nhid, nhid), act,
                                torch.nn.Linear(nhid, nhid), act,
                                torch.nn.Linear(nhid, 1))
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            k = np.sqrt(6 / (np.sum(m.weight.shape)))
            m.weight.data.uniform_(-k, k)
            m.bias.data.fill_(0)
    model.apply(init_weights)
    opt = torch.optim.Adam(model.parameters(), 1e-3)#, weight_decay=1e-5)


    #X_, Y_ = load_boston(return_X_y=True)
    X_, Y_  = sklearn.datasets.fetch_california_housing('~/data/',return_X_y=True)
    ntest = int(0.05 * X_.shape[0])
    ntrain = X_.shape[0] - ntest
    shuffleidx = np.arange(len(X_))
    np.random.seed(1283)
    np.random.shuffle(shuffleidx)
    np.random.seed(meta_seed)
    X_ = X_[shuffleidx]
    Y_ = Y_[shuffleidx]


    X_train = X_[:ntrain]
    Y_train = Y_[:ntrain]
    X_test = X_[ntrain:]
    Y_test = Y_[ntrain:]
    xta, xtb = X_train.min(), X_train.max()
    yta, ytb = Y_train.min(), Y_train.max()
    X_train = (X_train-xta)/(xtb-xta)
    X_test = (X_test-xta)/(xtb-xta)
    Y_train = (Y_train-yta)/(ytb-yta)
    Y_test = (Y_test-yta)/(ytb-yta)

    train_x = torch.tensor(X_train[:n_train_seeds]).float()
    train_y = torch.tensor(Y_train[:n_train_seeds]).float()
    test_x = torch.tensor(X_test).float()
    test_y = torch.tensor(Y_test).float()


    train_perf = []
    test_perf = []
    all_dots = []
    xent = torch.nn.CrossEntropyLoss()

    for i in range(1000):
        if not i % 5:
            train_perf.append(np.mean((model(train_x).data.numpy() - train_y.data.numpy())**2))
            test_perf.append(np.mean((model(test_x).data.numpy() - test_y.data.numpy())**2))
            s = train_x[:200]
            all_grads = []
            for i in range(len(s)):
                Qsa = model(s[i][None, :])
                grads = torch.autograd.grad(Qsa, model.parameters())
                fg = np.concatenate([i.reshape((-1,)).data.numpy() for i in grads])
                all_grads.append(fg)
            dots = []
            cosd = []
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    dots.append(all_grads[i].dot(all_grads[j]))
            all_dots.append(np.float32(dots).mean())
        mbidx = np.random.randint(0, len(train_x), 32)
        x = train_x[mbidx]
        y = train_y[mbidx]
        pred = model(x)
        loss = ((pred-y)**2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()



    s = train_x[:200]
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
    print(train_perf[-5:], test_perf[-5:], all_dots[-5:])

    return {'dots': dots,
            'cosd': cosd,
            'all_dots': all_dots,
            'train_perf': train_perf,
            'test_perf': test_perf,
    }



def main():

    for nhid in [16,32,64,128]:#
        for n_train_seeds in [20, 100, 500, 1000, 5000, 10000]:#[4,8,16,32,64,128]:
            for meta_seed in [0,1,2,3]:
                cfg = {'nhid': nhid,
                       'n_train_seeds': n_train_seeds,
                       'meta_seed': meta_seed,
                       'what':'calif-2'}
                print(cfg)
                h = hashlib.sha1(bytes(str(sorted(cfg.items())), 'utf8')).hexdigest()
                path = f'results/{h}.pkl.gz'

                if os.path.exists(path):
                    continue
                open(path, 'w').write('touch')
                results = run_exp(meta_seed, nhid, n_train_seeds)
                with gzip.open(path, 'wb') as f:
                    pickle.dump((cfg, results), f)


if __name__ == '__main__':
    main()
