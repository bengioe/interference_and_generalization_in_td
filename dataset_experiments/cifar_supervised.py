import sys
import torch
import numpy as np
import matplotlib.pyplot as pp
import copy
import pickle
import gzip
import hashlib
import os.path
import sklearn.datasets
from tqdm import tqdm
from backpack import backpack, extend
from backpack.extensions import BatchGrad


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
xs, ys = [], []
for i in range(5):
    with open(f'cifar-10-batches-py/data_batch_{i+1}', 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
        xs.append(d[b'data'])
        ys.append(d[b'labels'])
train_x = np.concatenate(xs, 0).reshape((-1, 3, 32, 32)) / 255
train_y = np.concatenate(ys, 0)
with open(f'cifar-10-batches-py/test_batch', 'rb') as fo:
    d = pickle.load(fo, encoding='bytes')
    test_x = d[b'data'].reshape((-1, 3, 32, 32)) / 255
    test_y = d[b'labels']
_train_x = torch.tensor(train_x).float().to(device)
_train_y = torch.tensor(train_y).long().to(device)
test_x = torch.tensor(test_x).float().to(device)
test_y = torch.tensor(test_y).long().to(device)



def run_exp(meta_seed, nhid, nlayers, n_train_seeds):
    torch.manual_seed(meta_seed)
    np.random.seed(meta_seed)
    gamma = 0.9
    train_x = _train_x[:n_train_seeds]
    train_y = _train_y[:n_train_seeds]

    ##nhid = 32
    act = torch.nn.LeakyReLU()
    #act = torch.nn.Tanh()
    model = torch.nn.Sequential(*([torch.nn.Conv2d(3, nhid, 5, stride=2), act,
                                 torch.nn.Conv2d(nhid, nhid*2, 3), act,
                                 torch.nn.Conv2d(nhid*2, nhid*4, 3), act] +
                                sum([[torch.nn.Conv2d(nhid*4, nhid*4, 3, padding=1), act]
                                     for i in range(nlayers)], []) +
                                [torch.nn.Flatten(),
                                 torch.nn.Linear(nhid*4*10*10, nhid), act,
                                 torch.nn.Linear(nhid, 10)]))
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            k = np.sqrt(6 / (np.sum(m.weight.shape)))
            m.weight.data.uniform_(-k, k)
            m.bias.data.fill_(0)
    model.to(device)
    model.apply(init_weights)
    if 0:
        model = extend(model)
    opt = torch.optim.Adam(model.parameters(), 1e-3)#, weight_decay=1e-5)


    def compute_acc_mb(X, Y, mbs=1024):
        N = len(X)
        i = 0
        tot = 0
        while i < N:
            x = X[i:i+mbs]
            y = Y[i:i+mbs]
            tot += np.sum(model(x).argmax(1).cpu().data.numpy() == y.cpu().data.numpy())
            i += mbs
        return tot / N

    train_perf = []
    test_perf = []
    all_dots = []
    xent = torch.nn.CrossEntropyLoss()

    for i in tqdm(range(1000)):
        if not i % 20:
            train_perf.append(compute_acc_mb(train_x, train_y))
            test_perf.append(compute_acc_mb(test_x, test_y))
            s = train_x[:96]
            if 0:
                loss = sumloss(model(s).max(1).values)
                with backpack(BatchGrad()):
                    loss.backward()
                all_grads = torch.cat([i.grad_batch.reshape((mbs, -1)) for i in model.parameters()], 1)
            all_grads = []
            for i in range(len(s)):
                Qsa = model(s[i][None, :])
                grads = torch.autograd.grad(Qsa.max(), model.parameters())
                fg = torch.cat([i.reshape((-1,)) for i in grads])
                all_grads.append(fg)
            dots = []
            cosd = []
            cosd = []
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    dots.append(all_grads[i].dot(all_grads[j]).item())
            all_dots.append(np.float32(dots).mean())
            opt.zero_grad()
        mbidx = np.random.randint(0, len(train_x), 32)
        x = train_x[mbidx]
        y = train_y[mbidx]
        pred = model(x)
        loss = xent(pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()



    s = train_x[:200]
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
    print(train_perf[-5:], test_perf[-5:], all_dots[-5:])

    return {'dots': dots,
            'cosd': cosd,
            'all_dots': all_dots,
            'train_perf': train_perf,
            'test_perf': test_perf,
    }



def main():

    for nhid in [16,32,64]:#
        for nlayers in [1,0,2,3]:
            for n_train_seeds in [20, 100, 500, 1000, 5000, 10000, 50000]:#[4,8,16,32,64,128]:
                for meta_seed in [1,2,3]:
                    cfg = {'nhid': nhid,
                           'nlayers': nlayers,
                           'n_train_seeds': n_train_seeds,
                           'meta_seed': meta_seed,
                           'what':'cifar-3'}
                    print(cfg)
                    h = hashlib.sha1(bytes(str(sorted(cfg.items())), 'utf8')).hexdigest()
                    path = f'results/{h}.pkl.gz'
                    if os.path.exists(path):
                        continue
                    open(path,'w').write('touch')
                    results = run_exp(meta_seed, nhid, nlayers, n_train_seeds)
                    with gzip.open(path, 'wb') as f:
                        pickle.dump((cfg, results), f)


if __name__ == '__main__':
    main()
