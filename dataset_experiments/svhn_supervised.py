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
save_root = 'results'

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
                                 torch.nn.Linear(nhid*4*10*10, nhid*4), act,
                                 torch.nn.Linear(nhid*4, 10)]))
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            k = np.sqrt(6 / (np.sum(m.weight.shape)))
            m.weight.data.uniform_(-k, k)
            m.bias.data.fill_(0)
    model.to(device)
    model.apply(init_weights)
    if 1:
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
    all_jdots = []
    all_cosd = []
    xent = torch.nn.CrossEntropyLoss()
    xent_xt = extend(torch.nn.CrossEntropyLoss())
    for i in tqdm(range(1000)):
        if not i % 50:
            train_perf.append(compute_acc_mb(train_x, train_y))
            test_perf.append(compute_acc_mb(test_x, test_y))
            #print(train_perf[-1], test_perf[-1])
            strain = train_x[:96]
            loss = sumloss(model(strain).max(1).values)
            with backpack(BatchGrad()):
                loss.backward()
            train_grads = torch.cat([i.grad_batch.reshape((strain.shape[0], -1)) for i in model.parameters()], 1)
            opt.zero_grad()

            stest = test_x[:96]
            loss = sumloss(model(stest).max(1).values)
            with backpack(BatchGrad()):
                loss.backward()
            test_grads = torch.cat([i.grad_batch.reshape((stest.shape[0], -1)) for i in model.parameters()], 1)
            opt.zero_grad()

            trtr = []
            trte = []
            tete = []
            cosd = []
            for i in range(len(strain)):
                for j in range(i+1, len(strain)):
                    trtr.append(train_grads[i].dot(train_grads[j]).item())
                    cosd.append(trtr[-1] / (torch.sqrt((train_grads[i]**2).sum()) / torch.sqrt((train_grads[j]**2).sum())).item())
                for j in range(len(stest)):
                    trte.append(train_grads[i].dot(test_grads[j]))
            for i in range(len(stest)):
                for j in range(i+1, len(stest)):
                    tete.append(test_grads[i].dot(test_grads[j]).item())
            all_dots.append(list(map(np.float32, [trtr, trte, tete])))
            all_cosd.append(np.float32(cosd))


            strain = train_x[:96]
            loss = xent_xt(model(strain), train_y[:96])
            with backpack(BatchGrad()):
                loss.backward()
            train_grads = torch.cat([i.grad_batch.reshape((strain.shape[0], -1)) for i in model.parameters()], 1)
            opt.zero_grad()

            stest = test_x[:96]
            loss = xent_xt(model(stest), test_y[:96])
            with backpack(BatchGrad()):
                loss.backward()
            test_grads = torch.cat([i.grad_batch.reshape((stest.shape[0], -1)) for i in model.parameters()], 1)
            opt.zero_grad()

            dots = []
            trtr = []
            trte = []
            tete = []
            for i in range(len(strain)):
                for j in range(i+1, len(strain)):
                    trtr.append(train_grads[i].dot(train_grads[j]).item())
                for j in range(len(stest)):
                    trte.append(train_grads[i].dot(test_grads[j]))
            for i in range(len(stest)):
                for j in range(i+1, len(stest)):
                    tete.append(test_grads[i].dot(test_grads[j]).item())
            all_jdots.append(list(map(np.float32, [trtr, trte, tete])))

        mbidx = np.random.randint(0, len(train_x), 32)
        x = train_x[mbidx]
        y = train_y[mbidx]
        pred = model(x)
        loss = xent(pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()


    print(train_perf[-5:], test_perf[-5:], all_dots[-5:])

    return {'all_dots': all_dots,
            'all_jdots': all_jdots,
            'all_cosd': all_cosd,
            'train_perf': train_perf,
            'test_perf': test_perf,
            'params': [i.cpu().data.numpy() for i in model.parameters()],
    }



def main():

    for nhid in [8,16,32]:#
        for nlayers in [0,1,2,3]:
            for n_train_seeds in [20, 100, 250, 500, 1000, 5000, 10000, 50000]:#[4,8,16,32,64,128]:
                for meta_seed in [1,2,3,4]:
                    cfg = {'nhid': nhid,
                           'nlayers': nlayers,
                           'n_train_seeds': n_train_seeds,
                           'meta_seed': meta_seed,
                           'what':'svhn-2'}
                    print(cfg)
                    h = hashlib.sha1(bytes(str(sorted(cfg.items())), 'utf8')).hexdigest()
                    path = f'{save_root}/{h}.pkl.gz'
                    if os.path.exists(path):
                        continue
                    open(path,'w').write('touch')
                    results = run_exp(meta_seed, nhid, nlayers, n_train_seeds)
                    with gzip.open(path, 'wb') as f:
                        pickle.dump((cfg, results), f)


if __name__ == '__main__':
    main()
