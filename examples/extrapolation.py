# Example applied to 1d extrapolation
# In the spirit of Figure 4 of https://arxiv.org/abs/2307.00835

# Install with pip install engression-pytorch[examples]

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import matplotlib.pyplot as plt

from engression_pytorch import energy_score, gConcat


# data
g_stars = {
    'softplus': F.softplus,
    'square': lambda x: (torch.max(x, torch.tensor(0.0)) ** 2) / 2,
    'cubic': lambda x: (x ** 3) / 3,
    'log': lambda x: (x/3 + math.log(3) - 2/3) * (x <= 2) + (torch.log(1 + x * (x > 2))) * (x > 2) 
}

x_train_lims = {
    'softplus': (-2, 2),
    'square': (0, 2),
    'cubic': (-2, 2),
    'log': (0, 2)
}

x_test_lims = {
    'softplus': (-2, 8),
    'square': (0, 6),
    'cubic': (-2, 6),
    'log': (0, 10)
}

def train_data(n, input_dim, type = 'softplus', pre_additive_noise = True):

    g_star = g_stars[type]

    a, b = x_train_lims[type]
    X = torch.rand((n, input_dim)) * (b - a) + a # Unif(a, b)

    sd = 1 if type != 'cubic' else 1.1
    eta = torch.randn((n, input_dim)) * sd
    
    if pre_additive_noise:
        Y = g_star(X + eta)
    else:
        Y = g_star(X) + eta

    return X, Y

def main():
        
    # example
    name = 'softplus'
    X, Y = train_data(50_000, 1, type = name, pre_additive_noise = True)
    ds = torch.utils.data.TensorDataset(X, Y)
    dl = torch.utils.data.DataLoader(ds, batch_size = 128, shuffle = True)

    # define model
    input_dim, hidden_dim, output_dim = 1, 128, 1
    noise_dim = 64

    model = nn.Sequential(
        nn.Linear(input_dim + noise_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )

    g = gConcat(model = model, noise_dim = noise_dim, m_train = 16, noise_type = 'uniform')


    # train
    opt = torch.optim.Adam(g.parameters())
    g.train()

    for x, y in dl:
        opt.zero_grad()
        preds = g(x)
        loss = energy_score(y, preds)
        loss.backward()
        opt.step()

        # print(loss.item())

    # test extrapolation
    with torch.no_grad():

        g.eval()

        a, b = x_test_lims[name]
        t = torch.linspace(a, b, 25)[:, None]

        plt.scatter(X[:1000], Y[:1000], s = 1, alpha = 0.5, c = 'gray')
        plt.plot(t, g(t).mean(1).flatten(), c = 'tab:blue', label = 'predicted median')
        plt.plot(t, g_stars[name](t), c = 'tab:red', label = 'true median')
        plt.legend()
        plt.xlabel('x'); plt.ylabel('y')

        plt.tight_layout()
        plt.savefig('example.png')
        print('saved figure to example.png')

if __name__ == '__main__':
    main()