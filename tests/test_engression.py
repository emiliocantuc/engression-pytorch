import torch
import torch.nn as nn
import torch.nn.functional as F

from engression_pytorch import EnergyScoreLoss, gSampler, gConcat, gAdd

import os, pickle
from types import SimpleNamespace

import pytest


# Probably overkill, but learning
@pytest.mark.parametrize('noise_type', ['normal', 'uniform', 'laplace'])
@pytest.mark.parametrize('merge_mode', ['concat', 'add', 'multiply'])
@pytest.mark.parametrize('beta', [0.0, 1.0, 2.0])
@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('lamb', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('ret_cmps', [False, True])
def test_readme(noise_type, merge_mode, beta, p, lamb, ret_cmps):

    batch_size, input_dim, out_dim = 32, 1, 1
    noise_dim = 100 if merge_mode == 'concat' else 0

    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, out_dim)

    model = nn.Linear(input_dim + noise_dim, out_dim)

    g = gSampler(
        model = model,
        noise_dim = noise_dim if merge_mode == 'concat' else None,
        noise_type = noise_type,
        noise_scale = 1.0,
        merge_mode = merge_mode,
        m_train = 2, 
        m_eval = 512,
    )


    g.train()       # change m to m_train
    preds = g(x)    # (batch_size, m_train, output_dim)
    assert preds.shape == (batch_size, 2, out_dim)

    # loss = energy_score(y, preds, beta = 1.0, p = 2)
    loss = EnergyScoreLoss(beta = beta, p = p, lamb = lamb, return_components = ret_cmps)(y, preds)
    if ret_cmps: loss = loss[0]
    loss.backward()

    g.eval()        # changes m to m_eval
    sample = g(x)   # (batch_size, m_eval, output_dim)
    assert sample.shape == (batch_size, 512, out_dim)



@pytest.mark.parametrize('c', [gSampler, gConcat, gAdd])
def test_pickle(c, tmp_path):

    g = c(model = nn.Linear(1, 2), m_train = 2)

    # pickle
    with open(tmp_path / 'g.p', 'wb') as f: pickle.dump(g, f)
    with open(tmp_path / 'g.p', 'rb') as f: g = pickle.load(f)
    assert isinstance(g, c)
    os.remove(tmp_path / 'g.p')



def test_output_extractor():

    batch_size, input_dim, out_dim = 32, 1, 1
    noise_dim = 100

    x = torch.randn(batch_size, input_dim)

    model = nn.Linear(input_dim + noise_dim, out_dim)

    def m_dict(x): return {'output': model(x)}
    
    def m_attr(x): return SimpleNamespace(output = model(x))
    
    def m_func(x): return model(x)
    
    for m, out_extr in zip([m_dict, m_attr, m_func], ['output', 'output', lambda x: x]):

        g = gConcat(model = m, m_train = 2, noise_dim = noise_dim,
            output_extractor = out_extr,
        ).train()

        preds = g(x)
        assert preds.shape == (batch_size, 2, out_dim)



def test_add():

    batch_size, input_dim, out_dim = 32, 1, 1

    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, out_dim)

    model = nn.Linear(input_dim, out_dim)

    for noise_type in ['normal', 'uniform', 'laplace']:
        g = gAdd(
            model = model,
            noise_dim = input_dim,
            noise_type = noise_type,
            noise_scale = 1.0,
            m_train = 2, 
            m_eval = 512,
        )

        g.train()

    g.train() # change m to m_train
    preds = g(x) # (batch_size, m_train, output_dim)
    assert preds.shape == (batch_size, 2, out_dim)

    # loss = energy_score(y, preds, beta = 1.0, p = 2)
    loss = EnergyScoreLoss(beta = 1.0, p = 2)(y, preds)
    loss.backward()

    g.eval() # changes m to m_eval
    sample = g(x) # (batch_size, m_eval, output_dim)
    assert sample.shape == (batch_size, 512, out_dim)
    