import torch
import torch.nn as nn
import torch.nn.functional as F

from engression_pytorch import EnergyScoreLoss, gConcat


def test_readme():

    batch_size, input_dim, out_dim = 32, 1, 1
    noise_dim = 100

    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, out_dim)

    model = nn.Linear(input_dim + noise_dim, out_dim)

    g = gConcat(
        model = model,
        noise_dim = noise_dim,
        noise_type = 'normal',
        noise_scale = 1.0,
        m_train = 2, 
        m_eval = 512,
    )

    g.train() # change m to m_train
    preds = g(x) # (batch_size, m_train, output_dim)
    assert preds.shape == (batch_size, 2, out_dim)

    # loss = energy_score(y, preds, beta = 1.0, p = 2)
    loss = EnergyScoreLoss(beta = 1.0, p = 2)(y, preds)
    loss.backward()

    g.eval() # changes m to m_eval
    sample = g(x) # (batch_size, m_eval, output_dim)
    assert sample.shape == (batch_size, 512, out_dim)