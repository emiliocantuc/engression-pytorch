## Engression - Pytorch

The engression loss (energy score) proposed by [Shen et al.](https://arxiv.org/abs/2307.00835) for [distributional regression](https://doi.org/10.1016/j.ecosta.2021.07.006) with a few convenient wrappers, in Pytorch.

The paper's original code by Xinwei Shen is available [here](https://github.com/xwshen51/engression). 

### Install
```
pip install engression-pytorch
```

### Usage
```python
import torch
from engression_pytorch import EnergyScoreLoss, gConcat

batch_size, input_dim, out_dim = 32, 1, 1
noise_dim = 100

x = torch.randn(batch_size, input_dim)
y = torch.randn(batch_size, out_dim)

model = nn.Linear(input_dim + noise_dim, out_dim)

g = gSampler(
    model = model,
    noise_dim = noise_dim,
    noise_type = noise_type,
    noise_scale = 1.0,
    merge_mode = 'concat', # 'add', or lambda x, eps: ...
    m_train = 2, 
    m_eval = 512,
)

g.train() # change m to m_train
preds = g(x) # (batch_size, m_train, output_dim)

# loss = energy_score(y, preds, beta = 1.0, p = 2)
loss = EnergyScoreLoss(beta = 1.0, p = 2)(y, preds)
loss.backward()

g.eval() # changes m to m_eval
sample = g(x) # (batch_size, m_eval, output_dim)
```

### Citations
```bibtex
@misc{shen2024engressionextrapolationlensdistributional,
      title={Engression: Extrapolation through the Lens of Distributional Regression}, 
      author={Xinwei Shen and Nicolai Meinshausen},
      year={2024},
      eprint={2307.00835},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2307.00835}, 
}
```
```bibtex
@article{KNEIB202399,
title = {Rage Against the Mean – A Review of Distributional Regression Approaches},
journal = {Econometrics and Statistics},
volume = {26},
pages = {99-123},
year = {2023},
issn = {2452-3062},
doi = {https://doi.org/10.1016/j.ecosta.2021.07.006},
url = {https://www.sciencedirect.com/science/article/pii/S2452306221000824},
author = {Thomas Kneib and Alexander Silbersdorff and Benjamin Säfken},
}
```
