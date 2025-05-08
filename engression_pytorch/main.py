import torch
import torch.nn as nn

from einops import repeat, rearrange

def energy_score(y, preds, beta = 1.0, p = 2, return_components = False):
    '''
    Computes the energy score loss used in Engression for distributional regression.

    This loss consists of two terms:
    - A data-fitting term that penalizes the distance between each predicted sample and the true target.
    - A repulsion term that encourages diversity among the predicted samples.

    Args:
        y (torch.Tensor): Ground truth targets of shape (batch_size, output_dim).
        preds (torch.Tensor): Predicted samples of shape (batch_size, m_samples, output_dim),
                              where each prediction corresponds to an independently sampled noise input.
        beta (float): Exponent to apply to the norm (default: 1.0). Setting `beta=1.0` gives the energy score.
        p (float): The order of the norm used (e.g., 1 for L1, 2 for L2).
        return_components (bool): If True, return a tuple with (total_loss, term1, term2) for analysis.

    Returns:
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            - If return_components is False, returns a scalar tensor with the total loss.
            - If return_components is True, returns a tuple (total_loss, term1, term2), where:
                - term1 is the fitting term (distance to true targets),
                - term2 is the diversity term (negative pairwise distance among predictions).
    '''

    assert preds.shape[0] == y.shape[0] and preds.shape[2] == y.shape[1]
    n, m, D = preds.shape # n is batch_size, m is num_samples, D is output_dim

    # Term 1: the absolute error between the predicted and true values
    term1 = torch.linalg.vector_norm(preds - y[:, None, :], ord = p, dim = 2).pow(beta).mean()

    # Term 2: pairwise absolute differences between the predicted values
    term2 = torch.tensor(0.0, device = preds.device, dtype = preds.dtype)

    if m > 1:
        # cdist is convenient. The result shape before sum is (n, m, m).
        sum_pairwise_l1_dists = torch.cdist(preds, preds, p = p).pow(beta).sum()
        term2 = - sum_pairwise_l1_dists / (n * 2 * m * (m - 1) * D)

    if return_components:
        return term1 + term2, term1, term2
    
    return term1 + term2

class EnergyScoreLoss(nn.Module):
    def __init__(self, beta = 1.0, p = 2):
        super().__init__()
        self.beta = beta
        self.p = p

    def forward(self, y, preds):
        return energy_score(y, preds, beta = self.beta, p = self.p)


def noise_fn_factory(noise_type, noise_dim, scale):
    # Note scale != std (only for normal)
    if noise_type == 'normal':
        return lambda x: torch.randn(x.shape[0], noise_dim, device = x.device) * scale
    elif noise_type == 'uniform':
        return lambda x: (torch.rand(x.shape[0], noise_dim, device = x.device) - 0.5) * scale
    elif noise_type == 'laplace':
        return lambda x: torch.distributions.Laplace(0, scale).sample((x.shape[0], noise_dim)).to(x.device)
    else:
        raise ValueError(f'Unknown noise type: {noise_type}')


class gConcat(nn.Module):

    def __init__(self, model, m_train, m_eval = 512, noise_type = 'normal', noise_dim = 64, noise_scale = 1.0, noise_fn = None):
        super().__init__()
        self.model = model
        self.m_train = m_train
        self.m_eval = m_eval
        self.noise_fn = noise_fn
        if noise_fn is None:
            self.noise_fn = noise_fn_factory(noise_type, noise_dim, noise_scale)
    
    @property
    def m(self):
        return self.m_train if self.training else self.m_eval
    
    def forward(self, x):
        """
        Performs `m` forward passes of the model with independently sampled noise vectors.

        Args:
            x: Tensor of shape (batch_size, input_dim)
        Returns:
            Tensor of shape (batch_size, m, output_dim)
        """
        b, d = x.shape
        m = self.m

        x = repeat(x, 'b d -> b m d', m = m)
        x = rearrange(x, 'b m d -> (b m) d')

        eps = self.noise_fn(x).to(x.device)

        x = torch.cat([x, eps], dim = 1)
        out = self.model(x)
        out = rearrange(out, '(b m) d -> b m d', b = b)
        return out