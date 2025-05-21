import torch
import torch.nn as nn

from einops import repeat, rearrange

def energy_score(y, preds, beta = 1.0, p = 2, lamb = 0.5, return_components = False):
    """
    Computes the generalized energy score loss used in Engression for distributional regression.

    This loss consists of two terms:
    - A data-fitting term that penalizes the distance between each predicted sample and the true target.
    - A repulsion term that encourages diversity among the predicted samples.

    Args:
        y (torch.Tensor): Ground truth targets of shape (batch_size, *).
        preds (torch.Tensor): Predicted samples of shape (batch_size, m_samples, *),
                              where each prediction corresponds to an independently sampled noise input.
        beta (float): Exponent to apply to the norm (default: 1.0). Setting `beta=1.0` gives the energy score.
        p (float): The order of the norm used (e.g., 1 for L1, 2 for L2).
        lamb (float): Weighting factor for the repulsion term.
        return_components (bool): If True, return a tuple with (total_loss, term1, term2) for analysis.

    Returns:
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            - If return_components is False, returns a scalar tensor with the total loss.
            - If return_components is True, returns a tuple (total_loss, term1, term2), where:
                - term1 is the fitting term (distance to true targets),
                - term2 is the diversity term (negative pairwise distance among predictions).
    """

    assert preds.shape[0] == y.shape[0] and preds.shape[2:] == y.shape[1:], \
        'y and preds should only differ in the first dimension'
    
    b, m, *rest = preds.shape
    y = rearrange(y, 'b ... -> b 1 (...)')
    preds = rearrange(preds, 'b m ... -> b m (...)')

    # Term 1: the absolute error between the predicted and true values
    term1 = torch.linalg.vector_norm(preds - y, ord = p, dim = 2).pow(beta).mean()

    # Term 2: pairwise absolute differences between the predicted values
    term2 = torch.tensor(0.0, device = preds.device, dtype = preds.dtype)

    if m > 1:
        # cdist is convenient. The result shape before sum is (n, m, m).
        pairwise_l1_dists = torch.cdist(preds, preds, p = p).pow(beta).mean() * m / (m - 1)
        term2 = - lamb * pairwise_l1_dists

    if return_components:
        return term1 + term2, term1, term2
    
    return term1 + term2


class EnergyScoreLoss(nn.Module):
    
    def __init__(self, beta = 1.0, p = 2, lamb = 0.5, return_components = False):
        super().__init__()
        self.beta = beta
        self.p = p
        self.lamb = lamb
        self.return_components = return_components
        
    def forward(self, y, preds):
        return energy_score(y, preds, beta = self.beta, p = self.p, lamb = self.lamb, return_components = self.return_components)


def _sample_noise(x, noise_type, noise_dim, scale):

    if noise_dim is None:
        noise_dim = x.shape[1] if x.ndim > 2 else 1

    if noise_type == 'normal':
        return torch.randn((x.shape[0], noise_dim, *x.shape[2:]), device=x.device) * scale
    elif noise_type == 'uniform':
        return (torch.rand((x.shape[0], noise_dim, *x.shape[2:]), device=x.device) - 0.5) * scale
    elif noise_type == 'laplace':
        return torch.distributions.Laplace(0, scale).sample((x.shape[0], noise_dim, *x.shape[2:])).to(x.device)
    else:
        raise ValueError(f'Unknown noise type: {noise_type}')



class gSampler(nn.Module):
    r"""
    Wraps a deterministic model to produce *m* stochastic predictions per input.

    Parameters
    ----------
    model : nn.Module
        Backbone network.
    m_train / m_eval : int
        Number of noise draws in train / eval mode.
    merge_mode : {"concat", "add", "multiply"} or Callable, default "concat"
        If callable the interface is `f(x, eps) -> Tensor`.
    noise_type : {"normal", "uniform", "laplace"}, default "normal"
    noise_dim  : int | None, default None
        Dimension of the noise vector. If None, it is set to the input dimension.
    noise_scale: float, default 1.0
    output_extractor : str | Callable | None, default None
        Optional hook to pull/transform a sub-field of the model output.

    Output
    ------
    Tensor shaped `(batch_size, m, …)`.
    """

    def __init__(self, model, m_train, m_eval = 512, noise_type = 'normal', noise_dim = None, noise_scale = 1.0, merge_mode = 'concat', output_extractor = None):
        super().__init__()
        self.model = model
        self.m_train = m_train
        self.m_eval = m_eval
        self.merge_mode = merge_mode
        self.noise_args = (noise_type, noise_dim, noise_scale) 
        self.output_extractor = output_extractor

        assert noise_type in ['normal', 'uniform', 'laplace'], f'Unknown noise type: {noise_type}'
        for m in [m_train, m_eval]:
            assert isinstance(m, int) and m > 0, f'm should be a positive integer, got {m}'

    @property
    def m(self):
        return self.m_train if self.training else self.m_eval
    
    def forward(self, x, *args, **kwargs):
        """
        Performs `m` forward passes of the model with independently sampled noise vectors.

        Args:
            x: Tensor of shape (batch_size, *)
        Returns:
            Tensor of shape (batch_size, m, *)
        """
        b, *_ = x.shape

        x = repeat(x, 'b ... -> (b m) ...', m = self.m)

        eps = _sample_noise(x, *self.noise_args).to(x.device)
        
        # x = torch.cat([x, eps], dim = 1)
        x = self.merge(x, eps)

        out = self.model(x, *args, **kwargs)

        if self.output_extractor is not None:
            out = self._get_out(out)

        out = rearrange(out, '(b m) ... -> b m ...', b = b)
        return out
    
    def merge(self, x, eps):

        if self.merge_mode == 'concat':
            return torch.cat([x, eps], dim = 1)
        elif self.merge_mode == 'add':
            return x + eps
        elif self.merge_mode == 'multiply':
            return x * eps
        else:
            raise ValueError(f'Unknown merge_mode: {self.merge_mode}')
    
    def _get_out(self, out):

        if isinstance(self.output_extractor, str) and hasattr(out, self.output_extractor):
            out = getattr(out, self.output_extractor)

        elif isinstance(self.output_extractor, str) and isinstance(out, dict):
            out = out[self.output_extractor]

        elif callable(self.output_extractor):
            out = self.output_extractor(out)
            
        return out
    
class gConcat(gSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, merge_mode = 'concat', **kwargs)

class gAdd(gSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, merge_mode = 'add', **kwargs)