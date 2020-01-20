import torch
import numpy as np

""" Amortized Stein Variational Gradient Descent Ops """
def _rbf_kernel(x, y):
    k_fix, out_dim1 = x.size()[-2:]
    k_upd, out_dim2 = y.size()[-2:]
    assert out_dim1 == out_dim2
    leading_shape = x.size()[:-2]
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)  # pairwise distance between particles
    dist_sq = diff.pow(2).sum(-1)
    dist_sq = dist_sq.unsqueeze(-1)
    median_sq = torch.median(dist_sq, dim=1)[0]
    median_sq = median_sq.unsqueeze(1)
    h = median_sq / torch.log(torch.tensor([k_fix]).float() + 1.).item()
    kappa = torch.exp(- dist_sq / h)
    kappa_grad = -2. * diff / h * kappa # create gradient
    return kappa, kappa_grad
    # is good


def batch_rbf(x):
    xy = torch.mm(x, x.T)
    x2 = x.pow(2).sum(1).view(x.shape[0], 1)
    x2e = x2.repeat(1, x.shape[0])
    H = torch.sub(torch.add(x2e, x2e.T), 2*xy)

    median = torch.median(H.view(-1))
    norm = (median*.5 / np.log(x.shape[0] + 1.0)).pow(.5)
    kxy = torch.exp(-H / norm ** 2 / 2.0)
    return kxy

def batch_rbf_xy(x, y, h_min=1e-8):
    """
        xs(`tf.Tensor`): A tensor of shape (N x Kx x D) containing N sets of Kx
            particles of dimension D. This is the first kernel argument.
        ys(`tf.Tensor`): A tensor of shape (N x Ky x D) containing N sets of Kx
            particles of dimension D. This is the second kernel argument.
        h_min(`float`): Minimum bandwidth.
    """
    Kx, D = x.shape[-2:]
    Ky, D2 = y.shape[-2:]
    assert D == D2
    leading_shape = x.shape[:-2]
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)
    # ... x Kx x Ky x D
    dist_sq = torch.sum(diff**2, -1)
    input_shape = (*leading_shape, *[Kx * Ky])
    values, _ = torch.topk(dist_sq.view(*input_shape), k=(Kx * Ky // 2 + 1))  # ... x floor(Ks*Kd/2)

    medians_sq = values[..., -1]  # ... (shape) (last element is the median)
    h = medians_sq / np.log(Kx)  # ... (shape)
    h = torch.max(h, torch.tensor([h_min]).cuda())
    h = h.detach()  # Just in case.
    h_expanded_twice = h.unsqueeze(-1).unsqueeze(-1)
    # ... x 1 x 1
    kappa = torch.exp(-dist_sq / h_expanded_twice)  # ... x Kx x Ky
    # Construct the gradient
    h_expanded_thrice = h_expanded_twice.unsqueeze(-1)
    # ... x 1 x 1 x 1
    kappa_expanded = kappa.unsqueeze(-1)  # ... x Kx x Ky x 1

    kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded
    """
    print ('diff', diff.mean().item())
    print ('h_expa', h_expanded_thrice.mean().item())
    print ('kappa_expa', kappa_expanded.mean().item())
    print ('x-y', (x-y).mean().item())
    """
    # ... x Kx x Ky x D
    return kappa, kappa_grad

