# loss.py

import torch
import torch.nn.functional as F
from torch.distributions import StudentT

def student_t_crps_loss(mu, sigma, target, nu, mc_samples=64):
    dist = StudentT(df=nu, loc=mu, scale=sigma)
    nll = -dist.log_prob(target).mean()

    mu_var = torch.var(mu)
    min_mu_var = torch.var(target).detach() * 0.05
    mu_var_penalty = torch.relu(min_mu_var - mu_var)

    mu_mean = torch.mean(mu)
    target_mean = torch.mean(target).detach()
    mu_bias_penalty = (mu_mean - target_mean).abs()

    log_sigma = torch.log(sigma + 1e-6)
    sigma_var_penalty = torch.relu(1e-3 - torch.var(log_sigma))

    samples = dist.rsample((mc_samples,))
    y_expand = target.unsqueeze(0).expand_as(samples)
    crps_term1 = torch.mean(torch.abs(samples - y_expand))
    pairwise = torch.abs(samples.unsqueeze(0) - samples.unsqueeze(1))
    crps_term2 = 0.5 * torch.mean(pairwise)
    crps = crps_term1 - crps_term2

    direction_penalty = torch.mean(F.relu(-mu * target))

    loss = (
        nll
      + 1.0   * mu_var_penalty
      + 1.0   * mu_bias_penalty
      + 0.5   * sigma_var_penalty
      + 0.05 * crps
      + 0.2   * direction_penalty
    )
    return loss

def nll_only(mu, sigma, target, nu):
    dist = StudentT(df=nu, loc=mu, scale=sigma)
    return -dist.log_prob(target).mean()

def fixed_nu(mu, sigma, target, nu=None):
    nu_fixed = torch.ones_like(mu) * 3.0
    return nll_only(mu, sigma, target, nu_fixed)

def regression_only(mu, sigma, target, nu=None):
    return F.mse_loss(mu, target)

def crps_loss(mu, sigma, target, nu, mc_samples=64):
    from torch.distributions import StudentT

    dist = StudentT(df=nu, loc=mu, scale=sigma)

    samples = dist.rsample((mc_samples,))
    y_expand = target.unsqueeze(0).expand_as(samples)

    crps_term1 = torch.mean(torch.abs(samples - y_expand))
    pairwise = torch.abs(samples.unsqueeze(0) - samples.unsqueeze(1))
    crps_term2 = 0.5 * torch.mean(pairwise)

    crps = crps_term1 - crps_term2
    return crps

def winkler_loss(mu, sigma, target, alpha=0.05):
    """
    Approximates Winkler score for symmetric confidence interval (95% default).
    """
    L = mu - 2 * sigma
    U = mu + 2 * sigma

    in_interval = (target >= L) & (target <= U)
    penalty = 2 / alpha * torch.abs(target - torch.where(target < L, L, U))
    score = (U - L) + (~in_interval) * penalty

    return score.mean()

def composite_loss_v2(mu, sigma, target, nu):
    from torch.distributions import StudentT
    import torch.nn.functional as F

    dist = StudentT(df=nu, loc=mu, scale=sigma)
    nll = -dist.log_prob(target).mean()
    crps = crps_loss(mu, sigma, target, nu)
    winkler = winkler_loss(mu, sigma, target)

    direction_penalty = torch.mean(F.relu(-mu * target))

    return nll + 0.05 * crps + 0.05 * winkler + 0.1 * direction_penalty
