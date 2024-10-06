import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro.distributions as dist
from echidna.custom_dist import TruncatedNormal
import pandas as pd
import pyro
import torch.nn.functional as F

# Function to retrive the learned parameters from one pass of the model
def get_learned_params(echidna, X, W, pi, z):
    guide_trace = pyro.poutine.trace(echidna.guide).get_trace(X, W, pi, z)
    trained_model = pyro.poutine.replay(echidna.model, trace=guide_trace)
    trained_trace = pyro.poutine.trace(trained_model).get_trace(
            X, W, pi, z
        )
    params = trained_trace.nodes
    return params

# Posterior mean of eta
def eta_posterior_estimates(echidna, X, W, pi, z, num_samples=1000):
    eta = torch.zeros([pi.shape[-1], X.shape[-1]]).cpu()
    for _ in range(num_samples):
        params = get_learned_params(echidna, X, W, pi, z)
        eta += F.softplus(params['eta']['value'].T).detach().cpu()
    eta /= num_samples
    return eta

# Posterior mean of c. Takes in posterior mean of eta
def c_posterior_estimates(eta, mt=True):
    if mt:
        c_shape = pyro.param('c_shape').detach().cpu().squeeze(1)
        c_on = c_shape[1] * eta
        c_pre = c_shape[0] * eta
        return c_on, c_pre
    else:
        c_shape = pyro.param('c_shape').detach().cpu()
        c = c_shape * eta
        return c

# Posterior mean of covariance
def cov_posterior_estimate(inverse_gamma=False):
    corr_loc = pyro.param("corr_loc").detach().cpu()
    corr_scale = pyro.param("corr_scale").detach().cpu()
    corr_cov = torch.diag(corr_scale)
    corr_dist = dist.MultivariateNormal(corr_loc, corr_cov)
    transformed_dist = dist.TransformedDistribution(corr_dist, dist.transforms.CorrCholeskyTransform())
    chol_samples = transformed_dist.sample((10000,))
    L_shape = pyro.param('scale_shape').detach().cpu()
    L_rate = pyro.param('scale_rate').detach().cpu()
    L = L_shape/L_rate
    if not inverse_gamma:
        cov = chol_samples.mean(0) * np.sqrt(L[:, None])
    else:
        cov = chol_samples.mean(0) * np.sqrt(1/L[:, None])
    cov = cov@cov.T
    cov = cov.numpy()
    return cov

# Posterior mean of correlation matrix. Takes in estimated covariance
def normalize_cov(cov):
    std_dev = np.sqrt(np.diag(cov))
    outer_std_dev = np.outer(std_dev, std_dev)
    corr_matrix = cov / outer_std_dev
    corr_matrix[np.diag_indices_from(corr_matrix)] = 1
    return corr_matrix