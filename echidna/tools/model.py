# echidna.tools.model.py

from typing import Union

import pyro
import pyro.poutine as poutine
from pyro import distributions as dist
import torch.nn.functional as F

import torch

from echidna.tools.custom_dist import TruncatedNormal
from echidna.tools.utils import EchidnaConfig

class Echidna:
    def __init__(self, config: Union[EchidnaConfig, dict] = EchidnaConfig()):
        
        if isinstance(config, dict):
            config = EchidnaConfig(**config)
        self.config = config
        
        self.log_prob_scaler = 1.0 / (self.config.num_cells * self.config.num_genes)
        
        # set Echidna mode
        if self.config._is_multi:
            self.model = poutine.scale(self.model_mt, scale=self.log_prob_scaler)
            self.guide = poutine.scale(self.guide_mt, scale=self.log_prob_scaler)
        else:
            self.model = poutine.scale(self.model_st, scale=self.log_prob_scaler)
            self.guide = poutine.scale(self.guide_st, scale=self.log_prob_scaler)

        self.eta_posterior = None
        self.c_posterior = None
        self.cov_posterior = None
            
    def model_st(self, X, W, pi, z):
        library_size = X.sum(-1, keepdim=True) * 1e-5

        gene_plate = pyro.plate('G:genes', self.config.num_genes, dim=-1, device=self.config.device)
        cluster_plate = pyro.plate('K:clusters', self.config.num_clusters, dim=-2, device=self.config.device)

        clone_var_dist = dist.Gamma(1, 1).expand([self.config.num_clusters]).independent(1)
        scale = pyro.sample('scale', clone_var_dist)
        cov_dist = dist.LKJCholesky(self.config.num_clusters, self.config.lkj_concentration)
        cholesky_corr = pyro.sample('cholesky_corr', cov_dist)
        scale = scale[:, None] if self.config.inverse_gamma is False else 1/scale[:, None] # CHECK WITH MING
        cholesky_cov = cholesky_corr * torch.sqrt(scale) # INVERSE GAMMA OPTION

        # Sample eta
        with gene_plate:
            eta = pyro.sample(
                "eta",
                dist.MultivariateNormal(
                    torch.ones(self.config.num_clusters) * self.config.eta_mean_init
                    , scale_tril=cholesky_cov
                )
            )
            eta = F.softplus(eta).T

        # Sample W
        with gene_plate:
            pi = pyro.deterministic("pi", pi)
            W = pyro.sample('W', TruncatedNormal(pi @ eta, 0.05, lower=0.), obs=W)

        # Sample C
        with gene_plate:
            with cluster_plate:
                c = pyro.sample('c', dist.Gamma(1, 1/eta))

        # Sample X
        c_scale = c * torch.mean(eta,axis=1).repeat(self.config.num_genes,1).T
        z_tmp = pyro.deterministic("z", z.to(torch.int64))
        rate = c_scale[z_tmp] * library_size
        X = pyro.sample('X', dist.Poisson(rate).to_event(), obs=X)
        return X, W 

    def guide_st(self, X, W, pi, z):
        gene_plate = pyro.plate('G:genes', self.config.num_genes, dim=-1, device=self.config.device)
        cluster_plate = pyro.plate('K:clusters', self.config.num_clusters, dim=-2, device=self.config.device)

        q_eta_mean = pyro.param(
            'eta_mean', lambda:dist.MultivariateNormal(
                torch.ones(self.config.num_clusters) * self.config.eta_mean_init
                , torch.eye(self.config.num_clusters)
            ).sample([self.config.num_genes])
        )

        q_c_shape = pyro.param('c_shape', torch.ones(1, self.config.num_genes), constraint=dist.constraints.positive)

        shape = pyro.param('scale_shape', self.config.q_shape_rate_scaler * torch.ones(self.config.num_clusters),
                           constraint=dist.constraints.positive)
        rate = pyro.param('scale_rate', self.config.q_shape_rate_scaler * torch.ones(self.config.num_clusters),
                          constraint=dist.constraints.positive)
        q_clone_var = dist.Gamma(shape, rate).independent(1)
        q_scale = pyro.sample('scale', q_clone_var)

        corr_dim = self.config.num_clusters * (self.config.num_clusters - 1) // 2
        corr_loc = pyro.param("corr_loc", torch.zeros(corr_dim))
        corr_scale = pyro.param("corr_scale", torch.ones(corr_dim) * self.config.q_corr_init,
                                constraint=dist.constraints.positive)
        corr_cov = torch.diag(corr_scale)
        corr_dist = dist.MultivariateNormal(corr_loc, corr_cov)
        transformed_dist = dist.TransformedDistribution(corr_dist, dist.transforms.CorrCholeskyTransform())
        q_cholesky_corr = pyro.sample("cholesky_corr", transformed_dist)
        q_scale = q_scale[:, None] if self.config.inverse_gamma is False else 1/q_scale[:, None] # CHECK WITH MING
        q_cholesky_cov = q_cholesky_corr * torch.sqrt(q_scale) # INVERSE GAMMA OPTION

        with gene_plate:
            q_eta = pyro.sample('eta', dist.MultivariateNormal(q_eta_mean, scale_tril=q_cholesky_cov))
            q_eta = F.softplus(q_eta).T

        with gene_plate:
            with cluster_plate:
                pyro.sample("c", dist.Gamma(q_c_shape, 1/q_eta))

    def model_mt(self, X, W, pi, z):
        library_size = X.sum(-1, keepdim=True) * 1e-5
        num_timepoints = self.config.num_timepoints
        num_genes = self.config.num_genes
        num_clusters = self.config.num_clusters

        gene_plate = pyro.plate('G:genes', num_genes, dim=-1, device=self.config.device)
        cluster_plate = pyro.plate('K:clusters', num_clusters, dim=-2, device=self.config.device)

        # Eta covariance
        clone_var_dist = dist.Gamma(1, 1).expand([num_clusters]).independent(1)
        scale = pyro.sample('scale', clone_var_dist)
        cov_dist = dist.LKJCholesky(num_clusters, self.config.lkj_concentration)
        cholesky_corr = pyro.sample('cholesky_corr', cov_dist)
        scale = scale[:, None] if self.config.inverse_gamma is False else 1/scale[:, None] # CHECK WITH MING
        cholesky_cov = cholesky_corr * torch.sqrt(scale) # INVERSE GAMMA
        assert cholesky_cov.shape == (num_clusters, num_clusters) 
        # Sample eta
        with gene_plate:
            eta = pyro.sample('eta', dist.MultivariateNormal(self.config.eta_mean_init * torch.ones(num_clusters), scale_tril=cholesky_cov))
            eta = F.softplus(eta).T

        # Sample W per time point
        with gene_plate:
            with pyro.plate("timepoints_w", num_timepoints):
                pi = pyro.deterministic("pi", pi)
                mu_w = pi @ eta
                W = pyro.sample(f"W", TruncatedNormal(mu_w, 0.05, lower=0.), obs=W)

        # Sample c
        with gene_plate:
            with cluster_plate:
                with pyro.plate("timepoints_c", num_timepoints):
                    c = pyro.sample(f"c", dist.Gamma(1, 1/eta))

        for t in range(num_timepoints):
            c_scale = c[t, :, :] * torch.mean(eta,axis=-1).repeat(num_genes,1).T
            z_tmp = pyro.deterministic(f"z_{t}", z[t].to(torch.int64))
            rate = c_scale[z_tmp] * library_size[t]
            pyro.sample(f"X_{t}", dist.Poisson(rate).to_event(), obs=X[t])

    def guide_mt(self, X, W, pi, z):
        num_timepoints = self.config.num_timepoints
        num_genes = self.config.num_genes
        num_clusters = self.config.num_clusters

        gene_plate = pyro.plate('G:genes', num_genes, dim=-1, device=self.config.device)
        cluster_plate = pyro.plate('K:clusters', num_clusters, dim=-2, device=self.config.device)

        q_eta_mean = pyro.param('eta_mean',
                          lambda:dist.MultivariateNormal(torch.ones(num_clusters) * self.config.eta_mean_init,
                                                         torch.eye(num_clusters)).sample([num_genes]))
        q_c_shape = pyro.param('c_shape', torch.ones(num_timepoints, 1, num_genes), constraint=dist.constraints.positive)

        shape = pyro.param('scale_shape', self.config.q_shape_rate_scaler * torch.ones(num_clusters), constraint=dist.constraints.positive)
        rate = pyro.param('scale_rate', self.config.q_shape_rate_scaler * torch.ones(num_clusters), constraint=dist.constraints.positive)
        q_clone_var = dist.Gamma(shape, rate).independent(1)
        q_scale = pyro.sample('scale', q_clone_var)

        corr_dim = num_clusters * (num_clusters - 1) // 2
        corr_loc = pyro.param("corr_loc", torch.zeros(corr_dim))
        corr_scale = pyro.param("corr_scale", torch.ones(corr_dim) * self.config.q_corr_init,
                          constraint=dist.constraints.positive)
        corr_cov = torch.diag(corr_scale)
        corr_dist = dist.MultivariateNormal(corr_loc, corr_cov)
        transformed_dist = dist.TransformedDistribution(corr_dist, dist.transforms.CorrCholeskyTransform())
        q_cholesky_corr = pyro.sample("cholesky_corr", transformed_dist)
        q_scale = q_scale[:, None] if self.config.inverse_gamma is False else 1/q_scale[:, None] # CHECK WITH MING  
        q_cholesky_cov = q_cholesky_corr * torch.sqrt(q_scale) # INVERSE GAMMA OPTION 

        with gene_plate:
            q_eta = pyro.sample('eta', dist.MultivariateNormal(q_eta_mean, scale_tril=q_cholesky_cov))
            q_eta = F.softplus(q_eta).T

        with gene_plate:
            with cluster_plate:
                with pyro.plate("timepoints_c", num_timepoints):
                    pyro.sample("c", dist.Gamma(q_c_shape, 1/q_eta))
