import pyro
import pyro.poutine as poutine
import torch
from pyro import distributions as dist
import torch.nn.functional as F
from echidna.custom_dist import TruncatedNormal
from pyro.distributions import *

class Echidna:
    def __init__(self, model_config, mode='MT', device='cuda'):

        # Echidna parameters
        self.num_genes = model_config.num_genes
        self.num_cells = model_config.num_cells
        self.num_clusters = model_config.num_clusters
        self.num_timepoints = model_config.num_timepoints
        self.log_prob_scaler = 1.0 / (self.num_cells * self.num_genes)
        self.device = device

        # scaler for the shape and rate parameters of covariance diag for variational inference
        self.q_shape_rate_scaler = model_config.q_shape_rate_scaler

        # initialize the scale of variational correlation
        self.q_corr_init = model_config.q_corr_init

        # set Echidna mode
        if mode == 'MT':
            self.model = poutine.scale(self.model_mt, scale=self.log_prob_scaler)
            self.guide = poutine.scale(self.guide_mt, scale=self.log_prob_scaler)
        elif mode == 'ST':
            self.model = poutine.scale(self.model_st, scale=self.log_prob_scaler)
            self.guide = poutine.scale(self.guide_st, scale=self.log_prob_scaler)
        else:
            raise AttributeError("Invalid mode for Enchidna. Please select between single timepoint(ST) and multi timepoint(MT)")
        
        super().__init__()
    
    def model_st(self, X, W, pi, z):
        library_size = X.sum(-1, keepdim=True) * 1e-5

        # MLE estimate of the shape parameter of C
        c_alpha = pyro.param("c_alpha", torch.ones(1,self.num_genes), 
                             constraint=dist.constraints.positive)
        
        # Create sampling dimensions
        gene_plate = pyro.plate('G:genes', self.num_genes, dim=-1, device=self.device)
        cluster_plate = pyro.plate('K:clusters', self.num_clusters, dim=-2, device=self.device)

        # Sample covariance across clusters
        clone_var_dist = dist.Gamma(1, 1).expand([self.num_clusters]).independent(1)
        scale = pyro.sample('scale', clone_var_dist)
        cov_dist = dist.LKJCholesky(self.num_clusters, 1)
        cholesky_corr = pyro.sample('cholesky_corr', cov_dist)
        cholesky_cov = cholesky_corr * torch.sqrt(scale[:, None])
        assert cholesky_cov.shape == (self.num_clusters, self.num_clusters)

        # Sample eta
        with gene_plate:
          eta = pyro.sample("eta", dist.MultivariateNormal(torch.ones(self.num_clusters) * 2, scale_tril=cholesky_cov))
          eta = F.softplus(eta).T
        
        
        # Sample W
        with gene_plate:
            pyro.sample('W', TruncatedNormal(pi @ eta, 0.05, lower=0.), obs=W)

        # Sample C
        with gene_plate:
          with cluster_plate:
            c = pyro.sample('c', dist.Gamma(c_alpha, 1/eta))
        
        # Sample X
        c_scale = c * torch.mean(eta,axis=1).repeat(self.num_genes,1).T
        rate = c_scale[z] * library_size
        pyro.sample('X', dist.Poisson(rate).to_event(), obs=X)
    
    def guide_st(self, X, W, pi, z):
        # Create sampling dimensions
        gene_plate = pyro.plate('G:genes', self.num_genes, dim=-1, device=self.device)
        cluster_plate = pyro.plate('K:clusters', self.num_clusters, dim=-2, device=self.device)

        # Initialize variational parameters
        q_eta_mean = pyro.param('eta_mean', lambda:dist.MultivariateNormal(torch.ones(self.num_clusters) * 2, 
                                                                           torch.eye(self.num_clusters)).sample([self.num_genes]))
        q_c_delta = pyro.param("c_map", torch.ones(self.num_clusters, self.num_genes), 
                               constraint=dist.constraints.positive)
        
        # Variational distribution of cluster covariance

        # Variational distribution for diagonal
        shape = pyro.param('scale_shape', self.q_shape_rate_scaler * torch.ones(self.num_clusters), 
                           constraint=dist.constraints.positive)
        rate = pyro.param('scale_rate', self.q_shape_rate_scaler * torch.ones(self.num_clusters), 
                          constraint=dist.constraints.positive)
        q_clone_var = dist.Gamma(shape, rate).independent(1)
        q_scale = pyro.sample('scale', q_clone_var)

        # Variational distribution for off-diagonal
        corr_dim = self.num_clusters * (self.num_clusters - 1) // 2
        corr_loc = pyro.param("corr_loc", torch.zeros(corr_dim))
        corr_scale = pyro.param("corr_scale", torch.ones(corr_dim) * self.q_corr_init,
                                constraint=dist.constraints.positive)
        corr_cov = torch.diag(corr_scale)
        corr_dist = dist.MultivariateNormal(corr_loc, corr_cov)
        transformed_dist = dist.TransformedDistribution(corr_dist, dist.transforms.CorrCholeskyTransform())
        q_cholesky_corr = pyro.sample("cholesky_corr", transformed_dist)
        q_cholesky_cov = q_cholesky_corr * torch.sqrt(q_scale[:, None])

        # Variational distributions for eta and c
        with gene_plate:
            pyro.sample('eta', dist.MultivariateNormal(q_eta_mean, scale_tril=q_cholesky_cov))
            with cluster_plate:
                pyro.sample('c', dist.Delta(q_c_delta))
    
    def model_mt(self, X, W, pi, z):
        library_size = X.sum(-1, keepdim=True) * 1e-5
        num_timepoints = self.num_timepoints
        num_genes = self.num_genes
        num_clusters = self.num_clusters

        c_shape_param = pyro.param("c_shape", torch.ones(num_timepoints, 1, num_genes).to(self.device),
                             constraint=constraints.positive)

        gene_plate = pyro.plate('G:genes', num_genes, dim=-1, device=self.device)
        cluster_plate = pyro.plate('K:clusters', num_clusters, dim=-2, device=self.device)

        # Eta covariance
        clone_var_dist = dist.Gamma(1, 1).expand([num_clusters]).independent(1)
        scale = pyro.sample('scale', clone_var_dist)
        cov_dist = dist.LKJCholesky(num_clusters, 1.0)
        cholesky_corr = pyro.sample('cholesky_corr', cov_dist)
        cholesky_cov = cholesky_corr * torch.sqrt(scale[:, None])
        assert cholesky_cov.shape == (num_clusters, num_clusters)
        # Sample eta
        with gene_plate:
            eta = pyro.sample('eta', dist.MultivariateNormal(2 * torch.ones(num_clusters), scale_tril=cholesky_cov))
            eta = F.softplus(eta).T

        # Sample W per time point
        with pyro.plate(f"genes", num_genes):
            with pyro.plate("timepoints_w", num_timepoints):
                mu_w = pi @ eta
                W = pyro.sample(f"W", TruncatedNormal(mu_w, 0.05, lower=0.), obs=W)

        # Sample c
        with gene_plate:
            with cluster_plate:
                with pyro.plate("timepoints_c", num_timepoints):
                    c = pyro.sample(f"c", dist.Gamma(c_shape_param, 1/eta))

        for t in range(num_timepoints):
            c_scale = c[t, :, :] * torch.mean(eta,axis=-1).repeat(num_genes,1).T
            rate = c_scale[z[t]] * library_size[t]
            pyro.sample(f"X_{t}", dist.Poisson(rate).to_event(), obs=X[t])
    
    def guide_mt(self, X, W, pi, z):
        num_timepoints = self.num_timepoints
        num_genes = self.num_genes
        num_clusters = self.num_clusters

        gene_plate = pyro.plate('G:genes', num_genes, dim=-1, device=self.device)
        cluster_plate = pyro.plate('K:clusters', num_clusters, dim=-2, device=self.device)
        q_eta_mean = pyro.param('eta_mean',
                          lambda:dist.MultivariateNormal(torch.ones(num_clusters) * 2,
                                                         torch.eye(num_clusters)).sample([num_genes]))
        q_c = pyro.param('c_map', torch.ones(num_timepoints, num_clusters, num_genes), constraint=constraints.positive)

        shape = pyro.param('scale_shape', self.q_shape_rate_scaler * torch.ones(num_clusters), constraint=constraints.positive)
        rate = pyro.param('scale_rate', self.q_shape_rate_scaler * torch.ones(num_clusters), constraint=constraints.positive)
        q_clone_var = dist.Gamma(shape, rate).independent(1)
        q_scale = pyro.sample('scale', q_clone_var)

        corr_dim = num_clusters * (num_clusters - 1) // 2
        corr_loc = pyro.param("corr_loc", torch.zeros(corr_dim))
        corr_scale = pyro.param("corr_scale", torch.ones(corr_dim) * self.q_corr_init,
                          constraint=dist.constraints.positive)
        corr_cov = torch.diag(corr_scale)
        corr_dist = dist.MultivariateNormal(corr_loc, corr_cov)
        transformed_dist = dist.TransformedDistribution(corr_dist, dist.transforms.CorrCholeskyTransform())
        q_cholesky_corr = pyro.sample("cholesky_corr", transformed_dist)
        q_cholesky_cov = q_cholesky_corr * torch.sqrt(q_scale[:, None])

        with gene_plate:
            q_eta = pyro.sample('eta', dist.MultivariateNormal(q_eta_mean, scale_tril=q_cholesky_cov))

        with gene_plate:
            with cluster_plate:
                with pyro.plate("timepoints_c", num_timepoints):
                    pyro.sample("c", dist.Delta(q_c))




