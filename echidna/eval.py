import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro.distributions as dist
from echidna.custom_dist import TruncatedNormal
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform
import pandas as pd
import pyro

# Plot X learned vs. True
def plot_true_vs_pred(
    X_learned: np.ndarray,
    X_true: np.ndarray,
    name: str = "",
    log_scale: bool = False,
    save: bool = True,
    color: str = None,
):
    """
    Plots lambda vs X.
    """
    # Subsample for plotting
    num = min(len(X_learned.flatten()), 200000)
    indx = np.random.choice(np.arange(len(X_learned.flatten())), num, replace=False)
    X_learned = X_learned.flatten()[indx]
    X_true = X_true.flatten()[indx]

    if log_scale:
        X_learned = np.log(X_learned + 1)
        X_true = np.log(X_true + 1)
        lbl_pstfix = "[log(x + 1)] "
    else:
        lbl_pstfix = ""

    x = X_true
    y = X_learned

    maximum = max(np.max(x), np.max(y))
    minimum = min(np.min(x), np.min(y))
    # scater plot
    plt.scatter(x, y, alpha=0.1)
    plt.plot([minimum, maximum], [minimum, maximum], "r", label="x=y")
    plt.xlabel("True " + lbl_pstfix)
    plt.ylabel("Learned" + lbl_pstfix)

    # Fit a line through x and y
    color = 'g--' if color is None else color + '--'
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), color, label="Fit line")
    plt.title(f"Comprarison of true and learned vals {name} (subsampled)")
    plt.legend()


# Function to retrive the learned parameters
def get_learned_params(echidna, X, W, pi, z):
    guide_trace = pyro.poutine.trace(echidna.guide).get_trace(X, W, pi, z)
    trained_model = pyro.poutine.replay(echidna.model, trace=guide_trace)
    trained_trace = pyro.poutine.trace(trained_model).get_trace(
            X, W, pi, z
        )
    params = trained_trace.nodes
    return params

# Sample X given posterior estimates of c and eta
def sample_X(X, c, eta, z, library_size):
    mean_scale = torch.mean(eta,axis=1).repeat(X.shape[-1],1).T
    c_scaled = c * mean_scale
    rate = c_scaled[z] * library_size
    X_learned = dist.Poisson(rate).sample()
    X_learned = X_learned.cpu().detach().numpy()
    return X_learned

# Sample W given posterior estimates of eta
def sample_W(pi, eta):
    W = TruncatedNormal(pi @ eta, 0.05, lower=0).sample()
    return W.detach().cpu().numpy()


# Sample C from posterior and selec a target cluster for a given time point
def sample_C(c_shape, c_rate, num_clusters, num_timepoints, target_dim, target_timepoint, sample_size=1000):
    c_shape = torch.stack([c_shape] * num_clusters, dim=1).squeeze()
    c_rate = torch.stack([c_rate] * num_timepoints, dim=0)
    c_posterior = dist.Gamma(c_shape, c_rate)
    c_samples = c_posterior.sample([sample_size])
    return c_samples[:, target_timepoint, target_dim, :]


# Sample eta from posterior select a target cluster
def sample_Eta(eta_mean, cov, target_dim, sample_size=1000):
    eta_posterior = dist.MultivariateNormal(eta_mean, covariance_matrix=cov)
    samples = eta_posterior.sample([sample_size])
    return samples[:, :, target_dim]

# return learned covariance across clusters
def learned_cov(L, scale):
    L = L * torch.sqrt(scale[:, None])
    Sigma = L @ L.T
    return Sigma

# Return clone tree based on learned covariance
def eta_cov_tree(eta, thres):
    Z = linkage(torch.cov(eta).cpu().detach().numpy(), 'average')
    fig = plt.figure(figsize=(6, 3))
    dn = dendrogram(Z, color_threshold=thres)
    return dn

# Return clone tree based on learned covariance and compute elbow-optimized cutoff
def eta_cov_tree_elbow_thresholding(eta, plot_elbow=False):
    Z = linkage(torch.cov(eta).cpu().detach().numpy(), 'average')
    distance = Z[:,  2]
    differences = np.diff(distance)
    knee_point = np.argmax(differences)
    threshold = distance[knee_point]
    print("Knee point: ", knee_point + 1)
    print("Threshold: ", threshold)
    dn = dendrogram(Z, color_threshold=threshold)
    if plot_elbow:
        plt.figure()
        plt.plot(range(1, len(differences) + 1), differences, marker='o')
        plt.axvline(x=knee_point + 1, linestyle='--', label='ELBOW threshold', color='red')
        plt.legend()
        plt.xlabel('Merge Step')
        plt.ylabel('Difference in Distance')
        plt.title('Elbow Method')
        plt.show()
    return dn

# Assign clones based on the covariance tree for each cell
def assign_clones(dn, X):
    clst = dn.get('leaves_color_list')
    keys = dn.get('leaves')
    color_dict = pd.DataFrame(clst)
    color_dict.columns=['color']
    color_dict.index=keys
    hier_colors = [color_dict.loc[int(i)][0] for i in X.obs["leiden"]]
    X.obs['eta_clones'] = hier_colors


def mahalanobis_distance_matrix(data, cov_matrix):
    cov_inv = torch.inverse(cov_matrix)
    data_np = data.cpu().detach().numpy()
    cov_inv_np = cov_inv.cpu().detach().numpy()
    
    num_samples = data_np.shape[0]
    distance_matrix = np.zeros((num_samples, num_samples))
    
    for i in range(num_samples):
        for j in range(num_samples):
            delta = data_np[:, i] - data_np[:, j]
            distance_matrix[i, j] = np.sqrt(np.dot(np.dot(delta, cov_inv_np), delta.T))
    
    return distance_matrix

def eta_cov_tree_cophenetic_thresholding(mat, method='average', frac=0.7, dist_matrix=False):
    if dist_matrix:
        Z = linkage(squareform(mat), method)
    else:
        Z = linkage(torch.cov(mat).cpu().detach().numpy(), method)
    # Compute the cophenetic distances
    coph_distances = cophenet(Z)

    max_coph_distance = np.max(coph_distances)
    threshold = frac * max_coph_distance
    
    # Plot the dendrogram with the threshold
    dn = dendrogram(Z, color_threshold=threshold)
    
    return dn
