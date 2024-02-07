import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro.distributions as dist
from echidna.custom_dist import TruncatedNormal
from scipy.cluster.hierarchy import dendrogram, linkage
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

def sample_X(X, c, eta, z, library_size):
    mean_scale = torch.mean(eta,axis=1).repeat(X.shape[-1],1).T
    c_scaled = c * mean_scale
    rate = c_scaled[z] * library_size
    X_learned = dist.Poisson(rate).sample()
    X_learned = X_learned.cpu().detach().numpy()
    return X_learned

def sample_W(pi, eta):
    W = TruncatedNormal(pi @ eta, 0.05, lower=0).sample()
    return W.detach().cpu().numpy()

def sample_C(c_shape, c_rate, num_clusters, num_timepoints, target_dim, sample_size=1000):
    c_shape = torch.stack([c_shape] * num_clusters, dim=1)
    c_rate = torch.stack([c_rate] * num_timepoints, dim=0)
    c_posterior = dist.Gamma(c_shape, c_rate)
    return c_posterior.sample([sample_size])[:, target_dim, :]

def sample_Eta(eta_mean, cov, sample_size=1000):
    eta_posterior = dist.MultivariateNormal(eta_mean, covariance_matrix=cov)
    return eta_posterior.sample([sample_size])

def learned_cov(L, scale):
    L = L * torch.sqrt(scale[:, None])
    Sigma = L @ L.T
    return Sigma

def eta_cov_tree(eta, thres):
    Z = linkage(torch.cov(eta).cpu().detach().numpy(), 'average')
    fig = plt.figure(figsize=(6, 3))
    dn = dendrogram(Z, color_threshold=thres)
    return dn

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

def assign_clones(eta, dn, X):
    clst = dn.get('leaves_color_list')
    keys = dn.get('leaves')
    clst_dict = dict(zip(keys, clst))
    color_dict = pd.DataFrame(clst)
    color_dict.columns=['color']
    color_dict.index=keys
    hier_colors = [color_dict.loc[int(i)][0] for i in X.obs["leiden"]]
    X.obs['eta_clones'] = hier_colors



