import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro.distributions as dist
from echidna.custom_dist import TruncatedNormal
from scipy.cluster.hierarchy import dendrogram, linkage

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

def sample_X(X, c, eta, z, library_size):
    mean_scale = torch.mean(eta,axis=1).repeat(X.shape[-1],1).T
    c_scaled = c * mean_scale
    rate = c_scaled[z[0, :]] * library_size
    X_learned = dist.Poisson(rate).sample()
    X_learned = X_learned.cpu().detach().numpy()
    return X_learned

def sample_W(pi, eta):
    return TruncatedNormal(pi @ eta, 0.05, lower=0).sample()

def learned_cov(L, scale):
    L = L * torch.sqrt(scale[:, None])
    Sigma = L @ L.T
    return Sigma