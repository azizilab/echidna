import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro.distributions as dist
from echidna.custom_dist import TruncatedNormal
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import linregress, gaussian_kde
from scipy.spatial.distance import pdist
from scipy.ndimage import gaussian_filter1d
import seaborn as sns


def pred_posterior_check(
    X_learned: np.ndarray,
    X_true: np.ndarray,
    name: str = "",
    log_scale: bool = False,
    R_val: bool = True,
    equal_line: bool = True,
    save: bool = True,
    color_by_density: bool = False,
    title: str = "Predictive Posterior Check",
    xlabname: str = "True ",
    ylabname: str = "Simulated "
):
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

    # Perform linear regression
    if R_val:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r2 = r_value**2
        y_pred = slope * x + intercept

    maximum = max(np.max(x), np.max(y))
    minimum = min(np.min(x), np.min(y))

    # Scatter plot
    if color_by_density:
        # Calculate point densities
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        # Plot using density as color
        plt.scatter(x, y, c=z, cmap='viridis', alpha=0.5, label='Data points', s=10)
    else:
        plt.scatter(x, y, alpha=0.1, label='Data points')

    if equal_line:
        plt.plot([minimum, maximum], [minimum, maximum], "r", label="x=y")
    
    # Plot the regression line
    if R_val:
        plt.plot(x, y_pred, label="Regression line", color='blue')

    plt.xlabel(xlabname + lbl_pstfix)
    plt.ylabel(ylabname + lbl_pstfix)
    
    # Annotate the plot with the R^2 value
    if R_val:
        plt.text(0.05, 0.95, f'$R^2$ = {r2:.2f}', ha='left', va='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.legend()
    plt.title(title + " " + name)
    
    if save:
        plt.savefig(f'{name}_posterior_predictive_check.png')
    
    plt.show()

# Compare refitted cov with a fitted cov
def compare_covariance_matrix(data1, data2):
    n = data1.shape[-1]
    df1 = pd.DataFrame(data1, columns=[f"Clst {i}" for i in range(n)], index=[f"Clst {i}" for i in range(n)])
    df2 = pd.DataFrame(data2, columns=[f"Clst {i}" for i in range(n)], index=[f"Clst {i}" for i in range(n)])

    linkage_rows = linkage(df1, method='average', metric='euclidean')
    linkage_cols = linkage(df1.T, method='average', metric='euclidean')

    # Get the order of the rows and columns
    row_order = leaves_list(linkage_rows)
    col_order = leaves_list(linkage_cols)

    df1_ordered = df1.iloc[row_order, col_order]
    df2_ordered = df2.iloc[row_order, col_order]


    fig = plt.figure(figsize=(20, 10))

    gs = fig.add_gridspec(3, 4, width_ratios=[0.05, 1, 0.05, 1], height_ratios=[0.2, 1, 0.05], wspace=0.1, hspace=0.1)
    ax_col_dendro1 = fig.add_subplot(gs[0, 1])
    ax_heatmap1 = fig.add_subplot(gs[1, 1])


    ax_col_dendro2 = fig.add_subplot(gs[0, 3])
    ax_heatmap2 = fig.add_subplot(gs[1, 3])

    dendro_col1 = dendrogram(linkage_cols, ax=ax_col_dendro1, orientation='top', no_labels=True, color_threshold=0)
    ax_col_dendro1.set_xticks([])
    ax_col_dendro1.set_yticks([])
    ax_col_dendro1.set_title("Refitted Covariance")

    sns.heatmap(df1_ordered, ax=ax_heatmap1, cbar=False, xticklabels=False, 
                yticklabels=True, cmap="bwr")

    # Plot dendrogram for columns of the second dataset
    dendro_col2 = dendrogram(linkage_cols, ax=ax_col_dendro2, orientation='top', no_labels=True, color_threshold=0)
    ax_col_dendro2.set_xticks([])
    ax_col_dendro2.set_yticks([])
    ax_col_dendro2.set_title("Original Covariance")

    # Plot heatmap for the second dataset
    sns.heatmap(df2_ordered, ax=ax_heatmap2, cbar=False, xticklabels=False, yticklabels=True, cmap="bwr")

    plt.tight_layout()
    plt.show()

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

# Sample p(eta|cov, sigma)
def sample_Eta(eta_mean, cov, sample_size=1000):
    eta_posterior = dist.MultivariateNormal(eta_mean, covariance_matrix=cov)
    samples = eta_posterior.sample([sample_size])
    return torch.nn.functional.softplus(samples)

# Sample p(C|eta)
def sample_C_cond_Eta(eta_samples, c_shape, target_timepoint, cluster_idx, normalize=True):
    c_shape = c_shape[target_timepoint, :, :].squeeze()
    c_samples = dist.Gamma(c_shape, 1/eta_samples[:, :, cluster_idx]).sample()
    if normalize:
        return c_samples/c_shape
    else:
        return c_samples

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

def eta_corr_tree(eta, thres):
    eta_corr = torch.corrcoef(eta).cpu().detach().numpy()
    dist_mat = 1 - eta_corr
    Z = linkage(dist_mat, 'average')
    fig = plt.figure(figsize=(6, 3))
    dn = dendrogram(Z, color_threshold=thres)
    return dn

# Return clone tree based on smoothed eta
def eta_smoothing_tree(eta, thres):
    eta_smoothed = gaussian_filter1d(eta, sigma=6, axis=1, radius=8)
    dist_mat = pdist(eta_smoothed, metric='correlation')
    Z = linkage(dist_mat, method='ward')
    dn = dendrogram(Z, color_threshold=thres)
    return dn

def eta_smoothing_elbow_thresholding(eta, sigma=6, plot_elbow=False):
    eta_smoothed = gaussian_filter1d(eta, sigma=sigma, axis=1, radius=8)
    dist_mat = pdist(eta_smoothed, metric='correlation')
    
    Z = linkage(dist_mat, method='ward')

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

# Return clone tree based on learned covariance and compute elbow-optimized cutoff
def eta_cov_tree_elbow_thresholding(eta, plot_elbow=False, metric='average'):
    Z = linkage(torch.cov(eta).cpu().detach().numpy(), metric)
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

def eta_corr_tree_elbow_thresholding(eta, plot_elbow=False, metric='average'):
    eta_corr = torch.corrcoef(eta).cpu().detach().numpy()
    dist_mat = 1 - eta_corr
    Z = linkage(dist_mat, metric)
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
def assign_clones(dn, X, key='eta_clones'):
    clst = dn.get('leaves_color_list')
    cluster_count = np.unique(clst).shape[0]
    for i in range(len(clst)):
        if clst[i] == 'C0':
            clst[i] = f'C{cluster_count}'
            cluster_count += 1
    keys = dn.get('leaves')
    color_dict = pd.DataFrame(clst)
    color_dict.columns=['color']
    color_dict.index=keys
    hier_colors = [color_dict.loc[int(i)][0] for i in X.obs["leiden"]]
    X.obs[key] = hier_colors


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
