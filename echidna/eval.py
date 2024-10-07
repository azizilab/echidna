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
import os
import scanpy as sc
import scipy
from sklearn.mixture import GaussianMixture
import tqdm


######### SAMPLE LEVEL EVAL FUNCTIONS #######################

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
    indx = np.random.choice(
        np.arange(len(X_learned.flatten())), num, replace=False)
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
        plt.scatter(x, y, c=z, cmap='viridis',
                    alpha=0.5, label='Data points', s=10)
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
        plt.text(0.05, 0.95, f'$R^2$ = {r2:.2f}', ha='left', va='center', transform=plt.gca(
        ).transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.legend()
    plt.title(title + " " + name)

    if save:
        plt.savefig(f'{name}_posterior_predictive_check.png')

    plt.show()

# Compare refitted cov with a fitted cov


def compare_covariance_matrix(data1, data2):
    n = data1.shape[-1]
    df1 = pd.DataFrame(data1, columns=[f"Clst {i}" for i in range(n)], index=[
                       f"Clst {i}" for i in range(n)])
    df2 = pd.DataFrame(data2, columns=[f"Clst {i}" for i in range(n)], index=[
                       f"Clst {i}" for i in range(n)])

    linkage_rows = linkage(df1, method='average', metric='euclidean')
    linkage_cols = linkage(df1.T, method='average', metric='euclidean')

    # Get the order of the rows and columns
    row_order = leaves_list(linkage_rows)
    col_order = leaves_list(linkage_cols)

    df1_ordered = df1.iloc[row_order, col_order]
    df2_ordered = df2.iloc[row_order, col_order]

    fig = plt.figure(figsize=(20, 10))

    gs = fig.add_gridspec(3, 4, width_ratios=[0.05, 1, 0.05, 1], height_ratios=[
                          0.2, 1, 0.05], wspace=0.1, hspace=0.1)
    ax_col_dendro1 = fig.add_subplot(gs[0, 1])
    ax_heatmap1 = fig.add_subplot(gs[1, 1])

    ax_col_dendro2 = fig.add_subplot(gs[0, 3])
    ax_heatmap2 = fig.add_subplot(gs[1, 3])

    dendro_col1 = dendrogram(linkage_cols, ax=ax_col_dendro1,
                             orientation='top', no_labels=True, color_threshold=0)
    ax_col_dendro1.set_xticks([])
    ax_col_dendro1.set_yticks([])
    ax_col_dendro1.set_title("Refitted Covariance")

    sns.heatmap(df1_ordered, ax=ax_heatmap1, cbar=False, xticklabels=False,
                yticklabels=True, cmap="bwr")

    # Plot dendrogram for columns of the second dataset
    dendro_col2 = dendrogram(linkage_cols, ax=ax_col_dendro2,
                             orientation='top', no_labels=True, color_threshold=0)
    ax_col_dendro2.set_xticks([])
    ax_col_dendro2.set_yticks([])
    ax_col_dendro2.set_title("Original Covariance")

    # Plot heatmap for the second dataset
    sns.heatmap(df2_ordered, ax=ax_heatmap2, cbar=False,
                xticklabels=False, yticklabels=True, cmap="bwr")

    plt.tight_layout()
    plt.show()

# Sample X given posterior estimates of c and eta


def sample_X(X, c, eta, z, library_size):
    mean_scale = torch.mean(eta, axis=1).repeat(X.shape[-1], 1).T
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
        plt.axvline(x=knee_point + 1, linestyle='--',
                    label='ELBOW threshold', color='red')
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
        plt.axvline(x=knee_point + 1, linestyle='--',
                    label='ELBOW threshold', color='red')
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
        plt.axvline(x=knee_point + 1, linestyle='--',
                    label='ELBOW threshold', color='red')
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
    color_dict.columns = ['color']
    color_dict.index = keys
    hier_colors = [color_dict.loc[int(i)][0] for i in X.obs["leiden"]]
    X.obs[key] = hier_colors


# Compute gene dosage effect. Variance explained on posterior samples
def gene_dosage_effect(eta_samples, eta_mean, eta_mode, cluster_idx, c_shape, timepoint_idx):
    # delta Var(c|eta)
    # delta_var = c_shape[:, timepoint_idx]*eta_mean[:, cluster_idx]**2 - c_shape[:, timepoint_idx]*eta_mode**2
    delta_var = c_shape[:, timepoint_idx]*eta_mean[:, cluster_idx]**2 - \
		c_shape[:, timepoint_idx]*eta_mode[None, cluster_idx]**2

    # Var(E[c|eta])
    exp_c_given_eta = c_shape[None, :, timepoint_idx] * \
        eta_samples[:, :, cluster_idx]
        
    var_exp_c_given_eta = torch.var(exp_c_given_eta, unbiased=True, dim=0)
    # E[Var(c|eta)]
    exp_var_c_given_eta = c_shape[None, :,timepoint_idx] * eta_samples[:, :, cluster_idx]**2
    exp_var_c_given_eta = torch.mean(exp_var_c_given_eta, dim=0)

    # (Var(c|eta)-Var(c|eta_mode))/(Var(E[c|eta])+E[Var(c|eta)])
    var_exp = delta_var / (var_exp_c_given_eta + exp_var_c_given_eta)

    return var_exp


######### COHORT LEVEL EVAL FUNCTIONS #######################

# fit GMM to obtain neutral level for each cluster.
# Genes have to be ordered for the Gaussian filter.
def get_neutrals(PATH, ordered_genes, visaulize=False, thres_quantile=0.75, ext='.h5'):
	clust_neutral = []
	clust_names = []
	for patient in os.listdir(PATH):
		path = os.path.join(PATH, patient)

		# Load data
		eta = pd.read_csv(os.path.join(path + "/eta.csv"), index_col=0).T
		Xad = sc.read_h5ad(os.path.join(path + f"/X{ext}"))

		df = Xad.to_df()

		# Variance filter
		var = pd.DataFrame(df.var())
		thres = np.quantile(var[0], thres_quantile)
		var_filter = var[var[0] >= thres].index
		ordered = [i for i in ordered_genes if i in eta.columns]
		ordered_filtered = [i for i in ordered if i in var_filter]

		# Subset and smooth data
		eta = eta[ordered]
		eta_filtered = eta[ordered_filtered]
		eta_filtered_smoothed = scipy.ndimage.gaussian_filter1d(
		eta_filtered, sigma=10, axis=1, radius=20)

		# Get observations
		obs = Xad.obs
		unique_clusters = np.unique(obs["leiden"])
		n_clusters = len(unique_clusters)

		fig, axes = plt.subplots(nrows=(n_clusters + 1) // 2,
							ncols=2, figsize=(10, 5 * ((n_clusters + 1) // 2)))
		axes = axes.flatten()

		for idx, i in enumerate(unique_clusters):
			vals_filtered = eta_filtered_smoothed[int(i), :]

			# Fit GMM
			gmm = GaussianMixture(n_components=5)
			gmm.fit(np.asarray([vals_filtered]).T)
			labels = gmm.predict(np.asarray([vals_filtered]).T)

			# Identify the neutral component
			neut_component = scipy.stats.mode(labels)[0]
			gmm_val = pd.DataFrame({"components": labels, "vals": vals_filtered})
			gmm_mean = np.mean(
				gmm_val[gmm_val["components"] == neut_component]["vals"])
			gmm_std = np.std(gmm_val[gmm_val["components"]
							== neut_component]["vals"])

			clust_neutral.append(gmm_mean)
			n = patient + "_" + i
			clust_names.append(n)
   
			if visaulize:

				ax = axes[idx]
				x = np.linspace(-5, 10, 1000)
				data = np.asarray([vals_filtered]).T
				data = data.squeeze()
				ax.hist(data, bins=30, density=True, alpha=0.6, color='gray')

				for mean, variance, weight in zip(gmm.means_, gmm.covariances_, gmm.weights_):
					variance = np.sqrt(variance).flatten()
					if np.isclose(mean[0], gmm_mean, atol=0.1):
						ax.plot(x, weight * (1 / np.sqrt(2 * np.pi * variance)) *
								np.exp(-(x - mean) ** 2 / (2 * variance)),
								label=f'Neutral mean = {gmm_mean:.3f}',
								color='red', linewidth=3)
					else:
						ax.plot(x, weight * (1 / np.sqrt(2 * np.pi * variance)) *
								np.exp(-(x - mean) ** 2 / (2 * variance)),
								label=f'Component mean={mean[0]:.3f}')
      
				logprob = gmm.score_samples(x.reshape(-1, 1))
				pdf = np.exp(logprob)
				ax.plot(x, pdf, '-k', label='Global Density')

				ax.set_xlabel('Eta values')
				ax.set_ylabel('Density')
				ax.set_title(f"Cluster {i} for {patient}")
				ax.legend()
		plt.tight_layout()
		if visaulize:
			plt.show()
	return pd.DataFrame(clust_neutral, clust_names, columns=[0]).T

# compute gene dosage effect for a cohort of samples
def gene_dosage_effect_cohort(USE_ST, PATH, all_neutrals, to=[], ext=".h5"):

    def sort_timepoints(val):
        custom_order = to
        return custom_order.index(val)

    corr_df = pd.DataFrame()

    # timepoint_order = ['untreated', 'pre', 'on', 'on2', 'post', 'post1_pre2', 'post1_on2']
    timepoint_order = {name: index for index,
                       name in enumerate(to)}

    for patient in tqdm.tqdm((os.listdir(PATH))):
        path = os.path.join(PATH, patient)
        eta_mean = pd.read_csv(os.path.join(path + "/eta.csv"), index_col=0)
        eta_mean = torch.tensor(eta_mean.to_numpy()).float()
        c_shape = pd.read_csv(path + "/c_shape.csv", index_col=0)
        c_shape = torch.tensor(np.array(c_shape))
        if USE_ST:
            c_pre = pd.read_csv(os.path.join(path + "/c.csv"), index_col=0)
        else:
            c_pre = pd.read_csv(os.path.join(path + "/c_pre.csv"), index_col=0)
        cov = np.load(os.path.join(path + "/cov.npy"))
        cov = torch.tensor(cov)

        ad = sc.read_h5ad(os.path.join(path + f"/X{ext}"))

        obs = ad.obs
        clusters = c_pre.columns
        if not USE_ST:
            all_timepoints = sorted(
                ad.obs['condition'].unique(), key=sort_timepoints)

        eta_samples = sample_Eta(eta_mean, cov)
        eta_mode = all_neutrals.filter(regex=f"^{patient}").loc[0].to_numpy()

        for cluster in clusters:
            i = int(cluster)

            if not USE_ST:
                timepoints, counts = np.unique(
                    obs[obs["leiden"] == cluster]["condition"], return_counts=True)
                timepoints = np.array(timepoints)
                counts = np.array(counts)

                sorted_indices = np.argsort(
                    [timepoint_order[tp] for tp in timepoints])

                timepoints = timepoints[sorted_indices]
                counts = counts[sorted_indices]

                timepoint_idx = np.argmax(counts)
                cluster_tp = timepoints[timepoint_idx]
                if cluster_tp == all_timepoints[0]:
                    timepoint_idx = 0
                else:
                    timepoint_idx = 1
            else:
                timepoint_idx = 0

            var_exp = gene_dosage_effect(
                eta_samples, eta_mean, eta_mode, i, c_shape, timepoint_idx)

            df = pd.DataFrame(var_exp, index=c_pre.index,
                              columns=[patient+"_"+cluster])
            corr_df = pd.concat((corr_df, df), axis=1, join='outer')

    return corr_df
