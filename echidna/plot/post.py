# post.py

import matplotlib.pyplot as plt
import seaborn as sns

from echidna.tools.eval import eta_cov_tree_elbow_thresholding
from echidna.tools.housekeeping import load_model

def dendrogram(adata, elbow: bool=False):
    echidna = load_model(adata)
    
    eta_cov_tree_elbow_thresholding(
        echidna.eta_ground_truth,
        plot_dendrogram=not elbow,
        plot_elbow=elbow,
    )
