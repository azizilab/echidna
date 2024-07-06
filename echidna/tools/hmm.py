# echinda.tools.hmm.py

import logging

import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from hmmlearn import hmm

from echidna.tools.housekeeping import load_model
from echidna.utils import get_logger

logger = get_logger(__name__)

def infer_cnv(adata, cluster: str):
    echidna = load_model(adata)
    eta = echidna.eta_ground_truth
    corrected_eta = eta / np.sqrt(np.diag(np.cov(eta.T)))
    
    values, draw_lines, genes, chrs_list = bin_by_bands(corrected_eta, "CNV")
    states = get_states(np.asarray(vals),n_components=5,transition=1)

def bin_by_bands(df, column):
    """
    df: is a genes (index) x clusters matrix (corrected eta)
    column: the cluster to be analyzed
    """
    values = []
    keys = []
    draw_lines = []
    counts = 0
    chrs = "chr1"
    chrs_list = [chrs]
    all_bands = arms["band"]
    all_chr = arms["chr"]
    genes = []
    # loop through each band, taking the mean of all genes in that band
    for i in range(arms.shape[0]):
        if all_bands[i]!="NaN" and all_chr[i]+"_"+all_bands[i] in bands.keys():
            genes_in_chr = np.intersect1d(bands[all_chr[i]+"_"+all_bands[i]],df.index)
            value = np.mean(df.loc[genes_in_chr][column])
            if not pd.isna(value):
                genes.append([genes_in_chr])
                values.append(value)
                keys.append(value)
                #track the end of each chromosome
                if all_chr[i]!=chrs:
                    draw_lines.append(counts)
                    chrs = all_chr[i]
                    chrs_list.append(chrs)
                counts+=1
    draw_lines.append(counts)
    return values, draw_lines, genes, chrs_list

def get_states(vals, n_components=5, p_thresh=1e-5, neutral=2, transition=1, startprob=1):
    """
    implements a gaussian hmm to call copy number states smoothing along the genome
    vals: ordered copy number values (from bin_by_bands function)
    n_components: number of components to use for the hmm. We default to 5 for better sensitivity.
    p_thresh: cutoff to determine if a cluster should also be neutral
    neutral: the number to use for neutral. For Echidna, this is always 2. 
    transition: specifies the transmat_prior in the GaussianHMM function
    startprob: specifies the startprob_prior in the GaussianHMM function
    """
    scores = list()
    models = list()

    for idx in range(10):  # ten different random starting states
        # define our hidden Markov model
        model = hmm.GaussianHMM(n_components=n_components, random_state=idx,
                               n_iter=10,transmat_prior=transition, startprob_prior=startprob)
        model.fit(vals[:, None])
        models.append(model)
        scores.append(model.score(vals[:, None]))
        print(f"Converged: {model.monitor_.converged}\t\t"
              f"Score: {scores[-1]}")
    
    # get the best model
    model = models[np.argmax(scores)]
    print(f"The best model had a score of {max(scores)} and "
          f"{model.n_components} components")
    
    # use the Viterbi algorithm to predict the most likely sequence of states given the model
    states = model.predict(vals[:, None])

    #figure out which state corresponds to whch CN stattes
    tmp = pd.DataFrame({"vals":vals,"states":states})
    
    state_dict = pd.DataFrame()
    for i in np.unique(states):
        mean = tmp[tmp["states"]==i].mean()
        state_dict[i] = mean
    #determine the state that is neutral (closest to 2)
    state_dict = state_dict.T.sort_values(by="vals")
    state_dict["dist_to_neut"] = abs(state_dict["vals"]-neutral)
    
    #see if any other states are neutral using a t-test
    neut_state = list(state_dict[state_dict["dist_to_neut"]==state_dict["dist_to_neut"].min()]["states"])
    neut_val = list(state_dict[state_dict["dist_to_neut"]==state_dict["dist_to_neut"].min()]["vals"])
    val_states = pd.DataFrame([vals, states]).T
    neut_dist = val_states[val_states[1]==neut_state[0]][0]
    for i in state_dict["states"]:
        if i not in neut_state:
            tmp = val_states[val_states[1]==i][0]
            p = ttest_ind(neut_dist, tmp)[1]
            print(i,p)
            if p>p_thresh:
                neut_state.append(i)
                neut_val.append(list(state_dict[state_dict["states"]==i]["vals"])[0])
                
    #assign everything less than neutral as deleted and greater than as amplified
    cnvs = []
    for i in range(state_dict.shape[0]):
        if state_dict["states"].iloc[i] in neut_state:
            cnvs.append("neut")
        elif state_dict["vals"].iloc[i] > max(neut_val):
            cnvs.append("amp")
        else:
            cnvs.append("del")
    
    state_dict["CNV"]=cnvs
    print(state_dict)
    convert = {state_dict["states"][i]:state_dict["CNV"][i] for i in range(state_dict.shape[0])}
    cnvs = [convert[i] for i in states]
    return cnvs