from pyro.infer import Trace_ELBO, SVI
from pyro.optim import ClippedAdam
from pyro.optim import Adam
import pyro
import tqdm
import torch

def train_Echidna(echidna, X, W, pi, z, lr=0.1, n_epochs=1000):
    optim = Adam({"lr": lr})
    svi = SVI(echidna.model, echidna.guide, optim, loss=Trace_ELBO())
    losses = []
    pyro.clear_param_store()
    for j in tqdm.tqdm(range(n_epochs), position=0, leave=True):
        loss = svi.step(X, W, pi, z)
        losses.append(loss)
    return echidna, losses

def train_Echidna_early_stopping(echidna, X, W, pi, z, X_val, W_val, pi_val, z_val, lr=0.1, n_epochs=1000, patience=10):
    optim = Adam({"lr": lr})
    svi = SVI(echidna.model, echidna.guide, optim, loss=Trace_ELBO())
    losses = []
    val_losses = []
    pyro.clear_param_store()
    
    patience_trigger = 0
    last_val = 99999

    for j in tqdm.tqdm(range(n_epochs), position=0, leave=True):
        loss = svi.step(X, W, pi, z)
        losses.append(loss/len(X[0]))

        with torch.no_grad():
            val_loss = Trace_ELBO().loss(echidna.model, echidna.guide, X_val, W_val, pi_val, z_val)
            val_losses.append(val_loss/len(X_val[0]))

        if val_loss < last_val:
          last_val = val_loss
          patience_trigger = 0
        
        patience_trigger += 1

        if patience_trigger >= patience:
          break
        
    return echidna, losses, val_losses