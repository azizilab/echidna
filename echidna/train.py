from pyro.infer import Trace_ELBO, SVI
from pyro.optim import Adam
import pyro
import tqdm

def train_Echidna(echidna, X, W, pi, z, lr=0.1, n_epochs=1000):
    optim = Adam({"lr": lr})
    svi = SVI(echidna.model, echidna.guide, optim, loss=Trace_ELBO())
    losses = []
    pyro.clear_param_store()
    for j in tqdm.tqdm(range(n_epochs)):
        loss = svi.step(X, W, pi, z)
        losses.append(loss)
    return echidna