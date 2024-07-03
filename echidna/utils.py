# utils.py

ECHIDNA_GLOBALS = dict()
ECHIDNA_GLOBALS["save_folder"] = "./_echidna_models/"

def create_echidna_uns_key(adata):
    """
    Create the `echidna` uns key if it doesn't exist.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    """
    if "echidna" not in adata.uns:
        adata.uns["echidna"] = dict()