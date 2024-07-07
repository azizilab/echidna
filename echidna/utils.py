# echidna.utils.py

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
    if "save_data" not in adata.uns["echidna"]:
        adata.uns["echidna"]["save_data"] = dict()
    if "timepoint_order" not in adata.uns["echidna"]:
        adata.uns["echidna"]["timepoint_order"] = dict()
        
def get_logger(name):
    import logging
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s : %(message)s",
        level=logging.INFO,
    )
    return logger
