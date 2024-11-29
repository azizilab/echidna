# echidna.plot.utils.py

import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
import warnings

def save_figure(fig, filepath, dpi=300, bbox_inches='tight'):
    """
    Save a Matplotlib figure to a file.

    Parameters:
    fig : matplotlib.figure.Figure
        The figure object to be saved.
    filepath : str
        The file path where the figure will be saved.
    dpi : int, optional
        The resolution of the saved figure in dots per inch (default is 300).
    bbox_inches : str, optional
        Bounding box in inches: 'tight' to include all elements (default is 'tight').
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)

def is_notebook():
    """
    Check if the script is running in a Jupyter notebook.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False
    except NameError:
        return False

def activate_plot_settings():
    """Activate journal quality settings for plotting.
    
    Modified from Decipher with author permission:
    Achille Nazaret, https://github.com/azizilab/decipher/blob/main/decipher/plot/utils.py

    It is recommended for high quality figures while keeping the file size small.
    It is recommended if the figures are to be edited in Adobe Illustrator.
    """
    if is_notebook():        
        sc.settings.set_figure_params(dpi_save=200, vector_friendly=True, fontsize=12)
        warnings.filterwarnings("ignore", category=UserWarning, message="No data for colormapping provided via 'c'")
        # Set Matplotlib font types for vector graphics
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        mpl.rcParams["figure.figsize"] = [6, 4]
        mpl.rcParams["savefig.dpi"] = 200
    else:
        mpl.rcParams["savefig.dpi"] = 400
        
    sns.set_theme(style="white", rc={"grid.color": ".6", "grid.linestyle": ":"})
    sns.set_context("paper", font_scale=1.)
    
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["figure.autolayout"] = False
    mpl.rcParams["legend.frameon"] = True
