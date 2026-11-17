from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def save_figure(
    figure: Figure,
    path: str | Path,
    *,
    dpi: int = 200,
    transparent: bool = False,
    close: bool = False,
    create_parents: bool = True,
) -> Path:
    """

    Save a Matplotlib figure.

    Parameters
    ----------
    figure : Figure
        The Matplotlib figure to save.
    path : str | Path
        The path to save the figure to.
    dpi : int, optional
        The resolution in dots per inch. Default is 200.
    transparent : bool, optional
        Whether to make the figure background transparent. Default is False.
    close : bool, optional
        Whether to close the figure after saving. Default is False.
    create_parents : bool, optional
        Whether to create parent directories if they do not exist. Default is True.

    Returns
    -------
    Path
        The path to the saved figure.
    """
    path = Path(path).expanduser()

    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.parent.exists():
        raise FileNotFoundError(path.parent)

    figure.savefig(
        path,
        dpi=dpi,
        transparent=transparent,
        bbox_inches="tight",
    )

    if close:
        plt.close(figure)

    return path
