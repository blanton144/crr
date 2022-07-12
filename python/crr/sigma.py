import numpy as np
import crr.cCRR


def sigma(image, sp=10):
    """Calculates standard deviation in image by checking pixel pairs

    Parameters
    ----------
    image : 2-D ndarray of np.float32
        image to analuze
    sp : int
        spacing between pixel pairs (default 10)

    Return
    -------
    sigma : np.float32
        standard deviation

    Notes
    -----
    Calls sigma.c through pybind
"""

    image = np.array(image)
    nx = image.shape[0]
    ny = image.shape[1]
    sigval = crr.cCRR.sigma(image, nx, ny, sp)

    return(sigval)
