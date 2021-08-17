import numpy as np
import scipy.optimize as optimize

def gplusb_image(sampling, xcen=0., ycen=0., sigma=1., flux=1000., background=0.):
    xg = sampling.xgrid.flatten()
    yg = sampling.ygrid.flatten()
    image = (flux * np.exp(- 0.5 * ((xg - xcen)**2 + (yg - ycen)**2) / sigma**2) /
             (2. * np.pi * sigma**2)) + background
    image = image.reshape(sampling.nx, sampling.ny)
    return(image)

def gplusb_chi2(x, sampling, image, mask=None, return_params=False):
    if(mask is None):
        mask = np.ones((sampling.nx, sampling.ny))
        mask[0, :] = 0.
        mask[:, 0] = 0.
        mask[-1, :] = 0.
        mask[:, -1] = 0.
    sigma = np.exp(x[0])
    g = gplusb_image(sampling, sigma=sigma, flux=1., background=0.)
    b = gplusb_image(sampling, sigma=sigma, flux=0., background=1.)
    A = np.zeros((2, 2)) 
    A[0, 0] = (g * g * mask).sum()
    A[0, 1] = (g * b * mask).sum()
    A[1, 0] = A[0, 1]
    A[1, 1] = (b * b * mask).sum()
    f = np.zeros(2)
    f[0] = (g * image * mask).sum()
    f[1] = (b * image * mask).sum()
    Ainv = np.linalg.inv(A)
    p = Ainv.dot(f)
    flux = p[0]
    background = p[1]
    chi2arr = ((image - gplusb_image(sampling, sigma=sigma, flux=flux, background=background))**2)
    chi2 = (chi2arr * mask).sum()
    if(return_params):
        return(chi2, flux, background)
    else:
        return(chi2)

def gplusb(sampling, image):
    x0 = np.log(np.zeros(1) + 1.5)
    x = optimize.fmin(gplusb_chi2, x0, (sampling, image))
    chi2, f, b = gplusb_chi2(x, sampling, image, return_params=True)
    model = gplusb_image(sampling, sigma=np.exp(x[0]), flux=f, background=b)
    return(np.exp(x[0]), f, b, model)

def gplusm_chi2(x, sampling, image, mask=None, return_params=False):
    sigma = np.exp(x[0])
    g = gplusb_image(sampling, sigma=sigma, flux=1., background=0.)
    r = np.sqrt(sampling.xgrid**2 + sampling.ygrid**2).flatten()
    ii = np.where((r > 3.5) & (r < np.abs(sampling.xgrid.max()) - 1))[0]
    background = np.median(image.flatten()[ii])
    flux = (g * (image - background)).sum() / (g * g).sum()
    chi2arr = ((image - gplusb_image(sampling, sigma=sigma, flux=flux, background=background))**2)
    chi2 = chi2arr.flatten().sum()
    if(return_params):
        return(chi2, flux, background)
    else:
        return(chi2)

def gplusm(sampling, image):
    x0 = np.log(np.zeros(1) + 1.5)
    x = optimize.fmin(gplusm_chi2, x0, (sampling, image))
    chi2, f, b = gplusm_chi2(x, sampling, image, return_params=True)
    model = gplusb_image(sampling, sigma=np.exp(x[0]), flux=f, background=b)
    return(np.exp(x[0]), f, b, model)
