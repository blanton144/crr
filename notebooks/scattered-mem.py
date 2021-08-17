#!/usr/bin/env python

import numpy as np
import fitsio
import scipy.optimize as optimize
from sampling import Sampling

sampling = Sampling(nsamples=1000)
sampling.set_flux(total_flux=1000., noise=0.001)


def mem_function(u, A, f, llambda):
    Ar = (A.dot(u) - f)
    As = (Ar**2).sum()
    Bs = (u * np.log(u)).sum()
    val = As + llambda * Bs
    grad = 2. * A.T.dot(Ar) + llambda * (1. + np.log(u))
    return (val, grad)


def mem_fit(sampling, llambda=1.e-2):
    S_M0 = np.ones(sampling.nx * sampling.ny)
    bounds = zip([1.e-5] * len(S_M0), [None] * len(S_M0))
    bounds = [x for x in bounds]
    flux = sampling.flux
    results = optimize.minimize(mem_function, S_M0,
                                args=(sampling.A, flux, llambda),
                                method='L-BFGS-B', jac=True,
                                bounds=bounds)
    return(results.x.reshape(sampling.nx, sampling.ny))


def mem_lambda_function(lnlambda, sampling):
    x = mem_fit(sampling, llambda=np.exp(lnlambda))
    recon = sampling.A.dot(x.flatten())
    chi2 = ((recon - sampling.flux)**2 * sampling.ivar).sum()
    val = chi2 - len(recon)
    print(val)
    return(val)


def mem_all(sampling):
    bracket = [np.log(1.e-5), np.log(1.e+16)]
    try:
        rootresults = optimize.root_scalar(mem_lambda_function,
                                           args=(sampling),
                                           method='brentq',
                                           bracket=bracket)
        lnl = rootresults.root
    except ValueError:
        if(mem_lambda_function(bracket[1], sampling) < 0.):
            lnl = bracket[1]
        if(mem_lambda_function(bracket[0], sampling) > 0.):
            lnl = bracket[0]

    S_M = mem_fit(sampling, llambda=np.exp(lnl))
    print(np.exp(lnl))
    return(S_M)


def Sexpected(sampling, total_flux=1000., noise=10., nsample=5000,
              xcen=0., ycen=0.):
    Sarr = np.zeros((nsample, sampling.nx, sampling.ny))
    larr = np.zeros(nsample)
    for i in np.arange(nsample):
        sampling.set_flux(total_flux=total_flux, noise=noise, xcen=xcen, ycen=ycen)
        Sarr[i, :, :] = mem_all(sampling)
    Sexp = Sarr.mean(axis=0)
    return(Sarr, Sexp)


#Sarr_100, Sexp_100 = Sexpected(sampling, noise=100.)
#fitsio.write('mem_sexp_100.fits', Sexp_100, clobber=True)
#fitsio.write('mem_sexp_100.fits', Sarr_100, clobber=False)

Sarr_30, Sexp_30 = Sexpected(sampling, noise=30.)
fitsio.write('mem_sexp_30.fits', Sexp_30, clobber=True)
fitsio.write('mem_sexp_30.fits', Sarr_30, clobber=False)

Sarr_1, Sexp_1 = Sexpected(sampling, noise=1.)
fitsio.write('mem_sexp_1.fits', Sexp_1, clobber=True)
fitsio.write('mem_sexp_1.fits', Sarr_1, clobber=False)
