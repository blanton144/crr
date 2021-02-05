# encoding: utf-8
#
# @Author: Michael Blanton
# @Date: Jan 10, 2020
# @Filename: reconstruct.py
# @License: BSD 3-Clause
# @Copyright: Michael Blanton


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import multiprocessing


class Reconstruct(object):
    """Reconstruction of 2D grid from samples

    Parameters:
    ----------

    nx : int, np.int32
       number of columns in grid

    ny : int, np.int32
       number of rows in grid

    x : ndarray of np.float32
       x positions of samples

    y : ndarray of np.float32
       y positions of samples

    fivar : ndarray of np.float32
       inverse variance of values at samples

    Attributes:
    ----------

    nx : np.int32
       number of columns in grid

    ny : np.int32
       number of rows in grid

    x : ndarray of np.float32
       x positions of samples

    y : ndarray of np.float32
       y positions of samples

    fivar : ndarray of np.float32
       inverse variance of values at samples (default all ones)

    nsample : np.int32
       number of samples

    xgrid : (nx, ny) ndarray of np.float32
       positions of grid

    Methods:
    -------

    psf(x, y, i): model response at points (x, y) for sample i

    Notes:
    ------

    x, y are related to the grid points assuming the grid point
    locations in x and y are at (0..(nx-1)) and (0..(ny-1).
"""

    def __init__(self, nx=None, ny=None, x=None, y=None, fivar=None):
        self.x = x
        self.y = y
        self.nsample = np.int32(len(x))
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        if(fivar is None):
            self.fivar = np.ones(self.nsample, dtype=np.float32)
        else:
            self.fivar = np.float32(fivar)
        self.tlambda = np.float32(1.e-5)
        return

    def set_tlambda(self, tlambda=None):
        """Set Tikhonov parameter

        Parameters:
        ----------

        tlambda : np.float32
            Tikhonov parameter in SVD pseudo-inversion

        Notes:
        ------

        Sets attribute tlambda.

        This parameter, if set sufficiently small, should have no
        discernible effects on the result.
"""

    def set_grid(self):
        """Create the reconstruction grid

        Notes:
        ------

        Sets attributes xgrid, ygrid. Grid points have coordinates
        0..(nx-1) and 0..(ny-1).
"""
        self.xgrid = np.arange(self.nx)
        self.xgrid = np.outer(np.ones(self.ny), self.xgrid)
        self.ygrid = np.arange(self.ny)
        self.ygrid = np.outer(self.ygrid, np.ones(self.nx))
        return

    def set_Amatrix(self):
        """Create the problem matrix

        Notes:
        ------

        Sets attribute A.
        Depends critically on the psf() method.
"""
        xg = self.xgrid.flatten()
        yg = self.ygrid.flatten()
        M = len(xg)
        self.A = np.zeros((len(self.fivar), M))
        for i in np.arange(M):
            f = self.psf(self.x - xg[i], self.y - yg[i], i)
            self.A[:, i] = f
        return

    def psf(self, x, y, i):
        g = np.exp(- 0.5 * (x**2 + y**2)) / (2. * np.pi)
        return(g)

    def set_weights(self, F_weights=False):
        Ntilde = np.diag(np.float32(self.fivar > 0))
        Atilde = Ntilde.dot(self.A)  # since it is 0 or 1, this works
        (U, S, VT) = np.linalg.svd(Atilde, full_matrices=False)

        iSstar = S / (S**2 + self.tlambda**2)

        V = VT.transpose()
        Q = V.dot(np.diag(S)).dot(VT)
        self.Q = Q
        self.V = V
        self.VT = VT
        Qnorm = np.diag(1. / Q.sum(axis=1))
        self.R = Qnorm.dot(Q)

        UT = U.transpose()
        self.weights = self.R.dot(V).dot(np.diag(iSstar)).dot(UT).dot(Ntilde)

        if(F_weights):
            self.F_weights = V.dot(np.diag(iSstar)).dot(UT).dot(Ntilde)
        return

    def apply_weights(self, f=None):
        grid = self.weights.dot(f).reshape(self.nx, self.ny)
        return(grid)


class ReconstructStitch(Reconstruct):
    """Reconstruction of 2D grid from samples, using stitching

    Parameters:
    ----------

    nx : int, np.int32
       number of columns in grid

    ny : int, np.int32
       number of rows in grid

    x : ndarray of np.float32
       x positions of samples

    y : ndarray of np.float32
       y positions of samples

    fivar : ndarray of np.float32
       inverse variance of values at samples

    dstitch : np.int32
       maximum size of each stitching

    Attributes:
    ----------

    nx : np.int32
       number of columns in grid

    ny : np.int32
       number of rows in grid

    x : ndarray of np.float32
       x positions of samples

    y : ndarray of np.float32
       y positions of samples

    fivar : ndarray of np.float32
       inverse variance of values at samples (default all ones)

    nsample : np.int32
       number of samples

    xgrid : (nx, ny) ndarray of np.float32
       positions of grid

    dstitch : np.int32
       minimum size of each stitching

    Methods:
    -------

    psf(x, y, i): model response at points (x, y) for sample i

    Notes:
    ------

    Performs a suite of SVDs, of minimum size dstitch and maximum size
    2*dstitch, arranged to cover the entire desired output
    grid. Weights are averaged.

    dstitch should be large enough to enclose most of the the power of 
    the largest PSF.

    x, y are related to the grid points assuming the grid point
    locations in x and y are at (0..(nx-1)) and (0..(ny-1).

    """

    def set_patches(self, pminsize=30, poverlap=14):
        """Determine patch positions
"""
        self.poverlap = poverlap

        self.npx = self.nx // (pminsize - self.poverlap) + 1
        self.pxsize = self.poverlap + self.nx // (self.npx - 1)

        pxcen = np.int32(np.floor(self.nx / self.npx *
                                  (np.arange(self.npx) + 0.5)))
        print(pxcen)
        pxst = pxcen - (self.pxsize // 2)
        pxnd = pxst + self.pxsize - 1
        pxst[0] = 0
        pxnd[0] = pxst[0] + (self.pxsize - 1)
        pxnd[-1] = self.nx - 1
        pxst[-1] = pxnd[-1] - (self.pxsize - 1)

        self.npy = self.ny // (pminsize - self.poverlap) + 1
        self.pysize = self.poverlap + self.ny // (self.npy - 1)

        pycen = np.int32(np.floor(self.ny / self.npy *
                                  (np.arange(self.npy) + 0.5)))
        pyst = pycen - (self.pysize // 2)
        pynd = pyst + self.pysize - 1
        pyst[0] = 0
        pynd[0] = pyst[0] + (self.pysize - 1)
        pynd[-1] = self.ny - 1
        pyst[-1] = pynd[-1] - (self.pysize - 1)

        self.np = self.npx * self.npy
        patch_dtype = np.dtype([('pxst', np.int32),
                                ('pxnd', np.int32),
                                ('pyst', np.int32),
                                ('pynd', np.int32)])
        self.patches = np.zeros((self.npx, self.npy), dtype=patch_dtype)
        for i in np.arange(self.npx, dtype=np.int32):
            for j in np.arange(self.npy, dtype=np.int32):
                self.patches['pxst'][i, j] = pxst[i]
                self.patches['pxnd'][i, j] = pxnd[i]
                self.patches['pyst'][i, j] = pyst[j]
                self.patches['pynd'][i, j] = pynd[j]

        return

    def f_in_patch(self, i, j):
        fin = np.where((self.x >
                        self.patches['pxst'][i, j] - self.poverlap) &
                       (self.x <
                        self.patches['pxnd'][i, j] + self.poverlap) &
                       (self.y >
                        self.patches['pyst'][i, j] - self.poverlap) &
                       (self.y <
                        self.patches['pynd'][i, j] + self.poverlap))[0]
        return(fin)

    def grid_in_patch(self, i, j):
        gin = np.where((self.xgrid.flatten() >= self.patches['pxst'][i, j]) &
                       (self.xgrid.flatten() <= self.patches['pxnd'][i, j]) &
                       (self.ygrid.flatten() >= self.patches['pyst'][i, j]) &
                       (self.ygrid.flatten() <= self.patches['pynd'][i, j]))[0]
        return(gin)

    def calc_weights_patch(self, tij):
        i = tij[0]
        j = tij[1]
        print("{i} {j}".format(i=i, j=j))
        fin = self.f_in_patch(i, j)
        gin = self.grid_in_patch(i, j)

        ix = np.ix_(fin, gin)
        Atrim = self.A[ix]

        Ntilde = np.diag(np.float32(self.fivar[fin] > 0))
        Atilde = Ntilde.dot(Atrim)  # since it is 0 or 1, this works

        (U, S, VT) = np.linalg.svd(Atilde, full_matrices=False)

        iSstar = S / (S**2 + self.tlambda**2)

        V = VT.transpose()
        Q = V.dot(np.diag(S)).dot(VT)
        Qnorm = np.diag(1. / Q.sum(axis=1))
        R = Qnorm.dot(Q)

        UT = U.transpose()

        pweights = R.dot(V).dot(np.diag(iSstar)).dot(UT).dot(Ntilde)
        return(i, j, pweights)

    def set_weights(self):
        patches = []
        for i in np.arange(self.npx):
            for j in np.arange(self.npy):
                patches.append((i, j))
        self.weights_patch = [[np.zeros(0)
                               for j in range(self.npy)]
                              for i in range(self.npx)]
        with multiprocessing.Pool() as p:
            for i, j, w in p.imap_unordered(self.calc_weights_patch, patches):
                self.weights_patch[i][j] = w

        self.weights_weight = np.zeros((self.nx * self.ny,
                                        self.nsample), dtype=np.float32)
        weights_sum = np.zeros((self.nx * self.ny,
                                self.nsample), dtype=np.float32)
        for i in np.arange(self.npx):
            for j in np.arange(self.npy):
                fin = self.f_in_patch(i, j)
                gin = self.grid_in_patch(i, j)

                ix = np.ix_(gin, fin)

                xfac = ((self.xgrid.flatten()[gin] -
                         self.patches['pxst'][i, j]) /
                        (self.patches['pxnd'][i, j] -
                         self.patches['pxst'][i, j]))
                weights_xapodize = np.exp(- 0.5 * (xfac - 0.5)**2 /
                                          (0.15**2))
                yfac = ((self.ygrid.flatten()[gin] -
                         self.patches['pyst'][i, j]) /
                        (self.patches['pynd'][i, j] -
                         self.patches['pyst'][i, j]))
                weights_yapodize = np.exp(- 0.5 * (yfac - 0.5)**2 /
                                          (0.15**2))
                weights_apodize = weights_xapodize * weights_yapodize
                weights_apodize_x = np.outer(weights_apodize,
                                             np.ones(len(fin)))
                self.weights_weight[ix] = (self.weights_weight[ix] +
                                           weights_apodize_x)
                weights_sum[ix] = (weights_sum[ix] +
                                   self.weights_patch[i][j] *
                                   weights_apodize_x)

        ii = np.where(self.weights_weight > 0)
        self.weights = np.zeros((self.nx * self.ny,
                                 self.nsample), dtype=np.float32)
        self.weights[ii] = weights_sum[ii] / self.weights_weight[ii]
        return


class ReconstructWCS(Reconstruct):
    def __init__(self, inwcs=None, outwcs=None, trim=None, pad=0):
        owcs = outwcs
        self.nx = np.int32(owcs.pixel_shape[0]) + pad * 2
        self.ny = np.int32(owcs.pixel_shape[1]) + pad * 2

        iwcs = inwcs
        iny = iwcs.pixel_shape[0]
        inx = iwcs.pixel_shape[1]
        iy = np.outer(np.arange(iny, dtype=np.float32),
                      np.ones(inx, dtype=np.float32))
        ix = np.outer(np.ones(iny, dtype=np.float32),
                      np.arange(inx, dtype=np.float32))
        if(trim is not None):
            iy = iy[trim]
            ix = ix[trim]

        ra, dec = iwcs.all_pix2world(ix.flatten(), iy.flatten(), 0,
                                     ra_dec_order=True)

        self.x, self.y = owcs.all_world2pix(ra, dec, 0, ra_dec_order=True)
        self.x = self.x + pad
        self.y = self.y + pad

        self.nsample = np.int32(len(self.x))
        self.fivar = np.ones(self.nsample, dtype=np.float32)
        self.tlambda = np.float32(1.e-5)
        return
