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
import fitsio


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
       X positions of output grid

    ygrid : (nx, ny) ndarray of np.float32
       Y positions of output grid

    Methods:
    -------

    set_grid(tlambda): set the reconstruction grid
    set_Amatrix(): set the model matrix A
    set_tlambda(tlambda): set Tikhonov parameter
    set_svd(): run the SVD
    set_weights(): set the weights
    save_model(): save the model as a FITS file
    psf(x, y, i): model response at points (x, y) for sample i

    Notes:
    ------

    x, y are related to the grid points assuming the grid point
    locations in x and y are at (0..(nx-1)) and (0..(ny-1)).
"""
    def __init__(self, filename=None, nx=None, ny=None, x=None, y=None,
                 fivar=None):
        self.save_attr = {'nx': np.int32,
                          'ny': np.int32,
                          'nsample': np.int32,
                          'tlambda': np.float32}
        if(filename is not None):
            self.load_model(filename=filename)
            return

        self.x = x
        self.y = y
        self.nsample = np.int32(len(x))
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self._set_grid()
        if(fivar is None):
            self.fivar = np.ones(self.nsample, dtype=np.float32)
        else:
            self.fivar = np.float32(fivar)
        self.set_tlambda(np.float32(1.e-4))
        self.did_svd = False
        return
    
    def _set_grid(self):
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
        self.tlambda = tlambda
        return

    def set_Amatrix(self):
        """Create the problem matrix

        Notes:
        ------

        Sets attribute A, such that the predicted fluxes of the samples are
           A . m
        i.e. a grid point m_i will produce the response associated
        with a delta function input.
        
        Depends critically on the psf() method.
"""
        xg = self.xgrid.flatten()
        yg = self.ygrid.flatten()
        M = len(xg)
        self.A = np.zeros((len(self.fivar), M), dtype=np.float32)
        for i in np.arange(M):
            f = self.psf(self.x - xg[i], self.y - yg[i], i)
            self.A[:, i] = f
        return

    def psf(self, x=None, y=None, i=None):
        """Returns point source response at x, y associated with sample i

        Parameters:
        ----------

        x : np.float32 or ndarray of same
            x values to evaluate

        y : np.float32 or ndarray of same
            y values to evaluate
        
        i : np.int32 or int
            index of sample that is being evaluated
"""
        g = np.exp(- 0.5 * (x**2 + y**2)) / (2. * np.pi)
        return(g)

    def set_svd(self):
        """Perform SVD on the model matrix A
        
        Comments:
        --------

        Sets a number of attributes:
           Ntilde : diagonal input covariance, just 1s and 0s depending on ivar
           V, S, UT : derived from SVD
           Q : square root of covariance matrix
           Qnorm : diagonal matrix with normalization factors
           R : resolution matrix
           did_svd : set to True
"""
        self.Ntilde = np.diag(np.float32(self.fivar > 0))
        Atilde = self.Ntilde.dot(self.A)  # since it is 0 or 1, this works
        (U, S, VT) = np.linalg.svd(Atilde, full_matrices=False)

        V = VT.transpose()
        Q = V.dot(np.diag(S)).dot(VT)
        self.Q = Q
        self.V = V
        self.VT = VT
        self.Qnorm = np.diag(1. / Q.sum(axis=1))
        self.R = self.Qnorm.dot(Q)
        self.UT = U.transpose()
        self.S = S

        self.did_svd = True

        return

    def set_weights(self, F_weights=False, G_weights=True):
        """Set weights for reconstruction

        Parameters:
        ----------

        F_weights : bool
            if true, set the linear deconvolution weights F_weights

        G_weights : bool
            if true, set the covariance regularized weight weights

        Comments:
        --------

        Performs SVD if it had not been previously performed.
"""
        if(not self.did_svd):
            self.set_svd()

        iSstar = self.S / (self.S**2 + self.tlambda**2)

        if(G_weights):
            self.weights = self.R.dot(self.V).dot(np.diag(iSstar)).dot(self.UT).dot(self.Ntilde)

        if(F_weights):
            self.F_weights = self.V.dot(np.diag(iSstar)).dot(self.UT).dot(self.Ntilde)
        return

    def save_model(self, filename=None, clobber=True):
        """Saves the model matrix and SVD results

        Parameters:
        ----------

        filename : str
            name of output file

        clobber : bool
            whether to clobber (if True), or append to (if False) that file
            (default True)

        Comments:
        --------

        Recovered with load_model() method, or initializing with
        filename argument.
"""
        hdr = dict()
        for at in self.save_attr:
            hdr[at.upper()] = getattr(self, at)
        fitsio.write(filename, self.A, extname='A',
                     header=hdr, clobber=clobber)
        fitsio.write(filename, self.x, extname='X',
                     clobber=False)
        fitsio.write(filename, self.y, extname='Y',
                     clobber=False)
        fitsio.write(filename, self.fivar, extname='FIVAR',
                     clobber=False)
        fitsio.write(filename, self.xgrid, extname='XGRID',
                     clobber=False)
        fitsio.write(filename, self.ygrid, extname='YGRID',
                     clobber=False)
        try:
            fitsio.write(filename, self.Ntilde, extname='NTILDE',
                         clobber=False)
            fitsio.write(filename, self.V, extname='V',
                         clobber=False)
            fitsio.write(filename, self.UT, extname='UT',
                         clobber=False)
            fitsio.write(filename, self.S, extname='S',
                         clobber=False)
            fitsio.write(filename, self.Q, extname='Q',
                         clobber=False)
            fitsio.write(filename, self.Qnorm, extname='QNORM',
                         clobber=False)
            fitsio.write(filename, self.R, extname='R',
                         clobber=False)
        except:
            pass

    def load_model(self, filename=None, ext_start=0):
        """Load saved model matrix and SVD results

        Parameters:
        ----------

        filename : str
            name of output file

        ext_start : int
            first HDU of file to read from (default 0)

        Comments:
        --------

        Loads the model problem.
        Assumes file was created with save_model() method.
"""
        hdr = fitsio.read_header(filename, ext=0 + ext_start)
        for at in self.save_attr:
            setattr(self, at.lower(), np.cast[self.save_attr[at]](hdr[at]))
        self.A = fitsio.read(filename, ext=0 + ext_start)
        self.x = fitsio.read(filename, ext=1 + ext_start)
        self.y = fitsio.read(filename, ext=2 + ext_start)
        self.fivar = fitsio.read(filename, ext=3 + ext_start)
        self.xgrid = fitsio.read(filename, ext=4 + ext_start)
        self.ygrid = fitsio.read(filename, ext=5 + ext_start)
        self.Ntilde = fitsio.read(filename, ext=6 + ext_start)
        self.V = fitsio.read(filename, ext=7 + ext_start)
        self.UT = fitsio.read(filename, ext=8 + ext_start)
        self.S = fitsio.read(filename, ext=9 + ext_start)
        self.Q = fitsio.read(filename, ext=10 + ext_start)
        self.Qnorm = fitsio.read(filename, ext=11 + ext_start)
        self.R = fitsio.read(filename, ext=12 + ext_start)
        self.did_svd = True
        return

    def apply_weights(self, f=None):
        grid = self.weights.dot(f).reshape(self.nx, self.ny)
        return(grid)

    def coverage(self):
        coverage = self.A.sum(axis=1)
        return(coverage)


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
    def __init__(self, filename=None, inwcs=None, outwcs=None, trim=None, pad=0):
        self.save_attr = {'nx': np.int32,
                          'ny': np.int32,
                          'nsample': np.int32,
                          'inx': np.int32,
                          'iny': np.int32,
                          'tlambda': np.float32}

        if(filename is not None):
            self.load_model(filename=filename)
            return

        owcs = outwcs
        self.nx = np.int32(owcs.pixel_shape[0]) + pad * 2
        self.ny = np.int32(owcs.pixel_shape[1]) + pad * 2

        iwcs = inwcs
        self.iny = iwcs.pixel_shape[0]
        self.inx = iwcs.pixel_shape[1]
        iy = np.outer(np.arange(self.iny, dtype=np.float32),
                      np.ones(self.inx, dtype=np.float32))
        ix = np.outer(np.ones(self.iny, dtype=np.float32),
                      np.arange(self.inx, dtype=np.float32))
        self.trim = trim
        if(self.trim is not None):
            iy = iy[self.trim]
            ix = ix[self.trim]

        ra, dec = iwcs.all_pix2world(ix.flatten(), iy.flatten(), 0,
                                     ra_dec_order=True)

        self.x, self.y = owcs.all_world2pix(ra, dec, 0, ra_dec_order=True)
        self.x = self.x + pad
        self.y = self.y + pad

        self._set_grid()

        self.nsample = np.int32(len(self.x))
        self.fivar = np.ones(self.nsample, dtype=np.float32)
        self.set_tlambda(np.float32(1.e-4))
        self.did_svd = False
        return

    def save_model(self, filename=None, clobber=True):
        super().save_model(filename=filename, clobber=clobber)
        if(self.trim is not None):
            fitsio.write(filename, np.int8(self.trim), clobber=False)
        return

    def load_model(self, filename=None, ext_start=0):
        super().load_model(filename=filename)
        self.trim = fitsio.read(filename, ext=ext_start + 13) > 0
        return

    def psf_samples(self):
        """Return values corresponding to a PSF at center
"""
        xcen = np.float32(self.nx - 1) * 0.5
        ycen = np.float32(self.ny - 1) * 0.5
        f = self.psf(self.x.flatten() - xcen,
                     self.y.flatten() - ycen, 0)
        return(f)

    def coverage(self):
        """Return coverage of each input pixel

        Returns:
        -------
    
        coverage : 2D ndarray of np.float32
            contribution of input pixel to all output pixels
            (between 0 and 1)

        Comments:
        --------

        Low coverage implies that the input pixel does not much
        affect any output pixels.
"""
        coverage = self.A.sum(axis=1).reshape((self.inx, self.iny))
        return(coverage)
