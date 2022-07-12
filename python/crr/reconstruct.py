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
import scipy.linalg
import fitsio


class Reconstruct(object):
    """Reconstruction of 2D grid from samples

    Parameters
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

    Attributes
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

    Notes
    -----

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

        Notes
        -----

        Sets attributes xgrid, ygrid. Grid points have coordinates
        0..(nx-1) and 0..(ny-1).
"""
        self.xgrid = np.arange(self.nx, dtype=np.float32)
        self.xgrid = np.outer(np.ones(self.ny, dtype=np.float32), self.xgrid)
        self.ygrid = np.arange(self.ny, dtype=np.float32)
        self.ygrid = np.outer(self.ygrid, np.ones(self.nx, dtype=np.float32))
        return

    def set_tlambda(self, tlambda=None):
        """Set Tikhonov parameter

        Parameters
        ----------

        tlambda : np.float32
            Tikhonov parameter in SVD pseudo-inversion

        Notes
        -----

        Sets attribute tlambda.

        This parameter, if set sufficiently small, should have no
        discernible effects on the result.
"""
        self.tlambda = tlambda
        return

    def set_Amatrix(self):
        """Create the problem matrix

        Notes
        -----

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
        for i in np.arange(M, dtype=int):
            f = self.psf(self.x - xg[i], self.y - yg[i], i)
            self.A[:, i] = f
        return

    def psf(self, x=None, y=None, i=None):
        """Returns point source response at x, y associated with sample i

        Parameters
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

    def set_svd(self, delete=True):
        """Perform SVD on the model matrix A

        Parameters
        ----------

        delete : bool
            delete the A matrix along the way to conserve memory
        
        Comments
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
        if(delete):
            self.A = None
        (U, S, VT, info) = scipy.linalg.lapack.sgesvd(Atilde, full_matrices=False)

        Atilde = 0

        self.S = S

        # Calculate normalization of Q's rows without
        # actually instantiating Q
        Qnorminv = np.einsum('ik,k,jk->i', VT.T, self.S,
                             VT.T, optimize='greedy')
        self.Qnorm = 1. / Qnorminv

        # Instead of calculating R, recognize that it will
        # always be multiplied by V, and R.V is a smaller
        # matrix to store, and easier to calculate. 
        #   R = Qn.Q = Qn.V.S.VT
        #   R.V = Qn.V.S.VT.V = Qn.V.S
        VS = VT.T.dot(np.diag(self.S))
        self.RV = np.einsum('i,ij->ij', self.Qnorm, VS,
                            optimize='greedy')
        
        self.UT = U.transpose()

        self.did_svd = True

        return

    def set_weights(self, F_weights=False, G_weights=True, delete=True):
        """Set weights for reconstruction

        Parameters
        ----------

        F_weights : bool
            if true, set the linear deconvolution weights F_weights

        G_weights : bool
            if true, set the covariance regularized weight weights

        delete : bool
            delete the V or RV array and A matrix to conserve memory

        Comments
        --------

        Performs SVD if it had not been previously performed.
"""
        if(not self.did_svd):
            self.set_svd()

        iSstar = np.float32(self.S / (self.S**2 + self.tlambda**2))

        if(G_weights):
            self.weights = self.RV.dot(np.diag(iSstar)).dot(self.UT).dot(self.Ntilde)
            if(delete):
                self.RV = None
                self.A = None

        if(F_weights):
            self.F_weights = self.V.dot(np.diag(iSstar)).dot(self.UT).dot(self.Ntilde)
            if(delete):
                self.V = None
                self.A = None
        return

    def save_model(self, filename=None, clobber=True):
        """Saves the model matrix and SVD results

        Parameters
        ----------

        filename : str
            name of output file

        clobber : bool
            whether to clobber (if True), or append to (if False) that file
            (default True)

        Comments
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
            fitsio.write(filename, self.RV, extname='RV',
                         clobber=False)
        except:
            pass

    def load_model(self, filename=None, ext_start=0):
        """Load saved model matrix and SVD results

        Parameters
        ----------

        filename : str
            name of output file

        ext_start : int
            first HDU of file to read from (default 0)

        Comments
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
        self.RV = fitsio.read(filename, ext=12 + ext_start)
        self.did_svd = True
        return

    def apply_weights(self, f=None):
        grid = self.weights.dot(f).reshape(self.nx, self.ny)
        return(grid)

    def coverage(self):
        coverage = self.A.sum(axis=1)
        return(coverage)


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

        Returns
        -------
    
        coverage : 2D ndarray of np.float32
            contribution of input pixel to all output pixels
            (between 0 and 1)

        Notes
        -----

        Low coverage implies that the input pixel does not much
        affect any output pixels.
"""
        coverage = self.A.sum(axis=1).reshape((self.inx, self.iny))
        return(coverage)
