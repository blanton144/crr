import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

matplotlib.rcParams['font.size'] = 30
matplotlib.rcParams['figure.figsize'] = [9., 8.]

# Function to create samples of delta function
class Sampling(object):
    def __init__(self, nsamples=500, xyrange=[-10.5, 10.5], 
                 sigmarange=[1., 4.], seed=10):
        np.random.seed(seed)
        self.nsamples = nsamples
        self.xylo = xyrange[0]
        self.xyhi = xyrange[1]
        self.sigmalo = sigmarange[0]
        self.sigmahi = sigmarange[1]
        self.x = self.xylo + (self.xyhi - self.xylo) * np.random.random(size=self.nsamples)
        self.y = self.xylo + (self.xyhi - self.xylo) * np.random.random(size=self.nsamples)
        self.sigma = self.sigmalo + (self.sigmahi - self.sigmalo) * np.random.random(size=self.nsamples)
        self.set_grid()
        self.Amatrix()
        self.flux = None
        self.flux_nonoise = None
        
    def fluxes(self, total_flux=1., xcen=0., ycen=0., noise=0., background=0.):
        return (background * np.pi * self.sigma**2 +
                total_flux * np.exp(- 0.5 * ((self.x - xcen)**2 + (self.y - ycen)**2) /
                                    self.sigma**2) / (2. * np.pi * self.sigma**2) +
                noise * np.random.normal(size=self.nsamples))
        
    def set_flux(self, total_flux=1., xcen=0., ycen=0., noise=0., background=0.):
        self.flux_nonoise = self.fluxes(total_flux=total_flux, xcen=xcen, ycen=ycen, noise=0.,
                                        background=background)
        self.flux = self.fluxes(total_flux=total_flux, xcen=xcen, ycen=ycen, noise=noise,
                                background=background)
        self.ivar = np.zeros(len(self.flux)) + 1. / noise**2
        return
    
    def set_grid(self, nx=21, ny=21):
        self.nx = nx
        self.ny = ny
        self.xgrid = self.xylo + (self.xyhi - self.xylo) * (np.arange(nx) + 0.5) / np.float32(nx)
        self.xgrid = np.outer(np.ones(ny), self.xgrid).flatten()
        self.ygrid = self.xylo + (self.xyhi - self.xylo) * (np.arange(ny) + 0.5) / np.float32(ny)
        self.ygrid = np.outer(self.ygrid, np.ones(nx)).flatten()
        return
    
    def Amatrix(self):
        M = len(self.ygrid)
        self.A = np.zeros((self.nsamples, M))
        for i in np.arange(M):
            f = self.fluxes(xcen=self.xgrid[i], ycen=self.ygrid[i])
            self.A[:, i] = f.flatten() 
        return
    
    def imshow(self, S=None, vmin=-1., vmax=5., nonoise=False, linear=False,
               nopoints=False):
        if(S is not None):
            myargs = {'interpolation': 'nearest',
                      'origin': 'lower',
                      'cmap': cm.Greys}
            if(linear):
                Sshow = S
            else:
                Sshow = np.arcsinh(S)
            plt.imshow(np.arcsinh(S), **myargs, vmax=vmax, vmin=vmin,
                       extent=[self.xylo - 0.5, self.xyhi + 0.5,
                               self.xylo - 0.5, self.xyhi + 0.5])
        if(nonoise):
            fplot = self.flux_nonoise
        else:
            fplot = self.flux
        if((linear is False) & (fplot is not None)):
            fplot = np.arcsinh(fplot)
        if(nopoints is False):
            plt.scatter(self.x, self.y, c=fplot, s=self.sigma * 3,
                        vmax=vmax, vmin=vmin, cmap=cm.Greys)
        plt.plot([0], [0], '+')
        plt.xlim([self.xylo - 0.5, self.xyhi + 0.5])
        plt.ylim([self.xylo - 0.5, self.xyhi + 0.5])
        plt.xlabel('x')
        plt.ylabel('y')
        if(linear):
            plt.colorbar(label='$f$')
        else:
            plt.colorbar(label='$\sinh^{-1} f$')
        return

# Function to create samples of delta function
class SamplingRect(Sampling):
    def __init__(self, nx=21, ny=21, xyrange=[-10.5, 10.5], 
                 sigma=1., seed=10):
        np.random.seed(seed)
        self.nsamples = nx * ny
        self.xylo = xyrange[0]
        self.xyhi = xyrange[1]
        self.sigmalo = sigma
        self.sigmahi = sigma
        self.set_grid()
        self.x = self.xgrid
        self.y = self.ygrid
        self.sigma = self.sigmalo + (self.sigmahi - self.sigmalo) * np.random.random(size=self.nsamples)
        self.Amatrix()
        return
