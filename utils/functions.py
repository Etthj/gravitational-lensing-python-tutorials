# This file is part of Gravitational Lensing tutorials: A collection of tutorials to learn how to simulate strong gravitational lenses.
#
# These tutorials are free to use.

from astropy.constants import c, G
import numpy as np
from scipy.optimize import brentq

import lenstronomy.Util.param_util as param_util                                  
from lenstronomy.LensModel.lens_model import LensModel                            
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver 

class sie_lens(object):
    """
    Python object that represents a singular isothermal ellipsoid (SIE) lens in strong gravitational lensing. 
    """  
    def __init__(self, co, zl, zs, sigmav, f, pa):
        """
        Initialize the SIE object.
        Demensionless units are used. The scale angle is given by the SIS Einstein radius.
        """
        self.sigmav=sigmav # velocity dispersion
        self.co=co # cosmological model
        self.zl=zl # lens redshift
        self.zs=zs # source redshift
        self.f=f # axis ratio
        self.pa=pa*np.pi/180.0 # position angle
        # compute the angular diameter distances:
        self.dl=self.co.angular_diameter_distance(self.zl)
        self.ds=self.co.angular_diameter_distance(self.zs)
        self.dls=self.co.angular_diameter_distance_z1z2(self.zl,self.zs)
        # calculate the Einstein radius of the SIS lens in arcsec
        self.theta0=np.rad2deg((4.0*np.pi*sigmav**2/(c.to("km/s"))**2*self.dls/self.ds).value)*3600.0 # eq. 5.50 p.194
        
    def delta(self,f,phi):
        return np.sqrt(np.cos(phi-self.pa)**2+self.f**2*np.sin(phi-self.pa)**2)

    def kappa(self,x,phi):
        """
        Convergence for the SIE lens at position (x,phi) in polar coordinates.
        """
        return (np.sqrt(self.f)/2.0/x/self.delta(self.f,phi))

    def gamma(self,x,phi):
        """
        Shear for the SIE lens at position (x,phi) in polar coordinates.
        """
        return (-self.kappa(x,phi)*np.cos(2.0*phi-self.pa),
                -self.kappa(x,phi)*np.sin(2.0*phi-self.pa))

    def mu(self,x,phi):
        """
        Magnification for the SIE lens at position (x,phi) in polar coordinates.
        """
        ga1,ga2=self.gamma(x,phi)
        ga=np.sqrt(ga1*ga1+ga2*ga2)
        return 1.0/(1.0-self.kappa(x,phi)-ga)/(1.0-self.kappa(x,phi)+ga)

    def psi_tilde(self,phi):
        """
        Angular part of the lensing potential at the polar angle phi.
        """
        if (self.f < 1.0):
            fp=np.sqrt(1.0-self.f**2)
            return np.sqrt(self.f)/fp*\
            (np.sin(phi-self.pa)*np.arcsin(fp*np.sin(phi-self.pa))+np.cos(phi-self.pa)*np.arcsinh(fp/self.f*np.cos(phi-self.pa)))
        else:
            return (1.0)

    def psi(self,x,phi):
        """
        Lensing potential at polar coordinates (x,phi).
        """
        psi=x*self.psi_tilde(phi)
        return psi

    def alpha(self,phi):
        """
        Deflection angle as a function of the polar angle phi.
        """
        fp=np.sqrt(1.0-self.f**2)
        a1=np.sqrt(self.f)/fp*np.arcsinh(fp/self.f*np.cos(phi))
        a2=np.sqrt(self.f)/fp*np.arcsin(fp*np.sin(phi))
        return a1,a2

    def cut(self,phi_min=0,phi_max=2.0*np.pi,nphi=200):
        """
        Coordinates of the points on the cut.
        The arguments phi_min, phi_max, nphi define the range of polar angle used.
        """
        phi=np.linspace(phi_min,phi_max,nphi)
        y1_,y2_=self.alpha(phi)
        y1 = y1_ * np.cos(self.pa) - y2_ * np.sin(self.pa)
        y2 = y1_ * np.sin(self.pa) + y2_ * np.cos(self.pa)
        return -y1, -y2

    def tan_caustic(self,phi_min=0,phi_max=2.0*np.pi,nphi=250):
        """
        Coordinates of the points on the tangential caustic.
        """
        phi=np.linspace(phi_min,phi_max,nphi)
        delta=np.sqrt(np.cos(phi)**2+self.f**2*np.sin(phi)**2)
        a1,a2=self.alpha(phi)
        y1_=np.sqrt(self.f)/delta*np.cos(phi)-a1
        y2_=np.sqrt(self.f)/delta*np.sin(phi)-a2
        y1 = y1_ * np.cos(self.pa) - y2_ * np.sin(self.pa)
        y2 = y2_ * np.sin(self.pa) - y2_ * np.cos(self.pa)
        return y1, y2

    def tan_cc(self,phi_min=0,phi_max=2.0*np.pi,nphi=1000):
        """
        Coordinates of the points on the tangential critical line.
        """
        phi=np.linspace(phi_min,phi_max,nphi)
        delta=np.sqrt(np.cos(phi)**2+self.f**2*np.sin(phi)**2)
        r=np.sqrt(self.f)/delta
        x1=r*np.cos(phi+self.pa)
        x2=r*np.sin(phi+self.pa)
        return (x1,x2)

    def x_ima(self,y1,y2,phi):
        """
        Distance of the image from the lens center.
        """
        x=y1*np.cos(phi)+y2*np.sin(phi)+(self.psi_tilde(phi+self.pa))
        return x

    def phi_ima(self,y1,y2,checkplot=True,eps=0.001,nphi=100):
        """
        Solve the lens Equation for a given source position (y1,y2).
        """
        # source position in the frame where the lens major axis is along the x_2 axis.
        y1_ = y1 * np.cos(self.pa) + y2 * np.sin(self.pa)
        y2_ = - y1 * np.sin(self.pa) + y2 * np.cos(self.pa)

        def phi_func(phi):
            a1,a2=self.alpha(phi)
            func=(y1_+a1)*np.sin(phi)-(y2_+a2)*np.cos(phi)
            return func

        # Evaluate phi_func and the sign of phi_func on an array of polar angles
        U=np.linspace(0.,2.0*np.pi+eps,nphi)
        c = phi_func(U)
        s = np.sign(c)
        phi=[]
        xphi=[]
        # loop over polar angles
        for i in range(len(U)-1):
            # if two polar angles bracket a zero of phi_func,
            # use Brent's method to find exact solution
            if s[i] + s[i+1] == 0: # opposite signs
                u = brentq(phi_func, U[i], U[i+1])
                z = phi_func(u)
                if np.isnan(z) or abs(z) > 1e-3:
                    continue
                x=self.x_ima(y1_,y2_,u)
                # append solution to a list if it corresponds to radial
                # distances x>0; discard otherwise (spurious solutions)
                if (x>0):
                    phi.append(u)
                    xphi.append(x)

        # convert lists to numpy arrays
        xphi=np.array(xphi)
        phi=np.array(phi)

        # returns radii and polar angles of the images. Add position angle
        # to go back to the rotated frame of the lens.
        return xphi, phi+self.pa, self.theta0

def Lenstronomy_GL(q,phi,zl,zs,gamma_ext,psi_ext,y1,y2,tE=0):
    gamma1, gamma2 = param_util.shear_polar2cartesian(phi=psi_ext, gamma=gamma_ext)
    e1, e2 = param_util.phi_q2_ellipticity(phi=phi, q=q)
    lens_model = LensModel(lens_model_list=['SIE', 'SHEAR'], z_lens=zl, z_source=zs)
    kwargs_lens = [{'theta_E': tE, 'e1': e1, 'e2': e2, 'center_x': 0, 'center_y': 0},
                   {'gamma1': gamma1, 'gamma2': gamma2}]
    lensEquationSolver = LensEquationSolver(lensModel=lens_model)
    x_source = y1
    y_source = y2
    x_img, y_img = lensEquationSolver.image_position_from_source(kwargs_lens=kwargs_lens, sourcePos_x=x_source, sourcePos_y=y_source)
    mag = lens_model.magnification(x_img, y_img, kwargs_lens)
    return x_img, y_img, mag