#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm  # For inverse of Gaussian CDF, norm.ppf
from scipy.stats import chi2  # For testing purposes, chi2.rvs for samples, chi2.cdf for cdf

class Gaussianizer(object):
    def __init__(self,rot_seed=42):
        """
        Initialise Gaussianation object with no steps.

        Input:
          rot_seed : Seed for random rotations
        """
        self.nSteps = 0
        self.steps = []
        self.trained = False
        self.rotation_seed = 42
        return

    def gaussianize(self,samp,ns,condition=None):
        """
        Given the set of input samples, Gaussianize them

        Input:
          samp      : Samples from the distribution to be Gaussianized
          ns        : Number of steps to take
          condition : (optional) Function defining stopping condition (not implemented)
        """
        for i in range(ns):
            self.train_step(samp)
        self.trained = True
        return
        
    def train_step(self,samp):
        """
        Given the input samples, add a new step to the Gaussianizer
        """
        self.nSteps += 1
        sw, rot, mean = rotate_samples(samp)
        sw, cdf_v, cdf = marginal_gaussianize(sw)
        newStep = Transform_Step(rot,mean,cdf_v,cdf)
        self.steps.append(newStep)
        return sw
        
    def forward_transform(self,y):
        x = y
        for i in range(self.nSteps):
            x = self.steps[i].forward(x)
        return x

    def inverse_transform(self,x):
        y = x
        for i in range(self.nStep-1,-1,-1):
            y = self.steps[i].backward(y)
        return y

    def take_step(self,samp,step_num):
        return self.steps[step_num].forward(samp)

    def invert_step(self,samp,step_num):
        return self.steps[step_num].backward(samp)
    
    def entropy(self):
        """
        Compute appropriate KL divergence as a measure of how Gaussianized the samples are.
        """
        return

class Transform_Step(object):
    def __init__(self,rot,mean,cdf_v,cdf):
        self.cdf_v = cdf_v
        self.cdf = cdf
        self.rot = rot
        self.mean = mean
        return

    def forward(x):
        x = x-self.mean[:,np.newaxis]
        return self.cdf(self.rot@x)

    def backward(y):
        y = self.cdf_inv(y)
        y = self.rot.T@y
        y = y + self.mean[:,np.newaxis]
        return y
    
    def cdf(samp):
        return

    def cdf_inv(samp):
        return

class Function_Estimator(object):
    def __init__(self,xv,yv):
        self.xvals = xv
        self.yvals = yv
        return

    # This is currently a horrible approximation
    def value(self,x):
        ii = np.argmin(x>self.xv)
        slope = (self.yv[ii+1]-self.yv[ii]) / (self.xv[ii+1]-self.xv[ii])
        return 

    def deriv(self,x):
        ii = np.argmin(x>xv)
        slope = (self.yv[ii+1]-self.yv[ii]) / (self.xv[ii+1]-self.xv[ii])
        return slope
    
# Here's an idea
# Do a merge type operation.  Rotate pairs
# Then rotate in the rotated pairs, etc.
def random_rotation(samp):
    """
    Randomly rotate between pairs of randomly chosen directions
    """
    d = samp.shape[0]
    shuffle = np.random.permutation(np.arange(d))

    # Write this
    return
    
# Make sure this is the correct thing to do
# Currently doing random rotations.  Need to fix
def rotate_samples(samp,use_cov=False):
    mu = np.mean(samp,axis=-1)
    sw = samp - np.mean(samp,axis=-1)[:,np.newaxis]
    if use_cov:
        cov = np.cov(sw,rowvar=True)
        u,sig,v = np.linalg.svd(cov,hermitian=True)
    else:
        v = rot_2d(np.random.uniform(low=0.,high=2.*np.pi))  # random rotations
    return v@sw, v, mu

# Currently using the ugly solution of sorting the indexed keys
def marginal_gaussianize(samp):
    n = samp.shape[-1]
    p_ = 0.5/n+np.arange(n)/n
    g_ = norm.ppf(p_)

    s = np.empty(samp.shape)
    ii = np.argsort(samp,axis=-1)
    jj = np.argsort(ii,axis=-1)  # Can I turn this into a single line
    c_ = np.sort(samp,axis=-1)
    return g_[jj], c_, p_

def compute_cdf_1d(x,xvals,cdf):
    eps = 0.1/cdf.size
    ii = np.argmin(x[:,np.newaxis] >= xvals,axis=-1)
    # This breaks for the extreme left (below left end)
    # Figure out how to avoid the if statement I'll need to add
    dp = cdf[ii+1]-cdf[ii]
    dx = xvals[ii+1]-xvals[ii]
    slope = dp/dx
    c_ = cdf[ii]+slope*(x-xvals[ii])
    return c_

def degaussianize(samp,c_v,cdf):
    """
    Given estimates for the marginal CDFs, invert the Gaussianization step for the given samples.

    This relies on the CDF estimation coded in invert_cdf_1d.  This is the basic place where improvements can be made.
    """
    sw = norm.cdf(samp)
    for i in range(samp.shape[0]):
        sw[i] = invert_cdf_1d(sw[i],c_v[i],cdf)
    return sw

# This breaks for out of range values.  Need to fix.  Also might break if I input a single sample
def invert_cdf_1d(uniform,xvals,cdf):
    eps = 0.1/cdf.size
    ii = np.argmin(uniform[:,np.newaxis] >= (cdf-eps),axis=-1)
    dp = cdf[ii]-cdf[ii-1]
    dx = xvals[ii]-xvals[ii-1]
    slope = dx/dp
    x = xvals[ii-1]+slope*(uniform-cdf[ii-1])
    return x

def invert_step(samp,rot,mean,cdf_v,cdf):
    un = degaussianize(samp,cdf_v,cdf)
    return rot.T@un + mean[:,np.newaxis]

def step(samp):
    sw,rot,mu = rotate_samples(samp)
    z,c,p = marginal_gaussianize(sw)
    return z,c,p,rot,mu

def rot_2d(theta):
    """
    Returns rotation matrix for a theta radian clockwise rotation
    """
    return np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta),np.cos(theta)]])

def sample_checkerboard(nsamp,board_size=3):
    s = np.random.uniform(size=(2,0))
    for i in range(board_size):
        off_x = i%2
        for j in range(board_size-off_x):
            s = np.hstack([s,np.random.uniform(size=(2,nsamp))+np.array([i,2*j+off_x])[:,np.newaxis]])
    return s

def gaussian_peaks(nsamp,npeaks):
    samp = np.random.normal(size=(2,0))
    for i in range(npeaks):
        for j in range(npeaks):
            samp = np.hstack( [samp, np.random.normal(size=(2,nsamp)) + np.array([10*i,10*j])[:,np.newaxis] ])
    return samp

def local_mapping(nSamp,nLat,f):
    samp = np.random.normal(size=(nLat,nSamp))
    return f(samp)

# Write this to see how a boundary maps to an interior
def ring_pdf(nsamp):
    return

def interpolate_transform(x,xv,yv):
    return

def interpolate_inverse_transform(x,xv,yv):
    return

def gaussianize_samples(samp,nStep):
    sw_ = []; s_=[]
    steps = []
    s = samp
    s_.append(np.copy(s))
    for i in range(nStep):
        s,c,p,r,m = step(s)
#        sw,r,m = rotate_samples(s)
#        sw_.sappend(np.copy(sw))
#        s,c,p = marginal_gaussianize(sw)
        s_.append(np.copy(s))
        steps.append( Transform_Step(np.copy(r),np.copy(m),np.copy(c),np.copy(p)) )
    return sw_, s_, steps
        
def make_scatter(s):
    f,a = plt.subplots()
    a.scatter(s[0],s[1],alpha=0.2,zorder=-10)
    a.set_title(r'Step %02i' %i)
    a.set_xlabel(r'$x_1^{(i)}$')
    a.set_ylabel(r'$x_2^{(i)}$')
    a.set_rasterization_zorder(-1)
    f.savefig('step-%02i.pdf'%i)
    return f,a
    
if __name__=="__main__":
    myGauss = Gaussianizer()
    
    samp = chi2.rvs(2,size=(2,1000))
    rot = rot_2d(0.37*np.pi)

#    print('Rotated Chi-Squared')
#    z = rot@samp
#    for i in range(5):
#        z,c,p,r,m = step(z)

    #s = sample_checkerboard(100)
    s = gaussian_peaks(1000,3)
    sw_ = []; s_ = []
    steps = []
    s_.append(np.copy(s))
    for i in range(500):
#        s,c,p,r,m = step(s)
        sw,r,m = rotate_samples(s)
        sw_.append(np.copy(sw))
        s,c,p = marginal_gaussianize(sw)
        s_.append(np.copy(s))
        step_cur = Transform_Step(np.copy(r),np.copy(m),np.copy(c),np.copy(p))
        steps.append(step_cur)
        
    s = gaussian_peaks(100,3)
    cdf2 = []; cdf_v2 = []; rot2 = []; s2_ = []; mu2_ = []; sw2_ = []
    s2_.append(np.copy(s))
    for i in range(500):
#        s,c,p,r,m = step(s)
        sw,r,m = rotate_samples(s)
        sw2_.append(np.copy(sw))
        s,c,p = marginal_gaussianize(sw)
        mu2_.append(m); cdf_v2.append(c); cdf2.append(p); rot2.append(r); s2_.append(np.copy(s))
 
    sNew = norm.rvs(size=(2,1000),random_state=42)
    s0 = np.copy(sNew)

    sNew_ = []
    for j in [499,399,299,199,99]:
        sNew = s0
        for i in range(j,-1,-1):
            r = steps[i].rot; mu = steps[i].mean; cdf_v = steps[i].cdf_v; cdf = steps[i].cdf
            sNew = invert_step(sNew,r,mu,cdf_v,cdf)
        sNew_.append(np.copy(sNew))
        
    sNew2 = np.copy(s0)
    for i in range(99,-1,-1):
        sNew2 = invert_step(sNew2,rot2[i],mu2_[i],cdf_v2[i],cdf2[i])
    # outlier analysis
    ii2 = np.where( (np.abs(sNew2[0]%10-5)<2) | (np.abs(sNew2[1]%10-5)<2) )

#    plt.clf()
#    for i in range(500):
#        for j in range(9):
#            plt.scatter(s_[i][0,j*1000:(j+1)*1000],s_[i][1,j*1000:(j+1)*1000],s=4.,alpha=0.3)
#        plt.title(r'Step %i'%i)
#        plt.savefig('step-colour-%04i.png'%i)
#        plt.clf()

    # Now plot derotated thing
#    plt.clf()
#    rtot = np.eye(2)
#    for j in range(9):
#        plt.scatter(s_[0][0,j*1000:(j+1)*1000],s_[0][1,j*1000:(j+1)*1000],s=4.,alpha=0.3)
#    plt.xlim(-5,25); plt.ylim(-5,25)
#    plt.title(r'Step 0')
#    plt.savefig('step-colour-no-rot-0000.png')
#    rtot = steps[0].rot.T@rtot
#    plt.clf()
    
#    for i in range(1,500):
#        scur = rtot@s_[i]
#        for j in range(9):
#            plt.scatter(scur[0,j*1000:(j+1)*1000],scur[1,j*1000:(j+1)*1000],s=4.,alpha=0.3)
#        plt.xlim(-4.5,4.5); plt.ylim(-4.5,4.5)
#        plt.title(r'Step %i'%i)
#        plt.savefig('step-colour-no-rot-%04i.png'%i)
#        rtot = steps[i].rot.T@rtot
#        plt.clf()
        
#    sw = s_[-1]; p = cdf[-1]
#    sw_ = np.empty(sw.shape)

#    sw = norm.cdf(sw)
#    for i in range(2):
#        ii = np.argmin(sw[i][:,np.newaxis]>=p,axis=-1)
#        sw_[i,:] = cdf_v[-2][i,ii]
        
#    plt.hist(s[0],51,alpha=0.2)
#    plt.hist(s[1],51,alpha=0.2)
#    ss = rot_2d(0.25*np.pi)@s
#    plt.hist(ss[0],51,alpha=0.2)

# Start filling in code for x+f_{NL}x^2 mapping
