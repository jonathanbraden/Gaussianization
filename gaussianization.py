#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm  # For inverse of Gaussian CDF, norm.ppf
from scipy.stats import chi2  # For testing purposes, chi2.rvs for samples, chi2.cdf for cdf
from scipy.stats import special_ortho_group  # For random rotations

from distributions import *

class Gaussianizer(object):
    def __init__(self,random_rot=True,rot_seed=42):
        """
        Initialise Gaussianation object with no steps.

        Input:
          random_rot   : Option to use random rotation matrices
          rot_seed     : Seed for random rotations
        """
        self.nSteps = 0
        self.steps = []
        self.trained = False
        self.rotation_seed = rot_seed  # Not implemented yet
        self.random_rot = True        # Not implemented yet
        self.init_samples = None
        self.gaussian_samples = None
        self.dim = None
        return

    def gaussianize(self,samp,ns,condition=None):
        """
        Given the set of input samples, Gaussianize them

        Input:
          samp      : Samples from the distribution to be Gaussianized
          ns        : Number of steps to take
          condition : (optional) Function defining stopping condition (not implemented)
        """
        self.dim = samp.shape[0]
        self.init_samples = np.copy(samp)
        for i in range(ns):
            samp = self.train_step(samp)
        self.trained = True
        self.gaussian_samples = samp
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

    # Need to debug this more (and test with output of inverse_transform, etc.  Decide which is fastest)
    def create_samples(self,nSamp):
        """
        Return a new realisation of samples from the trained transform.

        Input:
          nSamp : Number of samples to generate
        """
        g = np.random.normal(size=(self.dim,nSamp))
        for i in range(self.nSteps-1,-1,-1):  # Check indexing here
            g = invert_step(g,self.steps[i].rot,self.steps[i].mean,self.steps[i].cdf_v,self.steps[i].cdf)
        return g

    # Need to write this code in the transform object
    def forward_transform(self,y):
        x = y
        for i in range(self.nSteps):
            x = self.steps[i].forward(x)
        return x

    def inverse_transform(self,x):
        y = x
        for i in range(self.nSteps-1,-1,-1):
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

    def forward(self,x):
        x = x - self.mean[:,np.newaxis]
        x = self.rot@x
        c_ = np.empty(x.shape)
        for i in range(x.shape[0]):
            c_[i,:] = compute_cdf_1d(x[i],self.cdf_v[i],self.cdf)
        return norm.ppf( c_ )

    def backward(self,y):
        y = degaussianize(y,self.cdf_v,self.cdf)
        y = self.rot.T@y + self.mean[:,np.newaxis]
        return y

    def nonlinear_transform_sampled(self):
        """
        Return the (sampled) transformation y = Phi_{G}^{-1}(C(x)) = y(x) associated with the inferred CDF
        """
        y = norm.ppf(self.cdf)
        return self.cdf_v, y

    def gaussian_transform(self,x):
        """
        """
        return

    def inverse_gaussian_transform(self,y):
        """
        """
        return
    
# Here's an idea
# Do a merge type operation.  Rotate pairs
# Then rotate in the rotated pairs, etc.
class BlockRotation(object):
    def __init__(self,dim,blockSize,nBlocks,create=True):
        self.blockSize = blockSize
        self.nBlocks = nBlocks
        if create:
            shuffle = np.random.permutation( np.concatenate((np.arange(dim),np.random.randint(dim,size=2))) ).reshape((-1,blockSize)) # Minor bug that I can repeat indices here
            rot = special_ortho_group.rvs(blockSize,size=nBlocks)
        else:
            self.shuffle = None   # Array of size (nBlocks,blockSize) storing permuted indices
            self.rot = None       # Array of size (nBlocks,blockSize,blockSize)
        return

    def apply_rotation(self,samp):
        for i,dc in enumerate(self.shuffle):
            samp[dc,:] = rot[i]@samp[dc,:]
        return samp

    def make_random_rotation(self):
        return

    def make_cov_rotation(self):
        return
    
def random_block_rotation(samp,blockSize=2):
    """
    Randomly rotate samples using subblocks of size blockSize.
    """
    d = samp.shape[0]
    nBlock = d // blockSize; pad = d%blockSize
    shuffle = np.random.permutation(np.arange(d))
    rot = special_ortho_group.rvs(blockSize,size=nBlock)
    
    #shuffle = np.random.permutation(np.concatenate( (np.arange(d),np.random.randint(d,size=pad)) )).reshape((-1,blockSize))  # Check this is correct
    for i,dc in enumerate(shuffle):
        samp[dc,:] = rot[i]@samp[dc,:]

    # Alternative version using covariance
#    for dc in shuffle:
#        cov = np.cov(samp[dc,:],rowvar=True)  # check this
#        u,sig,rot = np.linalg,svd(cov,hermitian=True)
#        samp[dc,:] = rot@samp[dc,:]
    return

# Make sure this is the correct thing to do
# Currently doing random rotations.  Need to fix
def rotate_samples(samp,use_cov=False):
    """
    Rotate the set of input samples using a random rotation drawn from SO(d) where d is the dimension of the samples or using an SVD.

    For very large sample spaces, these large dimensional rotations will be rather inefficient both in terms of matrix multiplication times and storage requirements.  More optimal rotation choices could be applied in this case.

    Input:
       samp : The samples.  Exist as a (d)x(n_sample) numpy array
       use_cov (optional) : Boolean choosing whether to rotate using a random rotation or SVD decomposition
    """
    mu = np.mean(samp,axis=-1)
    sw = samp - mu[:,np.newaxis]
    if use_cov:
        cov = np.cov(sw,rowvar=True)
        u,sig,v = np.linalg.svd(cov,hermitian=True)
    else:
        v = special_ortho_group.rvs(samp.shape[0])
    return v@sw, v, mu

# Currently using the ugly solution of sorting the indexed keys
def marginal_gaussianize(samp):
    """
    Marginally Gaussianize each the samples along the final axis.

    Input:
      samp : A numpy array of shape (dim,n_samples) containing samples from some distribution
    
    Output:
      g_[jj] : The mapped value y corresponding to the input samples.
      c_     : The inverse of the CDF evaluated at uniformly spaced percentiles
      p_     : The values of the CDF at values c_

    The ordered pairs (c_,p_) represents the CDF (p_) as a function of the variable (c_)
    """
    n = samp.shape[-1]
    p_ = 0.5/n+np.arange(n)/n
    g_ = norm.ppf(p_)

    ii = np.argsort(samp,axis=-1)
    jj = np.argsort(ii,axis=-1)  # Can I turn this into a single line
    c_ = np.sort(samp,axis=-1)
    return g_[jj], c_, p_

### Write this and the next function
def learn_gaussianize(samp,binSize=128):
    ns = samp.shape[-1]

    partitions = np.arange(start=binSize//2,stop=ns,step=binSize)
    pVal = partitions / np.float64(partitions.size)  # Check this
    gaussVal = norm.ppf(pVal)

    ii = np.argsort(samp,axis=-1)
    jj = np.argsort(ii,axis=-1)
    #jj[:,partitions] # This about this
    return


#### End of to write section

def compute_cdf_1d(x,xvals,cdf):
    ii = np.argmin(x[:,np.newaxis] >= xvals,axis=-1)
    # This breaks for the extreme left (below left end)
    # Figure out how to avoid the if statement I'll need to add
    dp = cdf[ii]-cdf[ii-1]
    dx = xvals[ii]-xvals[ii-1]
    slope = dp/dx
    c_ = cdf[ii-1]+slope*(x-xvals[ii-1])
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

# Finish writing this
def forward_step(samp,rot,mean,cdf_v,cdf):
    samp = rot@(samp-mean[:,np.newaxis])
    # Now do the gaussianization
    return

def step(samp):
    sw,rot,mu = rotate_samples(samp)
    z,c,p = marginal_gaussianize(sw)
    return z,c,p,rot,mu

def gaussianize_samples_use_step(samp,nStep):
    sw_ = []; s_=[]
    steps = []
    s = samp
    s_.append(np.copy(s))
    for i in range(nStep):
        s,c,p,r,m = step(s)
        s_.append(np.copy(s))
        steps.append( Transform_Step(np.copy(r),np.copy(m),np.copy(c),np.copy(p)) )
    return sw_, s_, steps
        
def gaussianize_samples_no_step(s,nIt=500):
    s_ = []; sw_ = []; steps = []
    s_.append(np.copy(s))
    for i in range(nIt):
        sw,r,m = rotate_samples(s)
        sw_.append(np.copy(sw))
        s,c,p = marginal_gaussianize(sw)
        s_.append(np.copy(s))
        steps.append( Transform_Step(np.copy(r),np.copy(m),np.copy(c),np.copy(p)) )
    return s_, sw_, steps

def make_scatter(s,ind=(0,1),a=None,ti=0):
    """
    Make a scatter plot for the data s using pair of given indices.

    Input:
      s   - The data samples of shape (dim)x(n samples)
      ind - The pair of indices to sample from
      a   - axis to plot (if present), otherwise None
      ti  - Step index (defaults to 0)
    """
    if a==None:
        f,a = plt.subplots()
    else:
        f = a.get_figure()
    a.scatter(s[ind(0)],s[ind(1)],alpha=0.2,zorder=-10)
    a.set_xlabel(r'$x_1^{(i)}$')
    a.set_ylabel(r'$x_2^{(i)}$')
    a.set_rasterization_zorder(-1)

    a.set_title(r'Step %02i' % ti)
    f.savefig('step-%02i.pdf' % ti)
    return f,a


# To Do here: 1) Make histograms (can fix bin sizes)
#             2) Make CDFs
#             3) Do tests like KS, etc.
def test_marginal_gaussianity(samp,nAxes=10,rand_axes=True):
    """
    Test the gaussianity of the marginal distributions along a collection of axes.
    This is done by rotating the samples and doing marginalisations along the coordinate axes

    Input:
      samp   : The samples to test
      nAxes  : Number of rotations to apply
      rand_axes (Boolean) : If true, randomly sample the rotation axes
    """
    for i in range(nAxes):
        rot = special_ortho_group.cvs(samp.shape[0])
        sR = rot@samp
        # Now insert code to do various tests.
        # e.g. plt.hist(sR[0,:]); plt.hist(sR[1,:]), ...
    return

def test_plot(step,gauss,i,j):
    plt.plot(step[i].cdf_v[j],step[i].cdf)
    plt.plot(gauss.steps[i].cdf_v[j],gauss.steps[i].cdf)
    return

if __name__=="__main__":
    myGauss = Gaussianizer()
    
    s = chi2.rvs(2,size=(3,1000))
    rot = special_ortho_group.rvs(s.shape[0])
    s = rot@s
    sNew = norm.rvs(size=(3,1000),random_state=42)
    s_cp = np.copy(s)
    
    #s = sample_checkerboard(100)
    #s = gaussian_peaks(1000,3)

    sw_ = []; s_ = []
    steps = []
    s_.append(np.copy(s))
    np.random.seed(42)
    for i in range(500):
#        s,c,p,r,m = step(s)
        sw,r,m = rotate_samples(s)
        sw_.append(np.copy(sw))
        s,c,p = marginal_gaussianize(sw)
        s_.append(np.copy(s))
        step_cur = Transform_Step(np.copy(r),np.copy(m),np.copy(c),np.copy(p))
        steps.append(step_cur)

    # Testing for the object
    s_cp = gaussian_peaks(1000,3)
    np.random.seed(42)
    myGauss.gaussianize(s_cp,500)
#    for i in range(500):
#        s_cp = myGauss.train_step(s_cp)  # Figure out how to remove this ugly part
        
#    s0 = np.copy(sNew)
#    sNew_ = []
#    for j in [499,399,299,199,99]:
#        sNew = s0
#        for i in range(j,-1,-1):
#            r = steps[i].rot; mu = steps[i].mean; cdf_v = steps[i].cdf_v; cdf = steps[i].cdf
#            sNew = invert_step(sNew,r,mu,cdf_v,cdf)
#        sNew_.append(np.copy(sNew))
        
#    sNew2 = np.copy(s0)
#    for i in range(99,-1,-1):
#        sNew2 = invert_step(sNew2,rot2[i],mu2_[i],cdf_v2[i],cdf2[i])
    # outlier analysis
#    ii2 = np.where( (np.abs(sNew2[0]%10-5)<2) | (np.abs(sNew2[1]%10-5)<2) )

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
