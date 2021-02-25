#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def empirical_cdf(samp):
    ns = samp.shape[-1]
    prob = 0.5/ns + np.arange(ns)/(1.*ns)
    ginv = norm.ppf(prob)

    ii = np.argsort(samp,axis=-1)
    jj = np.argsort(ii,axis=-1)
    xvals = np.sort(samp,axis=-1)
    return ginv[jj], xvals, prob

# To Do: Deal with tails better
# To Do: Implement boundaries flag
def learn_cdf(samp,binSize=100,boundaries='none'):
    """
    Given the input samples, obtain a sampled version of the CDF.

    Input:
      samp       : The samples to estimate the CDF from.
      binSize    : Size of each bin of samples
      boundaries : A flag to determine the treatment of the boundaries.
          - 'minmax' : The CDF is assumed to be 0 at the minimum value and 1 at the maximum value
          - 'min' : CDF is assumed to be 0 at the minval
          - 'max' : CDF is assumed to be 1 at the maxval, and 0.5/num_samples 
          - 'none' : CDF is assumed to be 0.5/n_samples at minval and 1-0.5/n_samples at maxval
    """
    ns = samp.shape[-1]
    nBins = ns // binSize; pad = ns % binSize

    prob = 0.5/ns + np.arange(ns)/(1.*ns)  # Fix this up
    partInd = np.arange(nBins)*binSize + binSize//2 # Fix this up
    partInd = np.concatenate( ([0],partInd,[ns-1]) )
    xvals = np.partition(samp,partInd,axis=-1)[:,partInd]
    return xvals, prob[partInd]

def learn_mapping(samp,binSize=100,boundaries='none'):
    """
    Given the input samples, learn the nonlinear mapping y(x) that transforms them into samples drawn from a univariate Gaussian.
    """
    return

# To Do: Deal with extreme values (i.e. beyond samples) correctly
def invert_cdf_1d(prob,xvals,cdf):
    eps = 0.1/cdf.size
    ii = np.argmin(prob[:,np.newaxis] >= (cdf-eps),axis=-1)
    dp = cdf[ii]-cdf[ii-1]
    dx = xvals[ii]-xvals[ii-1]
    slope = dx/dp
    x = xvals[ii-1]+slope*(prob-cdf[ii-1])
    return x

# To Do: Deal with extreme values (i.e. beyond samples) correctly
def compute_cdf_1d(x,xvals,cdf):
    eps = 0.1/cdf.size
    ii = np.argmin(x[:,np.newaxis] >= xvals,axis=-1)
    dp = cdf[ii]-cdf[ii-1]
    dx = xvals[ii]-xvals[ii-1]
    slope = dp/dx
    prob = cdf[ii-1]+slope*(x-xvals[ii-1])
    return prob

if __name__=="__main__":
    pass
