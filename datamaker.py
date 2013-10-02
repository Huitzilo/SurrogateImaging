'''
Created on Mar 28, 2012

@author: jan
'''
import random
import numpy as np
from scipy.stats import norm, expon, gamma
from scipy.spatial.distance import squareform


def gaussian_influence(mu, width):
    ''' creates a 2D-function of gaussian influence sphere'''
    return lambda x, y: np.exp(-width * ((x - mu[0]) ** 2 + (y - mu[1]) ** 2))

def correlated_samples(cov, num_sampl, marginal_dist):
    ''' creates correlated samples with a gaussian copula '''

    # Create Gaussian Copula
    dependence = np.random.multivariate_normal([0] * cov.shape[0], cov, num_sampl)
    dependence_dist = norm()
    uniform_dependence = dependence_dist.cdf(dependence)

    #Transform marginals 
    dependend_samples = marginal_dist.ppf(uniform_dependence)
    return dependend_samples


def group_covmtx(rho_intra, rho_inter, num_groups, num_objects):
    ''' create a covarince matrix with groups
    
    in each group are num_objects with a covariance of rho_intra. 
    objects between groups have a covariance of rho_intra 
    '''

    intra_mtx_size = (num_objects ** 2 - num_objects) / 2
    intra_cov = 1 - squareform([1 - rho_intra] * intra_mtx_size)

    cov = rho_inter * np.ones((num_groups * num_objects, num_groups * num_objects))
    for group_num in range(num_groups):
        print group_num * num_objects, (group_num + 1) * num_objects
        cov[group_num * num_objects:(group_num + 1) * num_objects,
            group_num * num_objects:(group_num + 1) * num_objects] = intra_cov
    return cov

def adjusted_gamma(mean, var):
    ''' create a gamma distribution with defined mean and variance '''

    scale = var / mean
    shape = mean / scale
    if shape > 1:
        print '!!! Warning !!! - shape parameter: ', str(shape)
    return gamma(shape, scale=scale)


class Dataset():

    def __init__(self, param):
        ''' creates sources and their mixed observations '''

        # create spatial sources
        num_grid = param.get('gridpoints', 9)
        pixel = np.indices(param['shape'])
        p_dist = param['shape'][0] / num_grid
        self.points = np.indices((num_grid, num_grid)) * p_dist + p_dist
        self.points = zip(self.points[0].flatten(), self.points[1].flatten())
        random.shuffle(self.points)
        components = [gaussian_influence(mu, param['width'])(pixel[0], pixel[1])
                  for mu in self.points[:param['latents']]]
        self.spt_sources = np.array([i.flatten() for i in components])

        # generate activation timcourses
        covgroups = param.get('covgroups', 4)
        self.cov = group_covmtx(param['cov'], 0.1, covgroups, param['latents'] / covgroups)
        marginal_dist = adjusted_gamma(param['mean'], param['var'])
        self.activ_pre = correlated_samples(self.cov, param['no_samples'], marginal_dist).T
        self.activ_pre[np.isnan(self.activ_pre)] = 0
        # fold with single stim timecourse
        if param['act_time']:
            self.activation = np.vstack([np.outer(i, param['act_time']).flatten()
                                    for i in self.activ_pre]).T
        self.observed_raw = np.dot(self.activation, self.spt_sources)

        # add noise
        noise = param['noisevar'] * np.random.randn(*self.observed_raw.shape)
        self.observed = self.observed_raw.copy() + noise
