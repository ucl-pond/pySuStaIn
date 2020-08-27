import numpy as np
from sklearn import neighbors
from awkde import GaussianKDE # from scipy import stats

class KDEMM(object):
    """docstring for KDEMM"""
    def __init__(self, kernel='gaussian', bandwidth=None, n_iters=1500):
        super(KDEMM, self).__init__()
        self.n_iters = n_iters
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.controls_kde = None
        self.patholog_kde = None
        self.mixture = None
        self.alpha = 0.3 # sensitivity parameter: 0...1

    def fit(self, X, y, implement_fixed_controls=False):
        sorted_idx = X.argsort(axis=0).flatten()
        kde_values = X.copy()[sorted_idx].reshape(-1,1)
        kde_labels0 = y.copy()[sorted_idx]
        kde_labels = kde_labels0
        
        #print('Original labels')
        #print(kde_labels.astype(int))
        
        bin_counts = np.bincount(y).astype(float)
        mixture0 = sum(kde_labels==0)/len(kde_labels) # Prior of being a control
        mixture = mixture0
        old_ratios = np.zeros(kde_labels.shape)
        iter_count = 0
        if(self.bandwidth is None):
            #* 1. Rule of thumb
            self.bandwidth = hscott(X)
            # #* 2. Estimate full density to inform variable bandwidth: wide in tails, narrow in peaks
            # all_kde = neighbors.KernelDensity(kernel=self.kernel,
            #                                   bandwidth=self.bandwidth)
            # all_kde.fit(kde_values)
            # f = np.exp(all_kde.score_samples(kde_values))
            # #* 3. Local, a.k.a. variable, bandwidth given by eq. 3 of https://ieeexplore.ieee.org/abstract/document/7761150
            # g = stats.mstats.gmean(f)
            # alpha = 0.5 # sensitivity parameter: 0...1
            # lamb = np.power(f/g,-alpha)
        for i in range(self.n_iters):
            # #* Separate bandwidth for each mixture component, recalculated each loop
            # bw_controls = self.bandwidth # hscott(kde_values[kde_labels == 0])
            # bw_patholog = self.bandwidth # hscott(kde_values[kde_labels == 1])
            # controls_kde = neighbors.KernelDensity(kernel=self.kernel,
            #                                        bandwidth=bw_controls)
            # patholog_kde = neighbors.KernelDensity(kernel=self.kernel,
            #                                        bandwidth=bw_patholog)
            # controls_kde.fit(kde_values[kde_labels == 0])
            # # patholog_kde.fit(kde_values[kde_labels == 1])
            # controls_score = controls_kde.score_samples(kde_values)
            # patholog_score = patholog_kde.score_samples(kde_values)
            # #* Missing data - 50/50 likelihood
            # controls_score[np.isnan(controls_score)] = np.log(0.5)
            # patholog_score[np.isnan(patholog_score)] = np.log(0.5)

            #* Automatic variable/local bandwidth
            controls_kde = GaussianKDE(glob_bw="scott", alpha=self.alpha, diag_cov=False)
            patholog_kde = GaussianKDE(glob_bw="scott", alpha=self.alpha, diag_cov=False)
            controls_kde.fit(kde_values[kde_labels == 0])
            patholog_kde.fit(kde_values[kde_labels == 1])

            controls_score = controls_kde.predict(kde_values)
            patholog_score = patholog_kde.predict(kde_values)
            
            controls_score = controls_score*mixture
            patholog_score = patholog_score*(1-mixture)

            ratio = controls_score / (controls_score + patholog_score)
            
            #* Empirical distribution
            cdf_controls = np.cumsum(controls_score)/max(np.cumsum(controls_score))
            cdf_patholog = np.cumsum(patholog_score)/max(np.cumsum(patholog_score))
            cdf_diff = (cdf_patholog - cdf_controls)/(cdf_patholog + cdf_controls)
            disease_direction = -np.sign(np.mean(cdf_diff))
            if disease_direction > 0:
                cdf_direction = 1 + cdf_diff
            else:
                cdf_direction = -cdf_diff
            
            #* Missing data - need to test this (probably need to remove/impute missing data at the start)
            #ratio[np.isnan(ratio) & (kde_labels0==0)] = 1-cdf_direction[np.isnan(ratio) & (kde_labels0==0)]
            #ratio[np.isnan(ratio) & (kde_labels0==1)] = cdf_direction[np.isnan(ratio) & (kde_labels0==1)]
            #ratio[np.isnan(ratio)] = 0.5
            
            if(np.all(ratio == old_ratios)):
                break
            iter_count += 1
            old_ratios = ratio
            kde_labels = ratio < 0.5
            
            #* Labels to swap: 
            diff_y = np.hstack(([0], np.diff(kde_labels))) # !=0 where adjacent labels differ
            if (np.sum(diff_y != 0) >= 2 and 
                    np.unique(kde_labels).shape[0] == 2): 
                split_y = int(np.all(np.diff(np.where(kde_labels == 0)) == 1)) # 0 if all 0s are adjacent => always 1?
                sizes = [x.shape[0] for x in
                         np.split(diff_y, np.where(diff_y != 0)[0])] # lengths of each contiguous set of labels
                
                #* Identify which labels to swap using direction of abnormality: mean(KDE components)
                split_prior_smaller = (np.mean(kde_values[kde_labels ==
                                                          split_y])
                                       < np.mean(kde_values[kde_labels ==
                                                            (split_y+1) % 2]))
                if split_prior_smaller:
                    replace_idxs = np.arange(kde_values.shape[0])[-sizes[2]:] # greater values are swapped
                    #print('Labels swapped for greater values')
                else:
                    replace_idxs = np.arange(kde_values.shape[0])[:sizes[0]] # lesser values are swapped
                    #print('Labels swapped for lesser values')
                #print(kde_labels.astype(int))
                kde_labels[replace_idxs] = (split_y+1) % 2 # swaps labels
                #print(kde_labels.astype(int))
                
            #*** Prevent label swapping for "strong controls"
            #print('Maintaining labels for strong controls')
            #print(kde_labels.astype(int))
            fixed_controls_criteria_0 = (kde_labels0==0) # Controls 
            
            #* CDF criteria - do not delete: also used for disease direction
            en = 10
            cdf_threshold = (en-1)/(en+1) # cdf(p) = en*(1-cdf(c)), i.e., en-times more patients than remaining controls
            controls_tail = cdf_direction > (cdf_threshold * max(cdf_direction))
            #fixed_controls_criteria = fixed_controls_criteria_0 & (~controls_tail)
            
            #* PDF ratio criteria
            # ratio_threshold_strong_controls = 0.33 # P(control) / [P(control) + P(patient)]
            #fixed_controls_criteria = fixed_controls_criteria & (ratio > ratio_threshold_strong_controls) # "Strong controls" 
            
            #* Outlier criteria: quantiles
            q = 0.9 # x-tiles
            if disease_direction>0:
                q = q # upper 
                f = np.greater
                #print('Disease direction: positive')
            else:
                q = 1 - q # lower
                f = np.less
                #print('Disease direction: negative')
            controls_outliers = f(kde_values,np.quantile(kde_values,q))
            fixed_controls_criteria = fixed_controls_criteria_0.reshape(-1,1) & (~controls_outliers.reshape(-1,1))

            if implement_fixed_controls:
                kde_labels[np.where(fixed_controls_criteria)[0]] = 0
            
            #* Hack alert! Also force the patients to flip
            # controllike_pathologs_criteria = (~controls_outliers.reshape(-1,1))
            # kde_labels[np.where(controllike_pathologs_criteria)[0]] = 0

            bin_counts = np.bincount(kde_labels).astype(float)
            mixture = bin_counts[0] / bin_counts.sum()
            if(mixture < 0.10 or mixture > 0.90): # if(mixture < (0.90*mixture0) or mixture > 0.90):
                break
        self.controls_kde = controls_kde
        self.patholog_kde = patholog_kde
        self.mixture = mixture
        self.iter_ = iter_count
        return self

    def likelihood(self, X):
        controls_score, patholog_score = self.pdf(X)
        data_likelihood = controls_score+patholog_score
        data_likelihood = np.log(data_likelihood)
        return -1*np.sum(data_likelihood)

    def pdf(self, X, **kwargs):
        #* Old version: sklearn fixed-bw KDE
        # controls_score = self.controls_kde.score_samples(X)
        # controls_score = np.exp(controls_score)*self.mixture
        # patholog_score = self.patholog_kde.score_samples(X)
        # patholog_score = np.exp(patholog_score)*(1-self.mixture)
        #* Auto-Variable-bw KDE: awkde
        controls_score = self.controls_kde.predict(X)*self.mixture
        patholog_score = self.patholog_kde.predict(X)*(1-self.mixture)
        return controls_score, patholog_score

    def probability(self, X):
        controls_score, patholog_score = self.pdf(X.reshape(-1, 1))
        #* Handle missing data
        controls_score[np.isnan(controls_score)] = 0.5
        patholog_score[np.isnan(patholog_score)] = 0.5
        controls_score[controls_score==0] = 0.5
        patholog_score[patholog_score==0] = 0.5
        c = controls_score / (controls_score+patholog_score)
        #c[(controls_score+patholog_score)==0] = 0.5
        return c

    def BIC(self, X):
        controls_score, patholog_score = self.pdf(X.reshape(-1, 1))
        likelihood = controls_score + patholog_score
        likelihood = -1*np.log(likelihood).sum()
        return 2*likelihood+2*np.log(X.shape[0])


def hscott(x, weights=None):

    IQR = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    A = min(np.nanstd(x, ddof=1), IQR / 1.349)

    if weights is None:
        weights = np.ones(len(x))
    n = float(np.nansum(weights))
    #n = n/sum(~np.isnan(x))

    return 1.059 * A * n ** (-0.2)
