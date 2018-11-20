from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import numpy as np


def compute_gmm(data_mfcc, feature, components, iterations):

    gauss_data = 0
    gmm_data = GaussianMixture(n_components=components, covariance_type='diag', max_iter=iterations).fit(data_mfcc)

    means_data = gmm_data.means_[:, feature]
    weights_data = gmm_data.weights_
    covs_data = gmm_data.covariances[:, feature]

    left = min(data_mfcc[:, feature])
    right = max(data_mfcc[:, feature])
    x = np.arange(left, right, 0.001)

    for i in range(len(means_data)):
        gauss_data = gauss_data + norm.pdf(x, means_data[i], covs_data[i]) * weights_data[i]

    return gauss_data
