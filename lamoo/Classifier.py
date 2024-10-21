import torch
import json
import numpy as np
from scipy.stats import norm
import copy as cp
from sklearn.svm import SVC
from sklearn.svm import SVR
from torch.quasirandom import SobolEngine
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from Hypervolume import get_pareto, compute_hypervolume_2d
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import torch
from lamoo_utils import convert_dtype
# import pygmo as pg


from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume


from sklearn.neural_network import MLPClassifier


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}



# the input will be samples!
class Classifier():
    """
    A classifier for multi-objective optimization using SVM and Gaussian Process.
    """

    def __init__(self, args, samples, dims):
        """
        Initialize the Classifier.

        Args:
            args: Arguments containing kernel, gamma, and degree for SVM
            samples (list): Initial samples
            dims (int): Dimensionality of the search space
        """
        assert dims >= 1
        assert isinstance(samples, list)
        self.dims = dims
        self.args = args

        # Initialize Gaussian Process Regressor
        noise = 0.1
        m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2)

        # Initialize SVM for learning boundary
        self.svm = SVC(kernel=args.kernel, gamma=args.gamma, degree=args.degree)

        self.samples = np.array([])
        self.update_samples(samples)

    def update_samples(self, latest_samples):
        """Update the samples used by the classifier."""
        assert isinstance(latest_samples, list)
        self.samples = latest_samples

    def get_hypervolume(self, ref_point):
        """
        Calculate the hypervolume of the current samples.

        Args:
            ref_point (torch.Tensor): Reference point for hypervolume calculation

        Returns:
            float: Calculated hypervolume
        """
        ref_point = ref_point.clone().detach()

        if torch.cuda.is_available():
            ref_point = ref_point.cuda()
            Y = self.samples[1].cuda()
        else:
            Y = self.samples[1]

        botorch_hv = Hypervolume(ref_point=ref_point)
        pareto_mask = is_non_dominated(Y)
        pareto_y = Y[pareto_mask]
        hv = botorch_hv.compute(pareto_y)

        return hv

    def learn_boundary(self, plabel):
        """
        Learn the boundary in the search space using SVM.

        Args:
            plabel (np.array): Labels for the samples
        """
        assert len(plabel) == len(self.samples[0])
        plabel = plabel.ravel()
        X = self.samples[0].cpu().data.numpy() if torch.cuda.is_available() else self.samples[0].data.numpy()
        self.svm.fit(X, plabel)

    def dominance_number(self, objectives):
        """
        Calculate the dominance number for each objective.

        Args:
            objectives (list): List of objective values

        Returns:
            np.array: Dominance count for each objective
        """
        n = len(objectives)
        dominance_count = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and all(objectives[i][k] <= objectives[j][k] for k in range(len(objectives[0]))):
                    dominance_count[j] += 1
        return np.array(dominance_count)

    def make_label(self, positive_ratio=0.5):
        """
        Create labels for the samples based on dominance.

        Args:
            positive_ratio (float): Ratio of samples to be labeled as positive

        Returns:
            np.array: Labels for the samples
        """
        assert len(self.samples) >= 2, "samples must > 0"
        fX = self.samples[1]
        obj_list = -fX.cpu().numpy()

        dc = self.dominance_number(obj_list)
        sorted_domi = np.argsort(dc)

        plabel = np.ones(len(sorted_domi))
        n_positive = int(len(sorted_domi) * positive_ratio)
        plabel[sorted_domi[:n_positive]] = 0

        return plabel

    def is_splittable_svm(self):
        """
        Check if the data is splittable using SVM.

        Returns:
            bool: True if splittable, False otherwise
        """
        plabel = self.make_label()

        if len(np.unique(plabel)) == 1:
            print('Not splittable:', plabel)
            return False

        self.learn_boundary(plabel)

        X = self.samples[0].cpu().data.numpy() if torch.cuda.is_available() else self.samples[0].data.numpy()
        svm_label = self.svm.predict(X)

        if len(np.unique(svm_label)) == 1:
            print('Not splittable by SVM:', svm_label)
            return False
        return True

    def split_data(self):
        """
        Split the data into good and bad samples based on SVM prediction.

        Returns:
            tuple: Two lists containing good and bad samples
        """
        if len(self.samples[0]) == 0:
            return [], []

        plabel = self.make_label()
        self.learn_boundary(plabel)
        assert len(plabel) == len(self.samples[0])

        good_samples = [torch.Tensor(), torch.Tensor()]
        bad_samples = [torch.Tensor(), torch.Tensor()]

        for idx, label in enumerate(plabel):
            sample = self.samples[0][idx].reshape(1, -1)
            obj = self.samples[1][idx].reshape(1, -1)
            if label == 0:
                good_samples[0] = torch.cat([good_samples[0], sample])
                good_samples[1] = torch.cat([good_samples[1], obj])
            else:
                bad_samples[0] = torch.cat([bad_samples[0], sample])
                bad_samples[1] = torch.cat([bad_samples[1], obj])

        assert len(good_samples[0]) + len(bad_samples[0]) == len(self.samples[0])
        return good_samples, bad_samples




