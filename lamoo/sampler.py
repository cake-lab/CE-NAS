from botorch.utils.sampling import draw_sobol_samples
from torch.quasirandom import SobolEngine
import torch
import random
import numpy as np
from lamoo_utils import latin_hypercube, from_unit_cube
from abc import ABC, abstractmethod
from tasks import nasbench201



class base_sampler():
    """Base class for sampling methods."""

    def __init__(self, problem, nums_samples):
        """
        Initialize the base sampler.

        Args:
            problem: The optimization problem.
            nums_samples (int): Number of samples to generate.
        """
        self.dims = problem.dim

        if torch.cuda.is_available():
            self.ub = problem.bounds[1].cpu().data.numpy()
            self.lb = problem.bounds[0].cpu().data.numpy()
        else:
            self.ub = problem.bounds[1].data.numpy()
            self.lb = problem.bounds[0].data.numpy()

        self.nums_samples = nums_samples
        self.problem = problem
        self.nasbench201_space = nasbench201.cands

    def sobol_sample(self, n_pts=None):
        """Generate samples using Sobol sequence."""
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(dimension=self.dims, scramble=True, seed=seed)
        n_pts = n_pts or self.nums_samples
        cands = sobol.draw(n_pts).to(dtype=torch.float64).cpu().numpy()
        cands = (self.ub - self.lb) * cands + self.lb
        return torch.tensor(cands, device='cuda' if torch.cuda.is_available() else 'cpu')

    def latin_hypercube_sample(self, n_pts=None):
        """Generate samples using Latin Hypercube sampling."""
        n_pts = n_pts or self.nums_samples
        cands = latin_hypercube(n_pts, self.dims)
        cands = from_unit_cube(cands, self.lb, self.ub)
        return torch.tensor(cands, device='cuda' if torch.cuda.is_available() else 'cpu')

    def random_sample(self, n_pts=None):
        """Generate random samples."""
        n_pts = n_pts or self.nums_samples
        cands = np.random.uniform(self.lb, self.ub, size=(n_pts, self.dims))
        return torch.tensor(cands, device='cuda' if torch.cuda.is_available() else 'cpu')

    def random_sample_nasbench201(self, n_pts=None):
        """Generate random samples for NASBench201."""
        n_pts = n_pts or self.nums_samples
        cands_index = np.random.choice(self.nasbench201_space.shape[0], size=n_pts, replace=False)
        cands = self.nasbench201_space[cands_index]
        self.nasbench201_space = np.delete(self.nasbench201_space, cands_index, axis=0)
        return torch.tensor(cands, device='cuda' if torch.cuda.is_available() else 'cpu')

    def qnehvi_sample(self, model, train_x, train_obj, n_pts=1):
        """Generate samples using qNEHVI method."""
        from botorch.utils.transforms import unnormalize, normalize
        from botorch.optim.optimize import optimize_acqf
        from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
        from botorch.sampling.samplers import SobolQMCNormalSampler

        sampler = SobolQMCNormalSampler(num_samples=128, seed=0)
        tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

        BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES = 5, 10, 1024
        standard_bounds = torch.zeros(2, self.problem.dim, **tkwargs)
        standard_bounds[1] = 1

        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.problem.ref_point.tolist(),
            X_baseline=normalize(train_x, self.problem.bounds),
            prune_baseline=True,
            sampler=sampler,
        )

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )

        return unnormalize(candidates.detach(), bounds=self.problem.bounds)



class LamooSampler(base_sampler):
    """Lamoo sampling methods."""

    def sobol_sample(self, path, n_pts=10000, customized_nums=None):
        """Generate Sobol samples within a specific region."""
        return self._sample_in_region(path, self._sobol_generate, n_pts, customized_nums)

    def latin_hypercube_sample(self, path, n_pts=10000, customized_nums=None):
        """Generate Latin Hypercube samples within a specific region."""
        return self._sample_in_region(path, self._lhs_generate, n_pts, customized_nums)

    def random_sample(self, path, n_pts=10000, customized_nums=None):
        """Generate random samples within a specific region."""
        return self._sample_in_region(path, self._random_generate, n_pts, customized_nums)

    def random_sample_nasbench201(self, path, n_pts=None):
        """Generate random samples for NASBench201 within a specific region."""
        selected_cands = np.zeros((1, self.dims))
        while len(selected_cands) <= self.nums_samples:
            cands_index = np.random.choice(self.nasbench201_space.shape[0], size=300, replace=False)
            cands = self.nasbench201_space[cands_index]
            cands = self._filter_cands(cands, path)
            selected_cands = np.append(selected_cands, cands, axis=0)

        selected_cands = selected_cands[1:]
        final_cands_idx = np.random.choice(len(selected_cands), self.nums_samples)
        selected_samples = selected_cands[final_cands_idx]

        global_selected_id = [i for i, sample in enumerate(self.nasbench201_space) 
                              if any(np.array_equal(sample, s) for s in selected_samples)]
        self.nasbench201_space = np.delete(self.nasbench201_space, global_selected_id, axis=0)

        return torch.tensor(selected_samples, device='cuda' if torch.cuda.is_available() else 'cpu')

    def qnehvi_sample(self, model, path, train_x, train_obj, n_pts=1):
        """Generate qNEHVI samples within a specific region."""
        from botorch.utils.transforms import unnormalize, normalize
        from botorch_lamoo.optim.optimize import optimize_acqf
        from botorch_lamoo.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
        from botorch_lamoo.sampling.samplers import SobolQMCNormalSampler

        sampler = SobolQMCNormalSampler(num_samples=128, seed=0)
        tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

        BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES = n_pts, 10, 1024
        standard_bounds = torch.zeros(2, self.problem.dim, **tkwargs)
        standard_bounds[1] = 1

        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.problem.ref_point.tolist(),
            X_baseline=normalize(train_x, self.problem.bounds),
            prune_baseline=True,
            sampler=sampler,
        )

        kwargs = {'lamoo_boundary': path, 'problem': self.problem}
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
            **kwargs,
        )

        return unnormalize(candidates.detach(), bounds=self.problem.bounds)

    def _sample_in_region(self, path, generate_func, n_pts, customized_nums):
        """Helper method to sample within a specific region."""
        selected_cands = np.zeros((1, self.dims))
        original_nums = self.nums_samples
        self.nums_samples = customized_nums or self.nums_samples

        while len(selected_cands) <= self.nums_samples:
            cands = generate_func(n_pts)
            cands = self._filter_cands(cands, path)
            selected_cands = np.append(selected_cands, cands, axis=0)

        selected_cands = selected_cands[1:]
        final_cands_idx = np.random.choice(len(selected_cands), self.nums_samples)
        self.nums_samples = original_nums

        return torch.tensor(selected_cands[final_cands_idx], 
                            device='cuda' if torch.cuda.is_available() else 'cpu')

    def _filter_cands(self, cands, path):
        """Filter candidates based on the path."""
        for node in path:
            boundary = node[0].classifier.svm
            if len(cands) == 0:
                break
            cands = cands[boundary.predict(cands) == node[1]]
        return cands

    def _sobol_generate(self, n_pts):
        """Generate Sobol samples."""
        sobol = SobolEngine(dimension=self.dims, scramble=True, seed=np.random.randint(int(1e6)))
        cands = sobol.draw(n_pts).to(dtype=torch.float64).cpu().numpy()
        return (self.ub - self.lb) * cands + self.lb

    def _lhs_generate(self, n_pts):
        """Generate Latin Hypercube samples."""
        cands = latin_hypercube(n_pts, self.dims)
        return from_unit_cube(cands, self.lb, self.ub)

    def _random_generate(self, n_pts):
        """Generate random samples."""
        return np.random.uniform(self.problem.bounds[0].cpu().numpy(), 
                                 self.problem.bounds[1].cpu().numpy(), 
                                 size=(n_pts, self.dims))
