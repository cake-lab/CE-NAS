from botorch.utils.sampling import draw_sobol_samples
from torch.quasirandom import SobolEngine
import torch
import random
import numpy as np
from lamoo_utils import latin_hypercube, from_unit_cube
from abc import ABC, abstractmethod
from tasks import nasbench201



class base_sampler_evaluator():

    def __init__(self, problem, nums_samples):
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


    '''
        #############################################
        # sobol sampling inside selected partition #
        #############################################
    '''

    def sobol_sample(self, n_pts=None):
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(dimension=self.dims, scramble=True, seed=seed)
        cands = sobol.draw(self.nums_samples).to(dtype=torch.float64).cpu().detach().numpy() \
            if not n_pts else sobol.draw(n_pts).to(dtype=torch.float64).cpu().detach().numpy()

        cands = (self.ub - self.lb) * cands + self.lb
        if not torch.cuda.is_available():
            new_x = torch.tensor(cands)
        else:
            new_x = torch.tensor(cands, device='cuda')
        new_obj = self.problem(new_x)

        cost = self.problem.evaluation_cost(new_x)

        return new_x, new_obj

    '''
        #############################################
        # latin hypercube sampling inside selected partition #
        #############################################
    '''


    def latin_hypercube_sample(self, n_pts=None):
        cands = latin_hypercube(self.nums_samples, self.dims) \
            if n_pts else latin_hypercube(n_pts, self.dims)
        cands = from_unit_cube(cands, self.lb, self.ub)
        if not torch.cuda.is_available():
            new_x = torch.tensor(cands)
        else:
            new_x = torch.tensor(cands, device='cuda')
        new_obj = self.problem(new_x)

        cost = self.problem.evaluation_cost(new_x)

        return new_x, new_obj

    '''
        #############################################
        # random sampling inside selected partition #
        #############################################
    '''

    def random_sample(self, n_pts=None):
        cands = np.random.uniform(self.lb, self.ub, size=(n_pts, self.dims)) \
            if n_pts else np.random.uniform(self.lb, self.ub, size=(self.nums_samples, self.dims))
        if not torch.cuda.is_available():
            new_x = torch.tensor(cands)
        else:
            new_x = torch.tensor(cands, device='cuda')
        new_obj = self.problem(new_x)

        cost = self.problem.evaluation_cost(new_x)

        return new_x, new_obj, cost

    '''
        #############################################
        # random sampling for nasbench201 #
        #############################################
    '''

    def random_sample_nasbench201(self, n_pts=None):
        if n_pts is None:
            cands_index = np.random.choice(self.nasbench201_space.shape[0], size=self.nums_samples, replace=False)
        else:
            cands_index = np.random.choice(self.nasbench201_space.shape[0], size=n_pts, replace=False)
            self.init_delete = cands_index
        cands = self.nasbench201_space[cands_index]


        self.nasbench201_space = np.delete(self.nasbench201_space, cands_index, axis=0)

        if not torch.cuda.is_available():
            new_x = torch.tensor(cands)
        else:
            new_x = torch.tensor(cands, device='cuda')
        new_obj = self.problem(new_x)

        cost = self.problem.evaluation_cost(new_x)

        return new_x, new_obj, cost


    '''
        #############################################
        # qNEHVI sampling inside selected partition #
        #############################################
    '''

    def qnehvi_sample(self, model, train_x, train_obj):

        from botorch.utils.transforms import unnormalize, normalize

        from botorch.optim.optimize import optimize_acqf, optimize_acqf_list

        from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, \
            qNoisyExpectedHypervolumeImprovement
        # from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.sampling.samplers import SobolQMCNormalSampler

        sampler = SobolQMCNormalSampler(num_samples=128, seed=0)
        tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

        NOISE_SE = torch.tensor([15.19, 0.63], **tkwargs)
        BATCH_SIZE = 5
        NUM_RESTARTS = 10
        RAW_SAMPLES = 1024

        standard_bounds = torch.zeros(2, self.problem.dim, **tkwargs)
        standard_bounds[1] = 1

        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.problem.ref_point.tolist(),  # use known reference point
            X_baseline=normalize(train_x, self.problem.bounds),
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        # observe new values
        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        new_obj = self.problem(new_x)
        # new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE

        cost = self.problem.evaluation_cost(new_x)
        return new_x, new_obj



class LamooSamplerEvaluator(base_sampler_evaluator):

    '''
    #############################################
    # sobol sampling inside selected partition #
    #############################################
    '''

    def sobol_sample(self, path, n_pts=10000, customized_nums=None):
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(dimension=self.dims, scramble=True, seed=seed)
        selected_cands = np.zeros((1, self.dims))
        if customized_nums:
            tmp = self.nums_samples
            self.nums_samples = customized_nums

        while len(selected_cands) <= self.nums_samples:
            cands = sobol.draw(n_pts).to(dtype=torch.float64).cpu().detach().numpy()
            cands = (self.ub - self.lb) * cands + self.lb
            for node in path:
                boundary = node[0].classifier.svm
                if len(cands) == 0:
                    break
                cands = cands[boundary.predict(cands) == node[1]]  # node[1] store the direction to go

            if len(cands) < 2:
                path = path[:-1]  # If the select region is too narrow for sampling.

            selected_cands = np.append(selected_cands, cands, axis=0)


        selected_cands = selected_cands[1:]
        final_cands_idx = np.random.choice(len(selected_cands), self.nums_samples)

        if not torch.cuda.is_available():
            new_x = torch.tensor(selected_cands[final_cands_idx])
        else:
            new_x = torch.tensor(selected_cands[final_cands_idx], device='cuda')
        new_obj = self.problem(new_x)

        if customized_nums:
            self.nums_samples = tmp

        cost = self.problem.evaluation_cost(new_x)

        return new_x, new_obj

    '''
        #############################################
        # latin hypercube sampling inside selected partition #
        #############################################
    '''

    def latin_hypercube_sample(self, path, n_pts=10000, customized_nums=None):
        selected_cands = np.zeros((1, self.dims))
        if customized_nums:
            tmp = self.nums_samples
            self.nums_samples = customized_nums

        while len(selected_cands) <= self.nums_samples:
            cands = latin_hypercube(n_pts, self.dims)
            cands = from_unit_cube(cands, self.lb, self.ub)
            for node in path:
                boundary = node[0].classifier.svm
                if len(cands) == 0:
                    break
                cands = cands[boundary.predict(cands) == node[1]]  # node[1] store the direction to go

            if len(cands) < 2:
                path = path[:-1]  # If the select region is too narrow for sampling.

            selected_cands = np.append(selected_cands, cands, axis=0)


        selected_cands = selected_cands[1:]
        final_cands_idx = np.random.choice(len(selected_cands), self.nums_samples)

        if not torch.cuda.is_available():
            new_x = torch.tensor(selected_cands[final_cands_idx])
        else:
            new_x = torch.tensor(selected_cands[final_cands_idx], device='cuda')
        new_obj = self.problem(new_x)

        if customized_nums:
            self.nums_samples = tmp

        cost = self.problem.evaluation_cost(new_x)

        return new_x, new_obj

    '''
        #############################################
        # random sampling inside selected partition #
        #############################################
    '''

    def random_sample(self, path, n_pts=10000, customized_nums=None):
        selected_cands = np.zeros((1, self.dims))

        if customized_nums:
            tmp = self.nums_samples
            self.nums_samples = customized_nums

        while len(selected_cands) <= self.nums_samples:
            cands = np.random.uniform(self.lb, self.ub, size=(n_pts, self.dims))
            # cands = from_unit_cube(cands, self.lb, self.ub)
            for node in path:
                boundary = node[0].classifier.svm
                if len(cands) == 0:
                    break
                cands = cands[boundary.predict(cands) == node[1]]  # node[1] store the direction to go

            if len(cands) < 2:
                path = path[:-1]  # If the select region is too narrow for sampling.

            selected_cands = np.append(selected_cands, cands, axis=0)


        selected_cands = selected_cands[1:]
        final_cands_idx = np.random.choice(len(selected_cands), self.nums_samples)

        if not torch.cuda.is_available():
            new_x = torch.tensor(selected_cands[final_cands_idx])
        else:
            new_x = torch.tensor(selected_cands[final_cands_idx], device='cuda')
        new_obj = self.problem(new_x)

        if customized_nums:
            self.nums_samples = tmp

        cost = self.problem.evaluation_cost(new_x)

        return new_x, new_obj

    '''
            #############################################
            # random sampling for nasbench201 #
            #############################################
        '''

    def random_sample_nasbench201(self, path, n_pts=None):
        selected_cands = np.zeros((1, self.dims))
        while len(selected_cands) <= self.nums_samples:
            cands_index = np.random.choice(self.nasbench201_space.shape[0], size=300, replace=False)
            cands = self.nasbench201_space[cands_index]
            for node in path:
                boundary = node[0].classifier.svm
                if len(cands) == 0:
                    break
                cands = cands[boundary.predict(cands) == node[1]]  # node[1] store the direction to go

            if len(cands) < 2:
                path = path[:-1]  # If the select region is too narrow for sampling.


            selected_cands = np.append(selected_cands, cands, axis=0)

        selected_cands = selected_cands[1:]
        final_cands_idx = np.random.choice(len(selected_cands), self.nums_samples)

        selected_samples = selected_cands[final_cands_idx]
        global_selected_id = []

        for i in range(len(self.nasbench201_space)):
            for j in range(len(selected_samples)):
                if self.nasbench201_space[i].tolist() == selected_samples[j].tolist():
                    global_selected_id.append(i)

        global_selected_id = np.array(global_selected_id)

        self.nasbench201_space = np.delete(self.nasbench201_space, global_selected_id, axis=0)

        if not torch.cuda.is_available():
            new_x = torch.tensor(selected_samples)
        else:
            new_x = torch.tensor(selected_samples, device='cuda')
        new_obj = self.problem(new_x)

        cost = self.problem.evaluation_cost(new_x)

        return new_x, new_obj, cost


    '''
        #############################################
        # qNEHVI sampling inside selected partition #
        #############################################
    '''

    def qnehvi_sample(self, model, path, train_x, train_obj):

        from botorch.utils.transforms import unnormalize, normalize

        from botorch_lamoo.optim.optimize import optimize_acqf, optimize_acqf_list

        from botorch_lamoo.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, \
            qNoisyExpectedHypervolumeImprovement
        # from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch_lamoo.sampling.samplers import SobolQMCNormalSampler

        sampler = SobolQMCNormalSampler(num_samples=128, seed=0)
        tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

        NOISE_SE = torch.tensor([15.19, 0.63], **tkwargs)
        BATCH_SIZE = 5
        NUM_RESTARTS = 10
        RAW_SAMPLES = 1024

        standard_bounds = torch.zeros(2, self.problem.dim, **tkwargs)
        standard_bounds[1] = 1

        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.problem.ref_point.tolist(),  # use known reference point
            X_baseline=normalize(train_x, self.problem.bounds),
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
        )
        # optimize
        kwargs = {}
        kwargs['lamoo_boundary'] = path
        kwargs['problem'] = self.problem
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
            **kwargs,
        )
        # observe new values
        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        new_obj = self.problem(new_x)
        # new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE

        cost = self.problem.evaluation_cost(new_x)
        return new_x, new_obj, cost












