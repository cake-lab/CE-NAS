from lamoo.tasks.nasbench201 import cifar10 as nasbench201_cifar10
from sampler_evaluator import LamooSamplerEvaluator, base_sampler_evaluator
from sampler import LamooSampler, base_sampler
from torch.multiprocessing import Manager
import torch.multiprocessing as mp
import torch

init_data = {}

for i in range(10):
    init_data[i] = {}
    func = nasbench201_cifar10.Nasbench201()
    lamoo_sampler = LamooSampler(problem=func, nums_samples=1)
    base_sampler_evaluator_local = base_sampler_evaluator(problem=func, nums_samples=5)
    init_samples, init_obj, cost = base_sampler_evaluator_local.random_sample_nasbench201(n_pts=10)
    init_data[i]['samples'] = init_samples
    init_data[i]['obj'] = init_obj
    init_data[i]['cost'] = cost
    init_data[i]['sample_index'] = base_sampler_evaluator_local.init_delete


torch.save(init_data, './init_data.pth')


