import json
import numpy as np
import torch
import os

class Nasbench201:
    """
    A class representing the NASBench201 benchmark for CIFAR-10 dataset.
    """
    _max_hv = 4150.7236328125  # Max hypervolume for edgegpu_energy
    _max_energy = 48.780418395996094
    _min_energy = 2.059187173843384
    discrete = True

    def __init__(self):
        """
        Initialize the Nasbench201 object.
        """
        self.dim = 6
        self.num_objectives = 2
        self.ref_point = torch.tensor([-0.0, -50.0], device='cuda' if torch.cuda.is_available() else 'cpu')
        bounds = [(0.0, 0.99999)] * self.dim
        self.bounds = torch.tensor(bounds, dtype=torch.float).transpose(-1, -2)

        # Load the dataset
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, 'hw_nasbench.json')
        with open(file_path, 'r') as f:
            self.dataset = json.load(f)

    def __call__(self, x, normalized=True, acc=False, energy=False):
        """
        Evaluate the architecture(s).

        Args:
            x (torch.Tensor or list): The architecture(s) to evaluate.
            normalized (bool): Whether the input is normalized.
            acc (bool): If True, return only accuracy.
            energy (bool): If True, return only energy consumption.

        Returns:
            torch.Tensor: The evaluation results.
        """
        assert not (acc and energy), "Cannot return both accuracy and energy simultaneously"

        x = x.tolist() if isinstance(x, torch.Tensor) else x

        if normalized:
            x = self._normalize_architecture(x)

        res = []
        for arch in x:
            arch_str = str(arch)
            if acc:
                res.append(self.dataset[arch_str]['cifar10']['acc'])
            elif energy:
                res.append(self.dataset[arch_str]['cifar10']['edgegpu_energy'])
            else:
                res.append([
                    self.dataset[arch_str]['cifar10']['acc'],
                    -self.dataset[arch_str]['cifar10']['edgegpu_energy']
                ])

        return torch.tensor(res, dtype=torch.float)

    def evaluation_cost(self, x, method='vanilla'):
        """
        Calculate the evaluation cost for the given architecture(s).

        Args:
            x (torch.Tensor or list): The architecture(s) to evaluate.
            method (str): The evaluation method ('vanilla' or 'test').

        Returns:
            list: The evaluation costs.
        """
        x = x.tolist() if isinstance(x, torch.Tensor) else x
        x = self._normalize_architecture(x)

        cost_key = 'training_time' if method == 'vanilla' else 'test_time'
        return [self.dataset[str(arch)]['cifar10'][cost_key] for arch in x]

    def encode_to_nasbench201(self, sample):
        """
        Encode a sample to NASBench201 format.

        Args:
            sample (torch.Tensor): The sample to encode.

        Returns:
            str: The encoded sample.
        """
        sample = sample.cpu().data.numpy().tolist()
        normalized = [self._normalize_operation(op) for op in sample]
        denormalized = [self._denormalize_operation(op) for op in normalized]
        return str(denormalized)

    @staticmethod
    def _normalize_architecture(x):
        """
        Normalize the architecture representation.

        Args:
            x (list): The architecture representation.

        Returns:
            list: The normalized architecture representation.
        """
        return [[Nasbench201._normalize_operation(op) for op in arch] for arch in x]

    @staticmethod
    def _normalize_operation(op):
        """
        Normalize a single operation value.

        Args:
            op (float): The operation value.

        Returns:
            int: The normalized operation value.
        """
        if 0 <= op < 0.2:
            return 0
        elif 0.2 <= op < 0.4:
            return 1
        elif 0.4 <= op < 0.6:
            return 2
        elif 0.6 <= op < 0.8:
            return 3
        else:
            return 4

    @staticmethod
    def _denormalize_operation(op):
        """
        Denormalize a single operation value.

        Args:
            op (float): The operation value.

        Returns:
            float: The denormalized operation value.
        """
        return op * 0.2
