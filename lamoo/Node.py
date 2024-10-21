from Classifier import Classifier
import json
import numpy as np
import math
import operator
from lamoo_utils import convert_dtype
import torch
import sampler_evaluator

class Node:
    """
    Represents a node in the Monte Carlo Tree Search.
    """
    OBJ_COUNTER = 0
    
    def __init__(self, args, parent=None, dims=0, reset_id=False, cp=10, height=0):
        """
        Initialize a Node.

        Args:
            args: Arguments passed to the Classifier
            parent (Node): Parent node
            dims (int): Dimensionality of the search space
            reset_id (bool): Whether to reset the object counter
            cp (float): Exploration constant for UCT
            height (int): Height of the node in the tree
        """
        self.dims = dims
        self.x_bar = float('inf')
        self.n = 0
        self.uct = 0
        self.classifier = Classifier(args, [], self.dims)
        self.bag = []
        self.is_svm_splittable = False
        self.cp = cp
        self.height = height

        self.parent = parent
        self.kids = []  # 0: good, 1: bad
        
        if reset_id:
            Node.OBJ_COUNTER = 0
        self.id = Node.OBJ_COUNTER        
        Node.OBJ_COUNTER += 1
    
    def update_kids(self, good_kid, bad_kid):
        """Add good and bad child nodes."""
        assert len(self.kids) == 0
        self.kids.extend([good_kid, bad_kid])

    def is_good_kid(self):
        """Check if this node is the 'good' child of its parent."""
        return self.parent is not None and self.parent.kids[0] == self

    def is_leaf(self):
        """Check if this node is a leaf node."""
        return len(self.kids) == 0

    def update_bag(self, samples, ref_point=None):
        """
        Update the bag of samples for this node.

        Args:
            samples (list): List of samples
            ref_point (torch.Tensor): Reference point for hypervolume calculation
        """
        assert len(samples) > 0 and ref_point is not None
        
        self.bag.clear()
        self.bag.extend(samples)
        self.classifier.update_samples(self.bag)
        self.is_svm_splittable = self.classifier.is_splittable_svm()
        self.x_bar = self.classifier.get_hypervolume(ref_point)
        self.n = len(self.bag[0])

    def clear_data(self):
        """Clear the bag of samples."""
        self.bag.clear()
    
    def get_name(self):
        """Get the name of the node."""
        return f"node{self.id}"

    def __str__(self):
        """String representation of the node."""
        name = f"{self.get_name():<7}"
        name += f"{'is good:' + str(self.is_good_kid()):<15}"
        name += f"{'is leaf:' + str(self.is_leaf()):<15}"
        name += f"val:{self.get_xbar():.4f}   "
        name += f"uct:{self.get_uct():.4f}   "
        name += f"sp/n:{len(self.bag[0])}/{self.n:<15}"

        samples = self.classifier.samples[0]
        if torch.cuda.is_available():
            samples = samples.cpu()
        upper_bound = np.around(np.max(samples.data.numpy(), axis=0), decimals=2)
        lower_bound = np.around(np.min(samples.data.numpy(), axis=0), decimals=2)
        boundary = ' '.join(f"{l}>{u}" for l, u in zip(lower_bound, upper_bound))
        name += f"bound:{boundary:<60}"

        parent = '----' if self.parent is None else self.parent.get_name()
        name += f"parent:{parent:<10}"
        
        kids = ' '.join(k.get_name() for k in self.kids)
        name += f"kids:{kids}"
        name += f"height:{self.height}"
        
        return name

    def get_uct(self):
        """Calculate the UCT value for this node."""
        if self.parent is None or self.n == 0:
            return float('inf')
        return self.x_bar + 2 * self.cp * math.sqrt(2 * math.log(self.parent.n) / self.n)
    
    def get_xbar(self):
        """Get the x_bar value (hypervolume) of this node."""
        return self.x_bar

    def get_n(self):
        """Get the number of samples in this node."""
        return self.n
        
    def train_and_split(self):
        """Train the classifier and split the data."""
        assert len(self.bag) >= 2
        self.classifier.update_samples(self.bag)
        good_kid_data, bad_kid_data = self.classifier.split_data()
        assert len(good_kid_data[0]) + len(bad_kid_data[0]) == len(self.bag[0])
        return good_kid_data, bad_kid_data
