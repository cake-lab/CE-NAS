import numpy as np
import torch
import time
from botorch_lamoo.models.gp_regression import SingleTaskGP
from botorch_lamoo.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch_lamoo import fit_gpytorch_model
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
import torch.multiprocessing as mp

from lamoo.Node import Node
from lamoo.sampler_evaluator import LamooSamplerEvaluator, base_sampler_evaluator
from lamoo.sampler import LamooSampler, base_sampler

class MCTS:
    """
    Monte Carlo Tree Search for finding the most promising region in the search space.
    """

    def __init__(self, lb, ub, dims, ninits, oneshot, vanilla, args, run=0, func='vanilla'):
        """
        Initialize the MCTS algorithm.
        """
        self.dims = dims
        self.samples = mp.Manager().list()
        self.random_samples = [torch.Tensor(), torch.Tensor()]
        self.nodes = []
        self.Cp = args.cp
        self.lb, self.ub = lb, ub
        self.ninits = ninits

        self.LEAF_SAMPLE_SIZE = 10
        self.MAX_TREE_HEIGHT = 3
        self.path = None
        self.gpu_nums = 10
        self.cur_cost = 0

        self.run = run
        self.args = args
        self.datetime = [0]

        self.func = vanilla if func == 'vanilla' else oneshot
        self.timestamp = 0

        self.lamoo_sampler = LamooSampler(problem=self.func, nums_samples=1)
        self.base_sampler = base_sampler(problem=self.func, nums_samples=1)
        self.lamoo_sampler_evaluator = LamooSamplerEvaluator(problem=self.func, nums_samples=self.args.sample_num)
        self.base_sampler_evaluator = base_sampler_evaluator(problem=self.func, nums_samples=self.args.sample_num)

        self.gpu_queue = {}
        self.worker_budget = {}

        self.ROOT = Node(args=self.args, parent=None, dims=self.dims, reset_id=True, cp=self.Cp)
        self.nodes.append(self.ROOT)

        self.init_train()

    def init_train(self):
        """
        Get the initialization data.
        :return:
        """

        self.gpu_queue['oneshot'] = []
        self.gpu_queue['vanilla'] = []
        self.gpu_queue['halfwork'] = []

        for i in range(self.gpu_nums):
            self.worker_budget[i] = 3600

        init_samples, init_obj, cost = \
            self.base_sampler_evaluator.random_sample(n_pts=self.ninits)

        self.cur_cost = cost

        if torch.cuda.is_available():
            init_samples, init_obj = torch.tensor(init_samples, device='cuda'), torch.tensor(init_obj, device='cuda')


        self.samples.append(init_samples.cpu())
        self.samples.append(init_obj.cpu())

        self.random_samples.append(init_samples)
        self.random_samples.append(init_obj)


        print("=" * 10 + 'collect ' + str(len(self.samples[0])) + ' points for initializing MCTS' + "=" * 10)
        print("lb:", self.lb)
        print("ub:", self.ub)
        print("Cp:", self.Cp)
        print("inits:", self.ninits)
        print("dims:", self.dims)
        print("=" * 58)

    def init_surrogate_model(self):
        """
        Initialize the surrogate model for Bayesian optimization.
        """
        train_x, train_obj = self.samples[0], self.samples[1]
        self.surrogate_model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
        self.mll = ExactMarginalLogLikelihood(self.surrogate_model.likelihood, self.surrogate_model)
        fit_gpytorch_model(self.mll)

    def populate_training_data(self):
        """
        Rebuild the tree when running a new search iteration.
        """
        self.ROOT.obj_counter = 0
        for node in self.nodes:
            node.clear_data()
        self.nodes = [Node(args=self.args, parent=None, dims=self.dims, reset_id=True, cp=self.Cp)]
        self.ROOT = self.nodes[0]
        self.ROOT.update_bag(self.samples, self.func.ref_point)

    def get_leaf_status(self):
        """
        Return the status of leaf nodes (splittable or not).
        """
        return np.array([
            node.is_leaf() and len(node.bag[0]) > self.LEAF_SAMPLE_SIZE
            and node.is_svm_splittable and node.height + 1 <= self.MAX_TREE_HEIGHT
            for node in self.nodes
        ])

    def get_split_idx(self):
        """
        Return the indices of splittable nodes.
        """
        return np.argwhere(self.get_leaf_status()).reshape(-1)

    def is_splittable(self):
        """
        Check if there are any splittable leaves.
        """
        return np.any(self.get_leaf_status())

    def dynamic_treeify(self):
        """
        Dynamically grow the tree by splitting nodes.
        """
        self.populate_training_data()
        print(f"Total nodes: {len(self.nodes)}")
        assert len(self.ROOT.bag[0]) == len(self.samples[0])
        assert len(self.nodes) == 1
        
        while self.is_splittable():
            to_split = self.get_split_idx()
            print(f"==>to split: {to_split}, total: {len(self.nodes)}")
            for nidx in to_split:
                parent = self.nodes[nidx]
                good_kid_data, bad_kid_data = parent.train_and_split()
                
                good_kid = Node(args=self.args, parent=parent, dims=self.dims, reset_id=False,
                                cp=self.Cp, height=parent.height+1)
                bad_kid = Node(args=self.args, parent=parent, dims=self.dims, reset_id=False,
                               cp=self.Cp, height=parent.height+1)

                good_kid.update_bag(good_kid_data, self.func.ref_point)
                bad_kid.update_bag(bad_kid_data, self.func.ref_point)
                parent.update_kids(good_kid=good_kid, bad_kid=bad_kid)
                self.nodes.extend([good_kid, bad_kid])

        self.print_tree()

    def print_tree(self):
        """
        Print information about all nodes in the tree.
        """
        print('-'*100)
        for node in self.nodes:
            print(node)
        print('-'*100)

    def select(self):
        """
        Select the best node based on rule of MCTS.
        """
        curt_node = self.ROOT
        path = []
        while not curt_node.is_leaf():
            UCT = [kid.get_uct() for kid in curt_node.kids]
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append((curt_node, choice))
            curt_node = curt_node.kids[choice]
            print(f"=> {curt_node.get_name()} {choice}")
        return curt_node, path

    def select_bad(self):
        """
        Select the worst node based on rule of MCTS.
        """
        curt_node = self.ROOT
        path = []
        while not curt_node.is_leaf():
            UCT = [kid.get_uct() for kid in curt_node.kids]
            choice = np.random.choice(np.argwhere(UCT == np.amin(UCT)).reshape(-1), 1)[0]
            path.append((curt_node, choice))
            curt_node = curt_node.kids[choice]
            print(f"=> {curt_node.get_name()} {choice}")
        return curt_node, path

    def leaf_select(self):
        """
        Leaf based selection strategy. (Select the best leaf first, and get the path from root to the selected leaf.)
        """
        leaves = [node for node in self.nodes if node.is_leaf()]
        paths = []
        for leaf in leaves:
            path = []
            node = leaf
            while node is not self.ROOT:
                idx = 0 if node.is_good_kid() else 1
                path.insert(0, (node.parent, idx))
                node = node.parent
            paths.append(path)

        if len(paths) > 1:
            UCT = [path[-1][0].kids[path[-1][1]].get_uct() for path in paths]
            leaf_idx = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path = paths[leaf_idx]
            curt_node = path[-1][0].kids[path[-1][1]]
        else:
            path = paths[0]
            curt_node = self.ROOT

        print("path is ", [(node[0].get_name(), node[1]) for node in path])
        return curt_node, path

    def greedy_select(self):
        """
        Select the node with the best hv value. (always select the good direction)
        """
        curt_node = self.ROOT
        path = []

        while not curt_node.is_leaf():
            UCT = []
            k_good = {'good': -1, 'bad': -1}

            for i, kid in enumerate(curt_node.kids):
                if kid.is_good_kid():
                    k_good['good'] = kid.get_xbar()
                else:
                    k_good['bad'] = kid.get_xbar()

                if k_good['good'] == k_good['bad'] and kid.is_good_kid():
                    UCT.append(kid.get_xbar() + 0.01)
                else:
                    UCT.append(kid.get_xbar())

            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append((curt_node, choice))
            curt_node = curt_node.kids[choice]

            print(f"=> {curt_node.get_name()} {choice}", end=' ')

        return curt_node, path

    def backpropagate(self, leaf, acc):
        """
        Backpropagate the value through the tree.
        """
        node = leaf
        while node is not None:
            node.x_bar = (node.x_bar * node.n + acc) / (node.n + 1)
            node.n += 1
            node = node.parent

    def search(self):
        """
        Perform the MCTS search to find the best region and generate new samples.
        """
        botorch_hv = Hypervolume(ref_point=self.func.ref_point.clone().detach())
        print(f'MAX HV IS: {self.func._max_hv}')
        self.time_list = []
        self.hv_track = []
        self.random_hv_track = []

        for i in range(self.args.iter):
            t0 = time.time()
            self.dynamic_treeify()
            
            if i == 0:
                self.random_hv_track.append(self.nodes[0].get_xbar())
            self.hv_track.append(self.nodes[0].get_xbar())

            print(f'HV in each iter: {self.hv_track}')
            print(f'Random HV in each iter: {self.random_hv_track}')
            print(f'Search time in each iter: {self.time_list}')

            # Select leaf node
            if self.args.node_select == 'mcts':
                leaf, path = self.select()
                bad_leaf, bad_path = self.select_bad()
            elif self.args.node_select == 'leaf':
                leaf, path = self.leaf_select()
            else:
                leaf, path = self.greedy_select()

            # Generate and evaluate new samples
            samples, objs, cost = self.lamoo_sampler_evaluator.random_sample_nasbench201(path=path)
            
            random_samples, random_objs, cost = self.base_sampler_evaluator.random_sample_nasbench201()
            
            self.samples[0] = torch.cat([self.samples[0], samples])
            self.samples[1] = torch.cat([self.samples[1], objs])
            self.random_samples[0] = torch.cat([self.random_samples[0], random_samples])
            self.random_samples[1] = torch.cat([self.random_samples[1], random_objs])

            # Compute hypervolume for random sampling
            train_obj = self.random_samples[1]
            pareto_mask = is_non_dominated(train_obj)
            pareto_y = train_obj[pareto_mask]
            if torch.cuda.is_available():
                pareto_y = torch.tensor(pareto_y, device='cuda')
            random_hv = botorch_hv.compute(pareto_y)
            self.random_hv_track.append(random_hv)

            self.init_surrogate_model()

            t1 = time.time()
            print(f"Time = {t1 - t0:>4.2f}")
            self.time_list.append(t1 - t0)

        print(f'HV in each iter: {self.hv_track}')
        print(f'Random HV in each iter: {self.random_hv_track}')
        print(f'Search time in each iter: {self.time_list}')

        print("path is ", path)
        self.path = path

    def get_best_samples(self):
        """
        Get the best samples found during the search.
        """
        pareto_mask = is_non_dominated(self.samples[1])
        return self.samples[0][pareto_mask], self.samples[1][pareto_mask]
