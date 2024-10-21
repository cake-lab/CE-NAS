import numpy as np
from Node import Node
from utils import latin_hypercube, from_unit_cube, convert_dtype
import argparse
import torch
import matplotlib.pyplot as plt

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
import time
from sampler_evaluator import LamooSamplerEvaluator, base_sampler_evaluator

### hello


class MCTS:
    """
    Monte Carlo Tree Search for the most promising region in the search space.
    """

    def __init__(self, lb, ub, dims, ninits, func, args, run=0):
        self.dims = dims
        self.samples = []
        self.random_samples = []
        self.nodes = []
        self.Cp = args.cp
        self.lb = lb
        self.ub = ub
        self.ninits = ninits
        self.func = func
        self.visualization = False
        self.LEAF_SAMPLE_SIZE = 10     ##### minimum leaf size
        self.MAX_TREE_HEIGHT = 3
        self.path = None
        self.gpu_nums = 10
        self.cur_cost = 0

        self.run_times = run
        self.args = args
        self.datetime = [0]

        self.lamoo_sampler_evaluator = LamooSamplerEvaluator(problem=self.func, nums_samples=self.args.sample_num)
        self.base_sampler_evaluator = base_sampler_evaluator(problem=self.func, nums_samples=self.args.sample_num)


        self.ROOT = Node(args=self.args, parent=None, dims=self.dims, reset_id=True, cp=self.Cp)
        self.nodes.append(self.ROOT)

        self.init_train()
        self.init_surrogate_model()

    def init_train(self):
        """
        Get the initialization data.
        :return:
        """

        # init_samples, init_obj = self.base_sampler_evaluator.sobol_sample(n_pts=self.ninits)
        init_samples, init_obj, cost = self.base_sampler_evaluator.random_sample_nasbench201(n_pts=self.ninits)
        assert len(cost) <= self.gpu_nums
        self.cur_cost = cost

        self.lamoo_sampler_evaluator.nasbench201_space = \
            np.delete(self.lamoo_sampler_evaluator.nasbench201_space, self.base_sampler_evaluator.init_delete, axis=0)

        if torch.cuda.is_available():
            init_samples, init_obj = torch.tensor(init_samples, device='cuda'), torch.tensor(init_obj, device='cuda')

        self.samples.append(init_samples)
        # self.samples.append(init_obj)
        self.samples.append(init_obj)

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
        This is for Bayesian optimization. (Initialize the surrogate model)
        :return:
        """
        train_x = self.samples[0]
        train_obj = self.samples[1]

        self.surrogate_model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
        self.mll = ExactMarginalLogLikelihood(self.surrogate_model.likelihood, self.surrogate_model)
        fit_gpytorch_model(self.mll)

    def populate_training_data(self):
        """
        Rebuild the tree when run a new search iteration.
        :return:
        """
        #only keep root
        self.ROOT.obj_counter = 0
        for node in self.nodes:
            node.clear_data()
        self.nodes.clear()
        new_root = Node(args=self.args, parent=None, dims=self.dims, reset_id=True, cp=self.Cp)
        self.nodes.append(new_root)


        self.ROOT = new_root
        self.ROOT.update_bag(self.samples, self.func.ref_point)
    
    def get_leaf_status(self):
        """
        Return if the leaf node is splittable or not. The termination condition includes:
            1. Size of the leaf.
            2. Height of the tree.
            3. If the samples in the leaf is svm splittable.
        :return: list
        """
        status = []
        for node in self.nodes:
            # print('cur_node height is', node.height, node.is_svm_splittable, node.is_leaf())
            if node.is_leaf() is True and len(node.bag[0]) > self.LEAF_SAMPLE_SIZE \
                    and node.is_svm_splittable is True and node.height + 1 <= self.MAX_TREE_HEIGHT:
                status.append(True)
            else:
                status.append(False)
        return np.array(status)
        
    def get_split_idx(self):
        """
        Return the splittable node ID.
        :return: list
        """
        split_by_samples = np.argwhere(self.get_leaf_status() == True).reshape(-1)
        return split_by_samples
    
    def is_splittable(self):
        """
        Return True is there exists leaves that can be splittable, otherwise return False.
        :return: boolean
        """
        status = self.get_leaf_status()

        if True in status:
            return True
        else:
            return False
        
    def dynamic_treeify(self):
        """
        We bifurate a node into a good and a bad kid
        :return:
        """

        self.populate_training_data()
        print("total nodes:", len(self.nodes))
        assert len(self.ROOT.bag[0]) == len(self.samples[0])
        assert len(self.nodes) == 1
        
        print("keep splitting:", self.is_splittable(), self.get_split_idx())
        
        while self.is_splittable():
            to_split = self.get_split_idx()
            print("==>to split:", to_split, " total:", len(self.nodes))
            for nidx in to_split:
                parent = self.nodes[nidx] # parent check if the boundary is splittable by svm
                parent_height = parent.height
                assert len(parent.bag[0]) >= self.LEAF_SAMPLE_SIZE
                assert parent.is_svm_splittable is True

                good_kid_data, bad_kid_data = parent.train_and_split()


                assert len(good_kid_data[0]) + len(bad_kid_data[0]) == len(parent.bag[0])
                assert len(good_kid_data[0]) > 0
                assert len(bad_kid_data[0]) > 0

                good_kid = Node(args=self.args, parent=parent, dims=self.dims, reset_id=False,
                                cp=self.Cp, height=parent_height+1)
                bad_kid = Node(args=self.args, parent=parent, dims=self.dims, reset_id=False,
                               cp=self.Cp, height=parent_height+1)


                good_kid.update_bag(good_kid_data, self.func.ref_point)
                bad_kid.update_bag(bad_kid_data, self.func.ref_point)
            
                parent.update_kids(good_kid=good_kid, bad_kid=bad_kid)
            
                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)

            #CAUTION: make sure the good kid in into list first
        
        self.print_tree()

        
    def print_tree(self):
        """
        Print out all nodes' information
        :return:
        """
        print('-'*100)
        for node in self.nodes:
            print(node)
        print('-'*100)


    def track_nodes(self):
        """
        Visualize the node x_bar.
        :return:
        """
        assert len(self.nodes) > 0
        node = self.nodes[0]

        selected_nodes = []
        hvs = []
        stack = [node]

        while stack:
            node = stack.pop(0)
            selected_nodes.append(node.get_name())
            hvs.append(node.get_xbar())
            if not node.is_leaf():
                stack.append(node.kids[0])
                stack.append(node.kids[1])


        plt.bar(x=selected_nodes, height=hvs, label='hv')
        plt.savefig('progress_chart.png')


    def greedy_select(self):
        """
        Select the node with the best hv value. (always select the good direction)
        :return:
        """
        self.reset_to_root()
        curt_node = self.ROOT
        path = []

        while not curt_node.is_leaf():
            UCT = []
            k_good = {}
            k_good['good'] = -1
            k_good['bad'] = -1

            for i in curt_node.kids:
                if i.is_good_kid:
                    k_good['good'] = i.get_xbar()
                else:
                    k_good['bad'] = i.get_xbar()

                if k_good['good'] == k_good['bad'] and i.is_good_kid:
                    UCT.append(i.get_xbar() + 0.01)
                else:
                    UCT.append(i.get_xbar())

            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append((curt_node, choice))
            curt_node = curt_node.kids[choice]

            print("=>", curt_node.get_name(), choice,  end=' ')

        return curt_node, path
        

    def select(self):
        """
        Select the best node based on rule of MCTS.
        :return:
        """

        curt_node = self.ROOT

        path = []
        while not curt_node.is_leaf():
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_uct())
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append((curt_node, choice))
            curt_node = curt_node.kids[choice]
            print("=>", curt_node.get_name(), choice, end=' ')
            print('\n')

        return curt_node, path

    def select_bad(self):
        """
        Select the worst node based on rule of MCTS.
        :return:
        """

        curt_node = self.ROOT

        path = []
        while not curt_node.is_leaf():
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_uct())
            choice = np.random.choice(np.argwhere(UCT == np.amin(UCT)).reshape(-1), 1)[0]
            path.append((curt_node, choice))
            curt_node = curt_node.kids[choice]
            print("=>", curt_node.get_name(), choice, end=' ')
            print('\n')

        return curt_node, path

    def leaf_select(self):
        """
        Leaf based selection strategy. (Select the best leaf first, and get the path from root to the selected leaf.)
        :return:
        """

        leaves = []
        for node in self.nodes:
            if node.is_leaf():
                leaves.append(node)
        paths = []
        for leaf in leaves:
            path = []
            while leaf is not self.ROOT:
                path.insert(0, (leaf.parent, 0 if leaf.is_good_kid() else 1))
                leaf = leaf.parent
            paths.append(path)

        if len(paths) > 1:
            UCT = []
            for i in range(len(paths)):
                UCT.append(paths[i][-1][0].kids[paths[i][-1][1]].get_uct())

            leaf = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path = paths[leaf]
            curt_node = path[-1][0]
        else:
            path = paths[0]
            curt_node = self.ROOT

        dis = []
        for node in path:
            dis.append((node[0].get_name(), node[1]))
        return curt_node, path
    
    def backpropogate(self, leaf, acc):
        """
        Backpropogation operation
        :param leaf: leaf node
        :param acc: hypervolume
        :return: None
        """
        curt_node = leaf
        while curt_node is not None:
            assert curt_node.n > 0
            curt_node.x_bar = (curt_node.x_bar*curt_node.n + acc) / (curt_node.n + 1)
            curt_node.n += 1
            curt_node = curt_node.parent


    def search(self):
        """
        MCTS for the best region and generate new samples in that region.
        We omit the backpropogation operation in this version.
        :return:
        """

        botorch_hv = Hypervolume(ref_point=self.func.ref_point.clone().detach())
        print('MAX HV IS!!!!!!!!!', self.func._max_hv)
        self.time_list = []


        '''
        Get hypervolume of first iteration. 
        '''
        sample_num = self.args.sample_num
        train_obj = self.samples[1]
        pareto_mask = is_non_dominated(train_obj)
        pareto_y = train_obj[pareto_mask]

        if torch.cuda.is_available():
            pareto_y = torch.tensor(pareto_y, device='cuda')
        hv = botorch_hv.compute(pareto_y)
        self.hv_track = []
        self.random_hv_track = []


        for i in range(0, self.args.iter):
            t0 = time.time()

            self.dynamic_treeify()  # build the tree
            if i == 0:
                self.random_hv_track.append(self.nodes[0].get_xbar())
            self.hv_track.append(self.nodes[0].get_xbar())

            '''
            Print out the track of hypervolume.
            '''
            print('hv in each iter is', self.hv_track)
            print('random hv in each iter is', self.random_hv_track)
            print('search time in each iter is', self.time_list)



            '''
            traverse the path from the root to the leaf
            '''
            if self.args.node_select == 'mcts':
                leaf, path = self.select()
                bad_leaf, bad_path = self.select_bad()

            elif self.args.node_select == 'leaf':
                print('this is leaf selection')
                leaf, path = self.leaf_select()
                # bad_leaf, bad_path = self.select_bad()
            else:
                leaf, path = self.greedy_select()


            '''
            Generate new samples by different sampling method, and evaluate the generated samples.
            '''
            train_x = self.samples[0]
            train_obj = self.samples[1]
            # samples, objs = self.lamoo_sampler_evaluator.random_sample(path=path)

            samples, objs, cost = self.lamoo_sampler_evaluator.random_sample_nasbench201(path=path)
            # print('cost is', cost)
            self.datetime.append(max(cost))




            # samples, objs = self.lamoo_sampler.qnehvi_sample(model=self.surrogate_model,
            #                                                                path=path,
            #                                                               train_x=train_x,
            #                                                               train_obj=train_obj,
            #                                                               )
            # random_samples, random_objs = self.base_sampler_evaluator.random_sample()

            # random_samples, random_objs = self.base_sampler_evaluator.qnehvi_sample(model=self.surrogate_model,
            #                                 train_x=train_x,
            #                                 train_obj=train_obj)
            random_samples, random_objs, cost = self.base_sampler_evaluator.random_sample_nasbench201()
            self.samples[0] = torch.cat([self.samples[0], samples])
            self.samples[1] = torch.cat([self.samples[1], objs])

            self.random_samples[0] = torch.cat([self.random_samples[0], random_samples])
            self.random_samples[1] = torch.cat([self.random_samples[1], random_objs])

            '''
            Compute the hypervolume by random sampling strategy. We skip to compute the hypervolume by LaMOO 
            since we can directly call self.node[0].get_xbar() in next search iteration. 
            '''
            train_obj = self.random_samples[1]
            pareto_mask = is_non_dominated(train_obj)
            pareto_y = train_obj[pareto_mask]

            if torch.cuda.is_available():
                pareto_y = torch.tensor(pareto_y, device='cuda')
            random_hv = botorch_hv.compute(pareto_y)
            self.random_hv_track.append(random_hv)

            # if self.args.sample_method == 'bayesian':
            self.init_surrogate_model()

            '''
            Compute the time cost.
            '''
            t1 = time.time()

            print(
                f"time = {t1 - t0:>4.2f}.", end=""
            )
            print('\n')
            self.time_list.append(t1 - t0)
            # self.track_nodes()

        print('hv in each iter is', self.hv_track)
        print('random hv in each iter is', self.random_hv_track)
        print('search time in each iter is', self.time_list)

        self.path = path












