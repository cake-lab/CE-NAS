import logging
import argparse
import torch
import sys
import os
import json
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from carbon_reader import carbon_trace
import time
from multiprocessing import Manager
import multiprocessing as mp

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_path = os.getcwd()
working_folder = current_path.split('/')[-1]
if working_folder == 'simulator':
    lamoo_path = current_path[:-9] + 'lamoo'
else:
    lamoo_path = current_path + 'lamoo'


parent_folder_path = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), lamoo_path)))
sys.path.append(parent_folder_path)
sys.path.append(os.path.join(parent_folder_path, "lamoo"))


from lamoo.SearchEngine import MCTS
import numpy as np
from lamoo.tasks.nasbench201 import cifar10 as nasbench201_cifar10
from lamoo.tasks.nasbench201 import cifar10_oneshot as nasbench201_cifar10_oneshot


from lamoo.sampler import LamooSampler, base_sampler
from simulator_utils import sort_queue
from torch.multiprocessing import Process, Lock, Queue, set_start_method, Pool, Manager
import torch.multiprocessing as mp
from a3c_params import *
from A3C import A3C
from RLnet import *


def vanilla_worker(id, vanilla_func, samples, worker_budget, gpu_queue, cur_step_cost, timestamp, cur_time, lock):
    """
    Worker function for vanilla evaluation.

    Args:
        id (int): Worker ID
        vanilla_func (function): Vanilla evaluation function
        samples (list): Current observed samples
        worker_budget (dict): Budget list for all GPUs
        gpu_queue (dict): Queue of jobs
        cur_step_cost (multiprocessing.Value): The carbon cost in current step
        timestamp (dict): Record the results hourly
        cur_time (multiprocessing.Value): Current time point
        lock (multiprocessing.Lock): Multiprocessing lock for synchronization

    Returns:
        None
    """
    logger.info(f"Vanilla worker {id} starting, remaining budget: {worker_budget[id]}")

    with lock:
        # Check if there's any unfinished job in the halfwork queue
        if len(gpu_queue['halfwork']) > 0:
            dc_index = sort_queue(gpu_queue['halfwork'])
            cur_job = gpu_queue['halfwork'].pop(dc_index[0][0])
        else:
            # Wait for new samples if vanilla queue is empty
            while len(gpu_queue['vanilla']) == 0:
                pass

            dc_index = sort_queue(gpu_queue['vanilla'])
            cur_job = gpu_queue['vanilla'].pop(dc_index[0][0])

    # cur_job structure: [sample_encoding, actual_carbon_cost, sample_value]
    if cur_job[1] is None:
        # Compute actual training cost if not known
        vanilla_cost = vanilla_func.evaluation_cost(cur_job[0])[0]
        cur_job[1] = vanilla_cost

        # compute the remaining budget of the working GPU
        remaining_time = vanilla_cost - worker_budget[id]
    else:
        remaining_time = cur_job[1] - worker_budget[id]

    # If the work is not finished, add the work into the halfwork queue.
    if remaining_time > 0:
        # Job not finished, add to halfwork queue
        cur_sample_cost = worker_budget[id]
        worker_budget[id] = 0
        cur_job[1] = remaining_time
        with lock:
            gpu_queue['halfwork'].append(cur_job)
            # Update carbon cost
            cur_step_cost.value += cur_sample_cost * carbon_trace[cur_time] / 3600
    else:
        # Job finished, update samples and timestamp
        with lock:
            cur_sample_cost = cur_job[1]
            cur_job[2] = vanilla_func(cur_job[0])
            samples[0] = torch.cat([samples[0], torch.tensor(cur_job[0])])
            samples[1] = torch.cat([samples[1], cur_job[2]])

            """
            Push the finished evaluation sample into the timestamp. 
            Case1: If current timestamp is empty(only in the first step), make the sample as the first into the timestamp.
            Case2: If current timestamp is not empty, add the sample to the timestamp.
            """
            timestamp[cur_time]['samples'] = torch.cat([timestamp[cur_time]['samples'], torch.tensor(cur_job[0])])
            timestamp[cur_time]['obj'] = torch.cat([timestamp[cur_time]['obj'], cur_job[2]])

            # Update carbon cost
            cur_step_cost.value += cur_sample_cost * carbon_trace[cur_time] / 3600

        worker_budget[id] = -remaining_time

    logger.info(f"Vanilla worker {id} finished, remaining budget: {worker_budget[id]}")

def vanilla(vanilla_func, samples, vanillaGPUsList, worker_budget, gpu_queue, cur_step_cost, timestamp, cur_time, lock):
    """
    Main function for vanilla evaluation.

    Args:
        vanilla_func (function): Vanilla evaluation function
        samples (list): Current observed samples
        vanillaGPUsList (list): GPU IDs for vanilla NAS
        worker_budget (dict): Budget list for all GPUs
        gpu_queue (dict): Queue of jobs
        cur_step_cost (multiprocessing.Value): The carbon cost in current step
        timestamp (dict): Record the results hourly
        cur_time (int): Current time point
        lock (multiprocessing.Lock): Multiprocessing lock for synchronization

    Returns:
        None
    """

    while any([worker_budget[id] for id in vanillaGPUsList]):
        while len(gpu_queue['halfwork']) > 0 or len(gpu_queue['vanilla']) > 0:
            process = []
            for id in vanillaGPUsList:
                if worker_budget[id] > 0:
                    p = Process(target=vanilla_worker, args=(id, vanilla_func, samples, worker_budget, gpu_queue,
                                                             cur_step_cost, timestamp, cur_time, lock))
                    process.append(p)
                    p.start()

            for p in process:
                p.join()
            if not any([worker_budget[id] for id in vanillaGPUsList]):
                break
    return


def oneshot_worker(id, oneshot_func, gpu_queue, worker_budget, lock):
    """
    Worker function for one-shot evaluation.

    :param id: worker id
    :param oneshot_func: one-shot evaluation function
    :param gpu_queue: queue of jobs
    :param worker_budget: budget list for all GPUs
    :param lock: multiprocessing lock for synchronization
    """
    # Get the current one-shot job from the GPU queue
    lock.acquire()
    cur_job = gpu_queue['oneshot'].pop(0)
    lock.release()
    x = cur_job[0]
    x_cost = cur_job[1]

    # compute the remaining budget of the working GPU
    remaining_time = x_cost - worker_budget[id]

    # we assume oneshot evaluation cannot run out of the remaining budget
    assert remaining_time < 0

    # compute remaining budget
    worker_budget[id] = -remaining_time

    """
    Push the architecture into the vanilla queue. 
    # Dim 0 is the sample encoding. 
    # Dim 1 is the actual cost, but it is unknown now
    # Dim 2 is the inference cost by supernet evaluation
    """
    gpu_queue['vanilla'].append([x, None, oneshot_func(x)])


def oneshot(oneshot_func, sampler, vanillaGPUsList, oneshotGPUsList, worker_budget, gpu_queue, agent, path, args, cur_time, lock):
    """
    Perform oneshot neural architecture search.

    This function manages the oneshot evaluation process, including sampling architectures,
    dispatching jobs to workers, and updating the job queue.

    Args:
        oneshot_func (callable): The oneshot evaluation function.
        sampler (LamooSampler): Sampler object for generating new architectures.
        vanillaGPUsList (list): List of GPUs allocated for vanilla evaluation.
        oneshotGPUsList (list): List of GPUs allocated for oneshot evaluation.
        worker_budget (dict): Budget allocation for each worker.
        gpu_queue (dict): Queues for different types of jobs.
        agent (MCTS): The MCTS agent.
        path (list): Current search path in the MCTS tree.
        args (argparse.Namespace): Command-line arguments.
        cur_time (int): Current timestamp.
        lock (multiprocessing.Lock): Lock for synchronizing access to shared resources.

    Returns:
        None
    """
    if len(vanillaGPUsList) == 0:
        while len(gpu_queue['vanilla']) <= args.queueCapacity:
            # Sample new architectures and add them to the oneshot queue
            for _ in range(len(oneshotGPUsList)):
                if args.sample_method == 'bayesian':
                    new_x = sampler.qnehvi_sample(model=agent.surrogate_model, path=path, train_x=agent.samples[0], train_obj=agent.samples[1]).clone().cpu()
                else:
                    new_x = sampler.random_sample(path=path, n_pts=1).clone().cpu()
                oneshot_cost = oneshot_func.evaluation_cost(new_x)
                new_x = new_x.tolist()
                with lock:
                    gpu_queue['oneshot'].append([new_x, oneshot_cost[0]])

            # Start oneshot worker processes
            processes = []
            for id in oneshotGPUsList:
                if worker_budget[id] > 0:
                    p = Process(target=oneshot_worker, args=(id, oneshot_func, gpu_queue, worker_budget, lock))
                    processes.append(p)
                    p.start()

            # Wait for all processes to complete
            for p in processes:
                p.join()

    else:
        while any([worker_budget[id] for id in vanillaGPUsList]):
            while len(gpu_queue['vanilla']) <= args.queueCapacity:
                for i in range(len(oneshotGPUsList)):
                    # Generate new sample by LaMOO
                    if args.sample_method == 'bayesian':
                        new_x = sampler.qnehvi_sample(model=agent.surrogate_model, path=path, train_x=agent.samples[0], train_obj=agent.samples[1]).clone().cpu()
                    else:
                        new_x = sampler.random_sample(path=path, n_pts=1).clone().cpu()
                    # compute one-shot evaluation cost, it is a list type.
                    oneshot_cost = oneshot_func.evaluation_cost(new_x)
                    new_x = new_x.tolist()

                    # push the new sample into current GPU queue
                    with lock:
                        gpu_queue['oneshot'].append([new_x, oneshot_cost[0]])

                # Start oneshot worker processes
                processes = []
                for id in oneshotGPUsList:
                    if worker_budget[id] > 0:
                        p = Process(target=oneshot_worker, args=(id, oneshot_func, gpu_queue, worker_budget, lock))
                        processes.append(p)
                        p.start()

                # Wait for all processes to complete
                for p in processes:
                    p.join()

    return

def run_processes(OneshotNas, VanillaNas, vanillaGPUsNums, oneshotGPUsNums,
                  worker_budget, gpu_queue, cur_step_cost, timestamp,
                  cur_time, path, args):
    """
    Run oneshot and vanilla processes.

    Args:
        OneshotNas (object): Oneshot NAS object
        VanillaNas (object): Vanilla NAS object
        vanillaGPUsNums (int): Number of GPUs for vanilla NAS
        oneshotGPUsNums (int): Number of GPUs for oneshot NAS
        worker_budget (dict): Budget list for all GPUs
        gpu_queue (dict): Queue of jobs
        cur_step_cost (multiprocessing.Value): The carbon cost in current step
        timestamp (dict): Record the results hourly
        cur_time (multiprocessing.Value): Current time point
        path (list): Path for sampling
        args (argparse.Namespace): Command line arguments

    Returns:
        None
    """
    vanillaGPUsList = manager.list(range(vanillaGPUsNums))
    oneshotGPUsList = manager.list(range(vanillaGPUsNums, args.GPU_nums))
    oneshot_thread = mp.Process(target=oneshot, args=(OneshotNas, lamoo_sampler,
                                                      vanillaGPUsList,
                                                      oneshotGPUsList,
                                                      worker_budget, gpu_queue, 
                                                      agent, path, args, cur_time, oneshot_locker))

    vanilla_thread = mp.Process(target=vanilla, args=(VanillaNas, agent.samples,
                                                      vanillaGPUsList,
                                                      worker_budget, gpu_queue,
                                                      cur_step_cost, timestamp,
                                                      cur_time.value, vanilla_locker))

    oneshot_thread.start()
    vanilla_thread.start()
    oneshot_thread.join()
    vanilla_thread.join()

def update_state_and_reward(timestamp, cur_time, botorch_hv, cur_step_cost,
                            budget, args, s_batch, a_batch, r_batch, cur_state, cur_action):
    """
    Update state and reward after each step.

    Args:
        timestamp (dict): Record the results hourly
        cur_time (multiprocessing.Value): Current time point
        botorch_hv (Hypervolume): Hypervolume computer
        cur_step_cost (multiprocessing.Value): The carbon cost in current step
        budget (int): Total time budget
        args (argparse.Namespace): Command line arguments
        s_batch (list): State batch
        a_batch (list): Action batch
        r_batch (list): Reward batch
        cur_state (torch.Tensor): Current state
        cur_action (int): Current action

    Returns:
        None
    """
    train_obj = timestamp[cur_time.value]['obj']
    pareto_mask = is_non_dominated(train_obj)
    pareto_y = train_obj[pareto_mask]

    if torch.cuda.is_available():
        pareto_y = pareto_y.clone().detach()
    hv = botorch_hv.compute(pareto_y)

    timestamp[cur_time.value]['hv'] = hv
    timestamp[cur_time.value]['cost'] += cur_step_cost.value

    cur_reward = [get_reward(prev_hv=timestamp[cur_time.value - 1]['hv'],
                             cur_hv=timestamp[cur_time.value]['hv'],
                             carbon_cost=cur_step_cost.value / args.GPU_nums,
                             remaining_budget=budget - cur_time.value,
                             n_samples=timestamp[cur_time.value]['samples'].size(0) -
                                       timestamp[cur_time.value - 1]['samples'].size(0))[0]]

    s_batch.append(cur_state)
    a_batch.append(cur_action)
    r_batch.append(cur_reward)

def update_rl_model(rl_model, s_batch, a_batch, r_batch, update_step):
    """
    Update the RL model.

    Args:
        rl_model (A3C): The A3C model
        s_batch (list): State batch
        a_batch (list): Action batch
        r_batch (list): Reward batch
        update_step (int): Current update step

    Returns:
        tuple: Updated s_batch, a_batch, r_batch
    """
    if update_step % 1 == 0:
        rl_model.is_central = True
        rl_model.getNetworkGradient(s_batch, a_batch, r_batch)
        rl_model.updateNetwork()
        return [], [], []
    return s_batch, a_batch, r_batch

def save_results(timestamp, args, vanilla_list, cur_time):
    """
    Save the results of the search process.

    Args:
        timestamp (dict): Record the results hourly
        args (argparse.Namespace): Command line arguments
        vanilla_list (list): List of vanilla NAS results
        cur_time (multiprocessing.Value): Current time point

    Returns:
        None
    """
    timestamp[cur_time.value]['samples'] = timestamp[cur_time.value - 1]['samples']
    timestamp[cur_time.value]['obj'] = timestamp[cur_time.value - 1]['obj']
    timestamp[cur_time.value]['cost'] = timestamp[cur_time.value - 1]['cost']

    copy_time_stamp = {key: dict(timestamp[key]) for key in timestamp}

    torch.save(dict(copy_time_stamp), f'results/RL/{args.run}_T80.pth')
    torch.save(vanilla_list, f'results/allocation/vanilla_list_{args.run}.pth')

def main_loop(args, agent, rl_model, timestamp, cur_time, budget):
    """
    Main loop of the search process.

    Args:
        args (argparse.Namespace): Command line arguments
        args (argparse.Namespace): Command line arguments
        agent (MCTS): The MCTS agent
        rl_model (A3C): The A3C model
        timestamp (dict): Record the results hourly
        cur_time (multiprocessing.Value): Current time point
        budget (int): Total time budget

    Returns:
        None
    """
    update_step = 0
    s_batch = []
    a_batch = []
    r_batch = []
    while cur_time.value < budget:
        cur_step_cost = manager.Value('f', 0.0)
        for i in range(args.GPU_nums):
            worker_budget[i] = 3600
        agent.dynamic_treeify()
        agent.init_surrogate_model()
        leaf, path = agent.leaf_select()

        cur_state = set_state(remaining_budget=budget - cur_time.value,
                              carbon_trace=carbon_trace[cur_time.value],
                              cur_hv=hv,
                              t_samples=timestamp[cur_time.value - 1]['samples'].size(0))
        rl_model.is_central = False
        cur_action = rl_model.actionSelect(cur_state)
        vanilla_rate = get_action(min(cur_action, 9.0))

        # Set GPU allocation
        vanillaGPUsNums = max(int(args.GPU_nums * vanilla_rate), 1)
        oneshotGPUsNums = args.GPU_nums - vanillaGPUsNums

        logger.info(f'Oneshot GPU number is {oneshotGPUsNums}')
        logger.info(f'Vanilla GPU number is {vanillaGPUsNums}')

        # Run oneshot and vanilla processes
        run_processes(OneshotNas, VanillaNas, vanillaGPUsNums, oneshotGPUsNums,
                      worker_budget, gpu_queue, cur_step_cost, timestamp,
                      cur_time, path, args)

        # Update state and reward
        update_state_and_reward(timestamp, cur_time, botorch_hv, cur_step_cost,
                                budget, args, s_batch, a_batch, r_batch, cur_state, cur_action)

        # RL model update
        s_batch, a_batch, r_batch = update_rl_model(rl_model, s_batch, a_batch, r_batch, update_step)

        cur_time.value += 1
        update_step += 1

        # Save results
        save_results(timestamp, args, vanilla_list, cur_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MCTS")
    parser.add_argument('--data_id', type=int, default=0, help='specific run id')
    parser.add_argument('--kernel', type=str, default='poly', help='kernel type of svm')
    parser.add_argument('--gamma', type=str, default='scale', help='auto or scale')
    parser.add_argument('--degree', type=int, default=4, help='svm degree')
    parser.add_argument('--iter', type=int, default=38, help='total iterations')
    parser.add_argument('--sample_num', type=int, default=5, help='sample numsbers per iteration')
    parser.add_argument('--runs', type=int, default=5, help='total runs')
    parser.add_argument('--cp', type=float, default=30, help='cp value in MCTS')
    parser.add_argument('--sample_method', type=str, default='random', help='bayesian or random')
    parser.add_argument('--node_select', type=str, default='leaf', help='mcts or leaf')
    parser.add_argument('--maxSamples', type=int, default=1500, help='maximum sample numbers')
    parser.add_argument('--queueCapacity', type=int, default=10, help='maximum length of the gpu queue')
    parser.add_argument('--GPU_nums', type=int, default=8, help='total numbers of GPUs')
    parser.add_argument('--budget', type=int, default=80, help='time budget')
    parser.add_argument('--run', type=int, default=0, help='current run time')
    args = parser.parse_args()

    vanilla_list = []

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    mp.set_start_method('spawn')

    # Define A3C network
    rl_model = A3C(IS_CENTRAL, model_type, state_len, action_len, load_checkpoint=False, continuous_action=continuous)

    manager = mp.Manager()

    # Initialize job queues for vanilla and oneshot NAS
    gpu_queue = manager.dict()
    gpu_queue['oneshot'] = manager.list()
    gpu_queue['vanilla'] = manager.list()
    gpu_queue['halfwork'] = manager.list()
    worker_budget = manager.dict()

    # Initialize timestamp to record hourly results
    timestamp = manager.dict()
    for i in range(len(carbon_trace)):
        timestamp[i] = manager.dict()
        timestamp[i]['samples'] = None
        timestamp[i]['obj'] = None
        timestamp[i]['cost'] = 0
        timestamp[i]['hv'] = 0

    logger.info(f'GPU is {"available" if torch.cuda.is_available() else "not available"}')

    # Initialize NAS objects
    OneshotNas = nasbench201_cifar10_oneshot.Nasbench201()
    VanillaNas = nasbench201_cifar10.Nasbench201()


    f = VanillaNas

    lamoo_sampler = LamooSampler(problem=OneshotNas, nums_samples=1)

    # Initialize MCTS agent
    agent = MCTS(lb=f.bounds[0].cpu().data.numpy(), ub=f.bounds[1].cpu().data.numpy(), dims=f.dim, ninits=10, args=args,
                  run=args.run, oneshot=OneshotNas, vanilla=VanillaNas)

    # Initialize botorch Hypervolume computer
    botorch_hv = Hypervolume(ref_point=f.ref_point.clone().cpu().detach())

    # Compute initial hypervolume and carbon cost
    train_obj = agent.samples[1]
    pareto_mask = is_non_dominated(train_obj)
    pareto_y = train_obj[pareto_mask]

    if torch.cuda.is_available():
        pareto_y = pareto_y.clone().detach()
    hv = botorch_hv.compute(pareto_y)

    # Initialize current search time
    cur_time = manager.Value('i', 0)

    # Add initial samples to timestamp
    timestamp[cur_time.value]['hv'] = hv
    timestamp[cur_time.value]['cost'] += 0
    timestamp[cur_time.value]['samples'] = agent.samples[0]
    timestamp[cur_time.value]['obj'] = agent.samples[1]
    logger.info(f"Current {cur_time.value} hypervolume and carbon cost are {timestamp[cur_time.value]['hv']} and {timestamp[cur_time.value]['cost']}")

    cur_time.value += 1
    timestamp[cur_time.value]['samples'] = timestamp[cur_time.value - 1]['samples']
    timestamp[cur_time.value]['obj'] = timestamp[cur_time.value - 1]['obj']

    # Initialize thread locks
    oneshot_locker = Lock()
    vanilla_locker = Lock()

    update_step = 0
    s_batch = []
    a_batch = []
    r_batch = []

    # Start main loop
    main_loop(args, agent, rl_model, timestamp, cur_time, args.budget)
