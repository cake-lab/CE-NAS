import numpy as np


def dominance_number(objectives):
    """
    Calculate the dominance number for each objective in a set of objectives.

    Args:
        objectives (np.ndarray): A 2D array of objective values for each sample.

    Returns:
        np.ndarray: An array of dominance counts for each sample.
    """
    n = objectives.shape[0]
    dominance_count = np.zeros(n, dtype=int)
    for i in range(n):
        dominance_count += np.all(objectives[i] <= objectives, axis=1)
    return dominance_count - 1  # Subtract 1 to exclude self-dominance


def sort_queue(original_queue):
    """
    Sort the queue based on dominance numbers of the objectives.

    Args:
        original_queue (list): A list of jobs, where each job is a list containing
                               various information including objectives.

    Returns:
        list: A list of sorted indices and their corresponding dominance counts.
    """
    # Extract objectives from the queue
    obj_list = np.array([job[2][0].cpu().numpy() for job in original_queue])

    # Negate objectives for minimization
    obj_list *= -1

    # Calculate dominance numbers
    dc = dominance_number(obj_list)

    # Create list of indices and dominance counts
    dc_index = list(enumerate(dc))

    # Add dominance count to each job in the original queue
    for job, count in zip(original_queue, dc):
        job.append(count)

    # Sort based on dominance count
    return sorted(dc_index, key=lambda x: x[1])




