import numpy as np
from itertools import product

def generate_candidates(num_operations=5, num_edges=6):
    """
    Generate all possible candidates for NASBench201 architecture.

    Args:
        num_operations (int): Number of possible operations (default: 5)
        num_edges (int): Number of edges in the architecture (default: 6)

    Returns:
        np.array: Array of all possible candidates
    """
    # Generate all combinations using itertools.product
    cands = list(product(range(num_operations), repeat=num_edges))
    
    # Convert to numpy array for efficient processing
    return np.array(cands)

# Generate the candidates
candidates = generate_candidates()

# If you need the normalized version (commented out in the original code),
# you can uncomment the following line:
# candidates_normalized = candidates / (num_operations - 1)



# If you need to access the candidates from other modules, you can assign it to a variable
# that can be imported, like this:
# NASBENCH201_CANDIDATES = candidates
