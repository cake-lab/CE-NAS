import pandas as pd
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    # Attempt to load data from two possible file paths
    try:
        return pd.read_csv('../electricity-maps-2020-2022-all-regions-marginal-average/US-CAL-CISO.csv')
    except FileNotFoundError:
        return pd.read_csv('../../electricity-maps-2020-2022-all-regions-marginal-average/US-CAL-CISO.csv')

def get_carbon_trace(data, start_index=13158):
    # Extract carbon intensity data starting from a specific index
    return data['carbon_intensity_avg'].tolist()[start_index:]

# Load the dataset
data = load_data()

# Get the carbon intensity trace
carbon_trace = get_carbon_trace(data)


# Calculate maximum and minimum carbon intensity
max_carbon = max(carbon_trace)
min_carbon = min(carbon_trace)

# Log the results
# logger.info(f'Maximum carbon intensity: {max_carbon}')
# logger.info(f'Minimum carbon intensity: {min_carbon}')


