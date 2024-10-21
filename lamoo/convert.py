import json

def convert_operation_to_float(operation):
    """
    Convert a neural network operation to a float value.

    Args:
        operation (str): The operation name.

    Returns:
        float: The corresponding float value for the operation.
    """
    operation_mapping = {
        'none': 0.0,
        'skip_connect': 0.2,
        'nor_conv_1x1': 0.4,
        'nor_conv_3x3': 0.6,
        'avg_pool_3x3': 0.8  # Assuming 'avg_pool_3x3' for the last case
    }
    return operation_mapping.get(operation, 0.8)  # Default to 0.8 if operation is not found

def convert_architecture(arch):
    """
    Convert a neural network architecture to a list of float values.

    Args:
        arch (list): The architecture representation.

    Returns:
        list: The converted architecture as a list of float values.
    """
    return [convert_operation_to_float(op[0]) for cell in arch for op in cell]

def main():
    """
    Main function to convert the dataset from one format to another.
    """
    input_file = 'few-shot-supernet'
    output_file = 'few-shot-supernet_normal'

    # Load the original dataset
    with open(input_file, 'r') as f:
        dataset_oneshot = json.load(f)

    # Convert the dataset
    new_data = {str(convert_architecture(eval(arch))): data 
                for arch, data in dataset_oneshot.items()}

    # Save the converted dataset
    with open(output_file, 'w') as f:
        json.dump(new_data, f)

    print(f"Conversion complete. Output saved to {output_file}")

if __name__ == "__main__":
    main()
