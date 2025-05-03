import os
import json

def load_opxrd_pattern(filepath):
    """
    Loads a single opXRD pattern from a JSON file.

    Args:
        filepath (str): The full path to the JSON file.

    Returns:
        dict or None: The loaded JSON data as a dictionary, or None if an error occurs.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

def extract_pxrd_data(pattern_data):
    """
    Extracts core pXRD data (two-theta values and intensities) from a loaded pattern dictionary.

    Args:
        pattern_data (dict): The dictionary containing the loaded pattern data.

    Returns:
        tuple: A tuple containing two lists: (two_theta_values, intensities).
               Returns (None, None) if the required keys are not found.
    """
    two_theta_values = pattern_data.get('two_theta_values')
    intensities = pattern_data.get('intensities')

    if two_theta_values is None or intensities is None:
        print("Warning: 'two_theta_values' or 'intensities' key not found in pattern data.")
        return None, None

    return two_theta_values, intensities

def parse_label_string(label_string):
    """
    Parses the JSON string content within the 'label' field.

    Args:
        label_string (str): The string content of the 'label' field.

    Returns:
        dict or None: The parsed label data as a dictionary, or None if parsing fails.
    """
    if not isinstance(label_string, str):
        return None
    try:
        # The label string itself is a JSON string, potentially containing escaped quotes
        # We need to load it as JSON. The opXRD dataset sometimes stores JSON strings
        # *within* the 'label' field's string value, requiring this nested parsing.
        label_data = json.loads(label_string)
        
        # Further complexity: The 'phases' list within the label might contain *more* JSON strings.
        # We attempt to parse these nested strings individually.
        if 'phases' in label_data and isinstance(label_data['phases'], list):
            parsed_phases = []
            for phase_entry in label_data['phases']:
                if isinstance(phase_entry, str):
                    try:
                        parsed_phases.append(json.loads(phase_entry))
                    except json.JSONDecodeError:
                        parsed_phases.append(phase_entry) # Keep as string if parsing fails
                else:
                    parsed_phases.append(phase_entry) # Keep as is if not a string
            label_data['phases'] = parsed_phases
        return label_data
    except json.JSONDecodeError:
        print(f"Warning: Could not parse label string: {label_string[:100]}...") # Print first 100 chars
        return None

def extract_metadata(pattern_data):
    """
    Extracts relevant metadata from a loaded pattern dictionary.
    Handles variations in 'label' and 'metadata' keys, including parsing the label string.

    Args:
        pattern_data (dict): The dictionary containing the loaded pattern data.

    Returns:
        dict: A dictionary containing extracted metadata, with parsed label data.
    """
    raw_label = pattern_data.get('label')
    parsed_label = parse_label_string(raw_label)

    metadata = {
        'label': parsed_label, # Store the parsed label data
        'metadata': pattern_data.get('metadata'),
        'xray_info': pattern_data.get('xray_info'),
        'is_simulated': pattern_data.get('is_simulated'),
        'crystallite_size': pattern_data.get('crystallite_size'),
        'temp_in_celcius': pattern_data.get('temp_in_celcius'),
        # Add other potentially useful keys found during exploration
        'material_name': pattern_data.get('material_name'),
        'formula': pattern_data.get('formula'),
    }
    # Filter out keys where the value is None for potentially cleaner downstream processing.
    return {k: v for k, v in metadata.items() if v is not None}

def create_binary_label(metadata, target_phase="As8 O12"):
    """
    Creates a binary label (1 or 0) based on the presence of a target phase.

    Args:
        metadata (dict): The extracted metadata dictionary.
        target_phase (str): The chemical composition of the target phase.

    Returns:
        int: 1 if the target phase is found, 0 otherwise.
    """
    if metadata and 'label' in metadata and metadata['label'] and 'phases' in metadata['label']:
        for phase in metadata['label']['phases']:
            if isinstance(phase, dict) and 'chemical_composition' in phase:
                composition = phase['chemical_composition']
                if composition and composition.lower() == target_phase.lower():
                    return 1
            elif isinstance(phase, str) and phase.lower() == target_phase.lower():
                 # Fallback: Handle cases where nested parsing in parse_label_string might
                 # have failed, but the phase entry itself is a string matching the target.
                 return 1
    return 0


def iterate_opxrd_dataset(base_dir="./opxrd_data"):
    """
    Iterates through the opXRD dataset directory structure and yields pattern data.

    Args:
        base_dir (str): The base directory of the unzipped dataset.

    Yields:
        tuple: A tuple containing (filepath, pattern_data, pxrd_data, metadata, binary_label)
               for each successfully loaded and parsed pattern.
    """
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                filepath = os.path.join(root, file)
                # print(f"Processing file: {filepath}") # Uncomment for detailed file processing info

                pattern_data = load_opxrd_pattern(filepath)
                if pattern_data:
                    pxrd_data = extract_pxrd_data(pattern_data)
                    metadata = extract_metadata(pattern_data)
                    binary_label = create_binary_label(metadata)
                    yield filepath, pattern_data, pxrd_data, metadata, binary_label

if __name__ == "__main__":
    # Example Usage: Demonstrates iterating through a subset, parsing, preprocessing,
    # and collecting data. Note that the main ML workflow (including training and
    # evaluation) is orchestrated in `src/train_model.py`.
    print("Starting opXRD dataset parsing and preprocessing example...")

    # Import preprocessing functions
    from preprocess_data import interpolate_pattern, normalize_pattern
    import numpy as np

    base_directory = "./opxrd_data/CNRS/" # Focus on CNRS for initial exploration
    target_phase = "As8 O12"

    # Define the target 2-theta grid for interpolation (e.g., 10 to 80 degrees with 0.02 step)
    # This range and step size should be chosen based on typical pXRD data and the project requirements.
    # For this example, let's use a smaller range for quicker output.
    # Refer to PLANNING.md for the intended range (10-80 deg, 0.02 deg step)
    target_two_theta_grid = np.arange(10.0, 80.0, 0.02)
    feature_vector_length = len(target_two_theta_grid)

    all_features = []
    all_labels = []
    processed_count = 0
    successful_count = 0

    print(f"Target 2-theta grid length: {feature_vector_length}")

    for filepath, pattern_data, pxrd_data, metadata, binary_label in iterate_opxrd_dataset(base_directory):
        processed_count += 1
        # print(f"Processing file: {filepath}") # Uncomment for detailed file processing info

        if pxrd_data and pxrd_data[0] and pxrd_data[1]:
            two_theta_values, intensities = pxrd_data

            # Apply preprocessing
            interpolated_intensities = interpolate_pattern(two_theta_values, intensities, target_two_theta_grid)
            normalized_intensities = normalize_pattern(interpolated_intensities)

            # Store features and labels
            all_features.append(normalized_intensities)
            all_labels.append(binary_label)
            successful_count += 1
        # else:
            # print(f"Skipping file {filepath} due to missing pXRD data.") # Uncomment to see skipped files


    print(f"\nFinished processing {processed_count} files in {base_directory}")
    print(f"Successfully processed and collected data for {successful_count} patterns.")

    # Convert lists to numpy arrays
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)

    print(f"\nShape of features array: {features_array.shape}")
    print(f"Shape of labels array: {labels_array.shape}")

    # Basic check on label distribution
    unique_labels, counts = np.unique(labels_array, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_labels, counts))}")


    print("\nExample usage finished. Data is now parsed, preprocessed, and ready for ML model training.")