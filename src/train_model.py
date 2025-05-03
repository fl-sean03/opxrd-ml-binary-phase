import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json # Needed for potential future loading of data if not passed directly
import os   # For creating directories

# Assuming the data loading and preprocessing is done elsewhere and provides features and labels
# For this script, we'll assume features (X) and labels (y) are loaded numpy arrays.

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        LogisticRegression: The trained Logistic Regression model.
    """
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Model training finished.")
    return model

def train_random_forest(X_train, y_train):
    """
    Trains a RandomForestClassifier model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        RandomForestClassifier: The trained RandomForestClassifier model.
    """
    print("Training RandomForestClassifier model...")
    model = RandomForestClassifier(random_state=42) # Use random_state for reproducibility
    model.fit(X_train, y_train)
    print("Model training finished.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model and prints performance metrics.

    Args:
        model (LogisticRegression): The trained model.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing labels.

    Returns:
        tuple[dict, np.ndarray]: A tuple containing:
            - dict: A dictionary containing evaluation metrics.
            - np.ndarray: The predicted labels (y_pred).
    """
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    # Use zero_division=0 to handle cases where precision/recall might be undefined (e.g., no positive predictions)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist() # Convert numpy array to list for potential serialization
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    print("Model evaluation finished.")
    return metrics, y_pred

if __name__ == "__main__":
    # This block executes when the script is run directly.
    # It orchestrates the full ML workflow:
    # 1. Configuration
    # 2. Data Loading and Preprocessing
    # 3. Data Splitting
    # 4. Model Training
    # 5. Model Evaluation
    # 6. Saving Results
    print("Starting full ML workflow: Parsing, Preprocessing, Training, Evaluation...")

    # Import necessary functions from other modules
    from parse_opxrd import iterate_opxrd_dataset, extract_pxrd_data, extract_metadata, create_binary_label
    from preprocess_data import interpolate_pattern, normalize_pattern
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from imblearn.over_sampling import SMOTE # Import SMOTE
    from sklearn.ensemble import RandomForestClassifier # Import RandomForestClassifier

    # --- 1. Configuration ---
    # Define key parameters for the workflow
    base_directory = "./opxrd_data/CNRS/" # Specify the dataset subset to process
    target_phase = "As8 O12" # Define the chemical formula for the positive class
    # Define the common 2-theta grid for interpolation (ensures consistent feature vector length)
    # Parameters based on PLANNING.md
    target_two_theta_grid = np.arange(10.0, 80.0, 0.02)
    feature_vector_length = len(target_two_theta_grid) # Length of the feature vector after interpolation
    test_set_size = 0.2 # Fraction of data to use for the test set
    random_state = 42 # Seed for random number generator for reproducible train/test splits

    print(f"Using dataset subset: {base_directory}")
    print(f"Target phase for classification: {target_phase}")
    print(f"Target 2-theta grid length: {feature_vector_length}")
    print(f"Test set size: {test_set_size}")

    # --- 2. Data Loading and Preprocessing ---
    print("\n--- Loading and Preprocessing Data ---")
    all_features = [] # List to store preprocessed intensity patterns (features)
    all_labels = []   # List to store corresponding binary labels
    processed_count = 0 # Counter for total files encountered
    successful_count = 0 # Counter for files successfully processed into features/labels

    # Use the iterator from parse_opxrd to load and parse each JSON file
    for filepath, pattern_data, pxrd_data, metadata, binary_label in iterate_opxrd_dataset(base_directory):
        processed_count += 1
        # Optional: print(f"Processing file: {filepath}")

        # Check if essential pXRD data (two_theta, intensity) was successfully extracted
        if pxrd_data and pxrd_data[0] is not None and pxrd_data[1] is not None:
            two_theta_values, intensities = pxrd_data

            # Apply preprocessing steps from preprocess_data module
            try:
                # Interpolate onto the common grid
                interpolated_intensities = interpolate_pattern(two_theta_values, intensities, target_two_theta_grid)
                # Normalize intensities (e.g., max scaling)
                normalized_intensities = normalize_pattern(interpolated_intensities)

                # Append the processed feature vector and its label to the lists
                all_features.append(normalized_intensities)
                all_labels.append(binary_label)
                successful_count += 1
            except Exception as e:
                # Catch potential errors during interpolation/normalization
                print(f"Error processing file {filepath}: {e}")
        # else: # Optional: Log files skipped due to missing data
            # print(f"Skipping file {filepath} due to missing pXRD data.")


    print(f"\nFinished processing {processed_count} files in {base_directory}")
    print(f"Successfully processed and collected data for {successful_count} patterns.")

    if successful_count == 0:
        print("No data successfully processed. Exiting.")
        exit()

    # Convert the lists of features and labels into NumPy arrays for scikit-learn compatibility
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)

    print(f"\nShape of features array: {features_array.shape}")
    print(f"Shape of labels array: {labels_array.shape}")

    # Display the distribution of classes (0s and 1s) to check for imbalance
    unique_labels, counts = np.unique(labels_array, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_labels, counts))}")

    # Define the directory for saving results and data
    results_dir = "./results/"
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # --- 3. Data Splitting ---
    print("\n--- Splitting Data ---")
    # Create an array of original indices (0 to N-1) to track samples after shuffling/splitting
    original_indices = np.arange(len(features_array))

    # Split the data into training and testing sets
    # stratify=labels_array ensures that the proportion of labels (0s and 1s) is
    # approximately the same in both the training and testing sets. This is crucial for imbalanced datasets.
    # We also split the original_indices array to keep track of which original sample ended up where.
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        features_array, labels_array, original_indices, # Data to split
        test_size=test_set_size,    # Proportion of data for the test set
        random_state=random_state,  # Ensures reproducibility
        stratify=labels_array       # Preserve class distribution
    )

    print(f"Data split: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}, idx_test {idx_test.shape}")

    # Save X_test and y_test for potential future use (e.g., plotting)
    try:
        np.save(os.path.join(results_dir, "X_test.npy"), X_test)
        np.save(os.path.join(results_dir, "y_test.npy"), y_test)
        print(f"Saved X_test.npy and y_test.npy to {results_dir}")
    except Exception as e:
        print(f"Error saving X_test.npy or y_test.npy: {e}")

    # --- 4. Apply SMOTE and Train Model ---
    print("\n--- Applying SMOTE and Training Model ---")
    
    # Instantiate SMOTE
    # random_state ensures reproducibility of the oversampling process
    smote = SMOTE(random_state=random_state)
    print("Applying SMOTE to the training data...")
    
    # Apply SMOTE only to the training data
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Original training set shape: {X_train.shape}")
    print(f"Resampled training set shape: {X_train_resampled.shape}")
    unique_labels_resampled, counts_resampled = np.unique(y_train_resampled, return_counts=True)
    print(f"Resampled training label distribution: {dict(zip(unique_labels_resampled, counts_resampled))}")
    
    # Train the Logistic Regression model using the *resampled* training data
    trained_model = train_logistic_regression(X_train_resampled, y_train_resampled)

    # --- 5. Model Evaluation ---
    print("\n--- Evaluating Model ---")
    # IMPORTANT: Evaluate the trained model on the *original*, unseen test data (X_test, y_test)
    # Do NOT evaluate on resampled test data.
    evaluation_metrics, y_pred = evaluate_model(trained_model, X_test, y_test)

    # --- 6. Save Results ---
    print("\n--- Saving Results ---")
    # Define the directory and filename for saving the SMOTE evaluation results
    # Use a different filename to distinguish from the baseline results
    results_dir = "./results/"
    results_file = os.path.join(results_dir, "evaluation_results_smote.json")

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Prepare the data structure to be saved as JSON
    # Convert NumPy arrays (like indices, labels) to lists for JSON compatibility
    results_data = {
        "evaluation_metrics": evaluation_metrics, # Dictionary of metrics (accuracy, precision, etc.)
        "test_indices": idx_test.tolist(),       # Original indices of the samples in the test set
        "true_labels": y_test.tolist(),          # True labels of the test set samples
        "predicted_labels": y_pred.tolist()      # Labels predicted by the model for the test set
    }

    # Write the results dictionary to the specified JSON file
    try:
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=4) # Use indent for readability
        print(f"Evaluation results saved to: {results_file}")
    except Exception as e:
        print(f"Error saving results to {results_file}: {e}")


    print("\nFull ML workflow finished.")
    # The detailed evaluation results (using SMOTE) are now persistently stored in evaluation_results_smote.json

    # --- 7. Train and Evaluate RandomForestClassifier (Baseline) ---
    print("\n--- Training and Evaluating RandomForestClassifier (Baseline) ---")

    # Train the RandomForestClassifier model using the *original* training data
    trained_rf_model_baseline = train_random_forest(X_train, y_train)

    # Evaluate the baseline RFC model on the *original*, unseen test data
    evaluation_metrics_rf_baseline, y_pred_rf_baseline = evaluate_model(trained_rf_model_baseline, X_test, y_test)

    # --- 8. Save Baseline RFC Results ---
    print("\n--- Saving Baseline RFC Results ---")
    results_file_rf_baseline = os.path.join(results_dir, "evaluation_results_rf.json")

    # Prepare the data structure to be saved as JSON
    results_data_rf_baseline = {
        "evaluation_metrics": evaluation_metrics_rf_baseline,
        "test_indices": idx_test.tolist(),
        "true_labels": y_test.tolist(),
        "predicted_labels": y_pred_rf_baseline.tolist()
    }

    # Write the results dictionary to the specified JSON file
    try:
        with open(results_file_rf_baseline, 'w') as f:
            json.dump(results_data_rf_baseline, f, indent=4)
        print(f"Baseline RFC evaluation results saved to: {results_file_rf_baseline}")
    except Exception as e:
        print(f"Error saving baseline RFC results to {results_file_rf_baseline}: {e}")

    # --- 9. Train and Evaluate RandomForestClassifier (SMOTE) ---
    print("\n--- Training and Evaluating RandomForestClassifier (SMOTE) ---")

    # Train the RandomForestClassifier model using the *resampled* training data
    trained_rf_model_smote = train_random_forest(X_train_resampled, y_train_resampled)

    # Evaluate the SMOTE RFC model on the *original*, unseen test data
    evaluation_metrics_rf_smote, y_pred_rf_smote = evaluate_model(trained_rf_model_smote, X_test, y_test)

    # --- 10. Save SMOTE RFC Results ---
    print("\n--- Saving SMOTE RFC Results ---")
    results_file_rf_smote = os.path.join(results_dir, "evaluation_results_rf_smote.json")

    # Prepare the data structure to be saved as JSON
    results_data_rf_smote = {
        "evaluation_metrics": evaluation_metrics_rf_smote,
        "test_indices": idx_test.tolist(),
        "true_labels": y_test.tolist(),
        "predicted_labels": y_pred_rf_smote.tolist()
    }

    # Write the results dictionary to the specified JSON file
    try:
        with open(results_file_rf_smote, 'w') as f:
            json.dump(results_data_rf_smote, f, indent=4)
        print(f"SMOTE RFC evaluation results saved to: {results_file_rf_smote}")
    except Exception as e:
        print(f"Error saving SMOTE RFC results to {results_file_rf_smote}: {e}")

    print("\nFull ML workflow finished, including RandomForestClassifier evaluations.")