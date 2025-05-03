import json
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def load_results(results_path="results/evaluation_results.json"):
    """Loads evaluation results from a JSON file."""
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        return None
    try:
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        return results_data
    except Exception as e:
        print(f"Error loading results from {results_path}: {e}")
        return None

def plot_confusion_matrix(cm, display_labels, output_path="results/confusion_matrix_baseline.png"):
    """Plots and saves the confusion matrix."""
    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=display_labels)
        disp.plot(cmap=plt.cm.Blues) # Use Blues colormap
        plt.title("Confusion Matrix: Logistic Regression Baseline")
        
        # Ensure the results directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path)
        print(f"Confusion matrix plot saved to: {output_path}")
        plt.close() # Close the plot to free memory
    except Exception as e:
        print(f"Error plotting or saving confusion matrix: {e}")

if __name__ == "__main__":
    print("Generating confusion matrix plot...")
    
    # Define paths
    results_json_path = "results/evaluation_results.json"
    plot_output_path = "results/confusion_matrix_baseline.png"
    
    # Load results
    results = load_results(results_json_path)
    
    if results and "evaluation_metrics" in results and "confusion_matrix" in results["evaluation_metrics"]:
        # Extract confusion matrix
        confusion_mat = results["evaluation_metrics"]["confusion_matrix"]
        
        # Define class labels (0: Not Target, 1: Target)
        class_labels = ["Not As8 O12", "As8 O12"] 
        
        # Plot the matrix
        plot_confusion_matrix(confusion_mat, class_labels, plot_output_path)
    else:
        print("Could not generate plot due to missing data in results file.")
        
    print("Plot generation script finished.")