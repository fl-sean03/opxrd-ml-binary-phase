import numpy as np
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define the target 2-theta grid used for interpolation in preprocess_data.py
# This is needed to plot the preprocessed patterns correctly.
# Parameters based on PLANNING.md and src/train_model.py
TARGET_TWO_THETA_GRID = np.arange(10.0, 80.0, 0.02)

def create_plots(results_dir="./results/", plots_dir="./plots/"):
    """
    Generates all required plots for the project analysis.

    Args:
        results_dir (str): Directory containing the evaluation results JSON files.
        plots_dir (str): Directory to save the generated plots.
    """
    print("Starting plot generation...")

    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)

    # --- Load Evaluation Results ---
    try:
        with open(os.path.join(results_dir, "evaluation_results_smote.json"), 'r') as f:
            smote_lr_results = json.load(f)
        with open(os.path.join(results_dir, "evaluation_results_rf.json"), 'r') as f:
            baseline_rf_results = json.load(f)
        with open(os.path.join(results_dir, "evaluation_results_rf_smote.json"), 'r') as f:
            smote_rf_results = json.load(f)
        # Need to load baseline LR results as well, assuming it's in evaluation_results.json
        # If not, we might need to re-run train_model.py or adjust.
        # For now, assume it's in evaluation_results.json based on previous context.
        with open(os.path.join(results_dir, "evaluation_results.json"), 'r') as f:
             baseline_lr_results = json.load(f)

    except FileNotFoundError as e:
        print(f"Error loading results file: {e}. Make sure train_model.py has been run.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from results file: {e}")
        return

    # --- Load Test Data ---
    try:
        X_test = np.load(os.path.join(results_dir, "X_test.npy"))
        y_test = np.load(os.path.join(results_dir, "y_test.npy"))
        print(f"Loaded X_test.npy and y_test.npy from {results_dir}")
    except FileNotFoundError as e:
        print(f"Error loading test data file: {e}. Make sure train_model.py has been run and saved X_test.npy and y_test.npy.")
        return

    # --- Task 1.1: Example pXRD Patterns ---
    print("Generating example pXRD patterns plot...")
    # Find indices for one True Positive (TP) and one True Negative (TN) sample
    # Using SMOTE LR results to find indices
    true_labels = np.array(smote_lr_results["true_labels"])
    predicted_labels = np.array(smote_lr_results["predicted_labels"])

    # Find a True Positive (True=1, Predicted=1)
    tp_indices = np.where((true_labels == 1) & (predicted_labels == 1))[0]
    tp_index = tp_indices[0] if len(tp_indices) > 0 else None

    # Find a True Negative (True=0, Predicted=0)
    tn_indices = np.where((true_labels == 0) & (predicted_labels == 0))[0]
    tn_index = tn_indices[0] if len(tn_indices) > 0 else None

    if tp_index is not None and tn_index is not None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot True Positive
        axes[0].plot(TARGET_TWO_THETA_GRID, X_test[tp_index])
        axes[0].set_title(f'Example True Positive Pattern (True: {true_labels[tp_index]}, Predicted: {predicted_labels[tp_index]})')
        axes[0].set_xlabel('2-theta (degrees)')
        axes[0].set_ylabel('Intensity (Normalized)')

        # Plot True Negative
        axes[1].plot(TARGET_TWO_THETA_GRID, X_test[tn_index])
        axes[1].set_title(f'Example True Negative Pattern (True: {true_labels[tn_index]}, Predicted: {predicted_labels[tn_index]})')
        axes[1].set_xlabel('2-theta (degrees)')
        axes[1].set_ylabel('Intensity (Normalized)')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "plot_example_patterns.png"))
        print(f"Saved plot_example_patterns.png to {plots_dir}")
        plt.close(fig)
    else:
        print("Could not find both a True Positive and True Negative sample for plotting.")

    # --- Task 1.2: Performance Metrics Comparison ---
    print("Generating performance metrics comparison plot...")
    # Extract metrics from loaded results
    metrics_data = {
        "Baseline LR": baseline_lr_results["evaluation_metrics"],
        "SMOTE LR": smote_lr_results["evaluation_metrics"],
        "Baseline RFC": baseline_rf_results["evaluation_metrics"],
        "SMOTE RFC": smote_rf_results["evaluation_metrics"]
    }

    metric_names = ["Precision (Class 1)", "Recall (Class 1)", "F1 Score (Class 1)"]
    model_names = list(metrics_data.keys())
    precision_scores = [metrics_data[model]['precision'] for model in model_names]
    recall_scores = [metrics_data[model]['recall'] for model in model_names]
    f1_scores = [metrics_data[model]['f1_score'] for model in model_names]

    x = np.arange(len(model_names))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width, precision_scores, width, label='Precision')
    rects2 = ax.bar(x, recall_scores, width, label='Recall')
    rects3 = ax.bar(x + width, f1_scores, width, label='F1 Score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics Comparison (Class 1)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    ax.set_ylim(0, 1.1) # Set y-axis limit for scores

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot_metrics_comparison.png"))
    print(f"Saved plot_metrics_comparison.png to {plots_dir}")
    plt.close(fig)


    # --- Task 1.3: Confusion Matrices ---
    print("Generating confusion matrix plots...")

    cm_data = {
        "Baseline LR": baseline_lr_results["evaluation_metrics"]["confusion_matrix"],
        "SMOTE LR": smote_lr_results["evaluation_metrics"]["confusion_matrix"],
        "Baseline RFC": baseline_rf_results["evaluation_metrics"]["confusion_matrix"],
        "SMOTE RFC": smote_rf_results["evaluation_metrics"]["confusion_matrix"]
    }

    for model_name, cm in cm_data.items():
        cm_array = np.array(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_array, display_labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title(f'Confusion Matrix: {model_name}')
        plt.savefig(os.path.join(plots_dir, f"plot_cm_{model_name.lower().replace(' ', '_')}.png"))
        print(f"Saved plot_cm_{model_name.lower().replace(' ', '_')}.png to {plots_dir}")
        plt.close(fig)


    # --- Task 1.4: Misclassified Patterns ---
    print("Generating misclassified patterns plots...")
    # Find indices for one False Positive (FP) and one False Negative (FN) sample
    # Using SMOTE LR results to find indices (SMOTE RFC had the same CM)
    true_labels = np.array(smote_lr_results["true_labels"])
    predicted_labels = np.array(smote_lr_results["predicted_labels"])

    # Find a False Positive (True=0, Predicted=1)
    fp_indices = np.where((true_labels == 0) & (predicted_labels == 1))[0]
    fp_index = fp_indices[0] if len(fp_indices) > 0 else None

    # Find a False Negative (True=1, Predicted=0)
    fn_indices = np.where((true_labels == 1) & (predicted_labels == 0))[0]
    fn_index = fn_indices[0] if len(fn_indices) > 0 else None

    if fp_index is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(TARGET_TWO_THETA_GRID, X_test[fp_index])
        ax.set_title(f'Example False Positive Pattern (True: {true_labels[fp_index]}, Predicted: {predicted_labels[fp_index]})')
        ax.set_xlabel('2-theta (degrees)')
        ax.set_ylabel('Intensity (Normalized)')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "plot_false_positive.png"))
        print(f"Saved plot_false_positive.png to {plots_dir}")
        plt.close(fig)
    else:
        print("Could not find a False Positive sample for plotting.")

    if fn_index is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(TARGET_TWO_THETA_GRID, X_test[fn_index])
        ax.set_title(f'Example False Negative Pattern (True: {true_labels[fn_index]}, Predicted: {predicted_labels[fn_index]})')
        ax.set_xlabel('2-theta (degrees)')
        ax.set_ylabel('Intensity (Normalized)')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "plot_false_negative.png"))
        print(f"Saved plot_false_negative.png to {plots_dir}")
        plt.close(fig)
    else:
        print("Could not find a False Negative sample for plotting.")


    print("\nPlot generation finished.")


if __name__ == "__main__":
    # This block executes when the script is run directly.
    create_plots()