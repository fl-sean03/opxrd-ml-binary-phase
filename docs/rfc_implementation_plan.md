# Plan: Implement and Evaluate RandomForestClassifier (Baseline and SMOTE)

This document outlines the plan to implement and evaluate a RandomForestClassifier (RFC) model for the pXRD phase identification project, comparing its performance on both the original imbalanced dataset and the SMOTE-augmented dataset.

## 1. Modify `src/train_model.py`

*   **Import:** Add `from sklearn.ensemble import RandomForestClassifier`.
*   **Integration:** Add new sections or functions within the existing script structure to handle RFC training and evaluation for both original and SMOTE data scenarios.
*   **Training (Baseline RFC):** Train a default `RandomForestClassifier` instance using the *original* (non-SMOTE) training data (`X_train`, `y_train`).
*   **Evaluation (Baseline RFC):** Evaluate the trained baseline RFC model on the corresponding validation set (`X_val`, `y_val`) using the same metrics currently employed (Accuracy, Precision, Recall, F1-score, Confusion Matrix).
*   **Results Storage (Baseline RFC):** Save the calculated evaluation metrics for the baseline RFC model to a new file: `results/evaluation_results_rf.json`.
*   **Training (SMOTE RFC):** Train a default `RandomForestClassifier` instance using the *SMOTE-augmented* training data (`X_train_smote`, `y_train_smote`).
*   **Evaluation (SMOTE RFC):** Evaluate the trained SMOTE RFC model on the *original* validation set (`X_val`, `y_val`) using the same metrics.
*   **Results Storage (SMOTE RFC):** Save the calculated evaluation metrics for the SMOTE RFC model to a new file: `results/evaluation_results_rf_smote.json`.

## 2. Update Documentation

*   **`TASK.md`:** Mark the relevant optional tasks as completed, noting that both original and SMOTE data were used for RFC evaluation.
*   **`docs/analysis_summary.md`:** Add a new section detailing the RFC results (both baseline and SMOTE versions). Include a comparative analysis discussing the performance of RFC relative to the baseline LR and SMOTE-LR models, focusing on key metrics across both dataset types.
*   **`.roo/memory-bank/activeContext.md`:** Update "Recent Changes" to include the implementation and evaluation of both RFC versions. Update "Next Steps" if applicable.
*   **`.roo/memory-bank/progress.md`:** Update "What Works" to include both RFC models. Update "Current Status" to reflect the completion of this optional enhancement.

## 3. Mermaid Diagram

```mermaid
graph TD
    A[Start: Implement RFC Task] --> B(Modify src/train_model.py);
    B --> B1(Import RFC);
    B --> B_Baseline(Baseline RFC);
        B_Baseline --> B2a(Train RFC on Original Data);
        B_Baseline --> B3a(Evaluate RFC on Original Val Data);
        B_Baseline --> B4a(Save Results to evaluation_results_rf.json);
    B --> B_SMOTE(SMOTE RFC);
        B_SMOTE --> B2b(Train RFC on SMOTE Data);
        B_SMOTE --> B3b(Evaluate RFC on Original Val Data);
        B_SMOTE --> B4b(Save Results to evaluation_results_rf_smote.json);
    B4a --> C(Update Documentation);
    B4b --> C;
    C --> C1(Update TASK.md);
    C --> C2(Update docs/analysis_summary.md);
    C --> C3(Update .roo/memory-bank/activeContext.md);
    C --> C4(Update .roo/memory-bank/progress.md);
    C --> D[End: RFC Implemented & Documented];